import torch
import torch.nn.functional as F
import math
from typing import Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class StepContext:
    """Encapsulates the intermediate state passed between processing steps."""
    sim: float
    valid_pull_mask: Optional[torch.Tensor] = None
    is_committed: Optional[torch.Tensor] = None
    best_idx: Optional[torch.Tensor] = None
    best_sim: Optional[torch.Tensor] = None
    masked_sim_matrix: Optional[torch.Tensor] = None
    margin_conf: Optional[torch.Tensor] = None

@dataclass
class AnchorContext:
    """Encapsulates the output of the AnchorTracker micro-geometry locking."""
    masked_sim_matrix: torch.Tensor
    best_sim: torch.Tensor
    margin: torch.Tensor
    best_idx: torch.Tensor
    anchor_connected: torch.Tensor
    hits: torch.Tensor

# ==============================================================================
# 1. DIFFUSION STATE (The State Machine)
# ==============================================================================

class DiffusionState:
    """Tracks ODE solver progress, persistent anchors, and radar hits across steps."""
    def __init__(self, expected_steps: int):
        self.reset(expected_steps)

    def reset(self, expected_steps: int):
        self.step_counter = 1
        self.last_ts = -1.0
        self.expected_steps = expected_steps
        
        self.ref_hits_master: Optional[torch.Tensor] = None
        self.step_hits_accum: Optional[torch.Tensor] = None
        
        self.ref_mask: Optional[torch.Tensor] = None
        self.last_sim = 0.0
        self.prev_sim_for_vel = 0.0
         
        self.commit_assign: Dict[str, torch.Tensor] = {}
        self.commit_hits: Dict[str, torch.Tensor] = {}
        self.persistent_life: Optional[torch.Tensor] = None
        self.persistent_anchors: Optional[torch.Tensor] = None
        
        self.H: Optional[int] = None
        self.W: Optional[int] = None
        self.fft_cache: Dict[tuple, dict] = {}
        self.od_face_mask: Optional[torch.Tensor] = None     
        self.od_face_mask_accum: Optional[torch.Tensor] = None  
        self.od_face_factor: Optional[torch.Tensor] = None      
        self.od_gen_start: Optional[int] = None
        self.od_gen_end: Optional[int] = None

    @property
    def progress(self) -> float:
        return min(1.0, (self.step_counter - 1) / max(self.expected_steps - 1, 1))

    @property
    def temporal_threshold(self) -> int:
        if self.expected_steps <= 4:
            return 2  
        return 3 if self.expected_steps <= 8 else max(3, round(self.expected_steps / 8.0))

    def update_from_kwargs(self, kwargs: dict, expected_steps: int):
        ts_val = kwargs["timestep"][0].item() if torch.is_tensor(kwargs["timestep"]) else kwargs["timestep"]
        
        if self.last_ts == -1.0 or ts_val > self.last_ts:
            self.reset(expected_steps)
        elif ts_val < self.last_ts:
            self._advance_step()
            
        self.last_ts = ts_val
        
        input_tensor = kwargs.get("input", None)
        if input_tensor is not None:
            self.H = input_tensor.shape[2]
            self.W = input_tensor.shape[3]

    def _advance_step(self):
        if self.step_hits_accum is not None:
            if self.ref_hits_master is None:
                self.ref_hits_master = self.step_hits_accum.clone()
            else:
                master = self.ref_hits_master
                curr = self.step_hits_accum
                dynamic_clamp = max(0.02, 0.07 - 0.02 * self.step_counter)
                master_norm = master / master.abs().mean(dim=-1, keepdim=True).clamp(min=dynamic_clamp)
                curr_norm = curr / curr.abs().mean(dim=-1, keepdim=True).clamp(min=dynamic_clamp)
                
                self.ref_hits_master = (master_norm * 0.70) + (curr_norm * 0.30)
        
        self.step_hits_accum = None
        self.ref_mask = None
        if self.od_face_mask_accum is not None:
            self.od_face_mask = self.od_face_mask_accum
        self.od_face_mask_accum = None
        
        if self.persistent_life is not None:
            self.persistent_life -= 1
            self.persistent_life = torch.clamp(self.persistent_life, min=0)
            self.persistent_anchors = self.persistent_life > 0
            
        self.step_counter += 1

# ==============================================================================
# 2. MASK PROCESSOR (Spatial & Masking Engine)
# ==============================================================================

class MaskProcessor:
    def __init__(self, src_mask: Optional[torch.Tensor]):
        self.src_mask = src_mask
        self._cache: Dict[int, torch.Tensor] = {}

    def get_1d_token_mask(self, count: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.src_mask is None: return None
        if count in self._cache: return self._cache[count].to(device)
            
        mh, mw = self.src_mask.shape[-2:]
        target = mh / max(mw, 1)
        best = (1, count)
        best_err = float("inf")
        limit = int(count ** 0.5) + 3
        
        for h in range(1, limit):
            if count % h == 0:
                w = count // h
                for hh, ww in ((h, w), (w, h)):
                    err = abs(hh / max(ww, 1) - target)
                    if err < best_err:
                        best_err = err
                        best = (hh, ww)
                        
        pooled = F.adaptive_avg_pool2d(self.src_mask[None, None], best).view(-1)
        mask_1d = pooled >= 0.5
        self._cache[count] = mask_1d
        return mask_1d.to(device)

# ==============================================================================
# 3. SALIENCY RADAR (Feature Isolation)
# ==============================================================================

class SaliencyRadar:
    @staticmethod
    def compute_causal_hits(matrix: torch.Tensor, B: int, R: int, contrast_floor: float) -> torch.Tensor:
        topk_v, topk_i = torch.topk(matrix, k=4, dim=-1)
        probs = torch.softmax(topk_v * 5.0, dim=-1)
        local_entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=-1)
        causal_weight = (1.0 - local_entropy / math.log(4)).clamp(min=0.0)
        return torch.zeros((B, R), device=matrix.device).scatter_add_(1, topk_i[..., 0], causal_weight)

    @classmethod
    def compute_identity_gain(cls, sim_matrix: torch.Tensor, masked_sim_radar: torch.Tensor, B: int, R: int, current_step: int, expected_steps: int, contrast_floor: float) -> torch.Tensor:
        global_hits = cls.compute_causal_hits(sim_matrix, B, R, contrast_floor)
        face_hits = cls.compute_causal_hits(masked_sim_radar, B, R, contrast_floor)
        ratio = face_hits / (global_hits + 1e-4)
        identity_gain_hits = torch.log1p(ratio)
        dynamic_clamp = max(0.02, 0.07 - 0.02 * current_step)
        return identity_gain_hits / identity_gain_hits.abs().mean(dim=-1, keepdim=True).clamp(min=dynamic_clamp)

# ==============================================================================
# 4. ANCHOR TRACKER (Micro-Geometry Locking)
# ==============================================================================

class AnchorTracker:
    @staticmethod
    def evaluate_and_update(state: DiffusionState, sim_matrix: torch.Tensor, user_mask: Optional[torch.Tensor],
                            face_isolation_strictness: float, hard_anchor_margin: float, confidence_gate: float,
                            contrast_floor: float, block_type: str, block_idx: int) -> AnchorContext:
        B, G, R = sim_matrix.shape
        neg_inf = torch.finfo(sim_matrix.dtype).min

        if state.ref_mask is None:
            total_hits = state.ref_hits_master if state.ref_hits_master is not None else torch.zeros((B, R), device=sim_matrix.device)
            mask_size = user_mask.sum().item() if user_mask is not None else R
            k_tokens = max(1, int(mask_size * face_isolation_strictness))
            
            hits_to_eval = torch.where(user_mask.unsqueeze(0), total_hits, torch.zeros_like(total_hits)) if user_mask is not None else total_hits
            _, top_indices = hits_to_eval.topk(k_tokens, dim=-1)
            mask = torch.zeros_like(total_hits, dtype=torch.bool).scatter_(1, top_indices, True)
            
            if state.persistent_anchors is not None:
                safe_anchors = state.persistent_anchors & user_mask.unsqueeze(0) if user_mask is not None else state.persistent_anchors
                mask = mask | safe_anchors
            state.ref_mask = mask

        r_mask = state.ref_mask.unsqueeze(1).expand(-1, G, -1)
        masked_sim_matrix = torch.where(r_mask, sim_matrix, torch.tensor(neg_inf, device=sim_matrix.device, dtype=sim_matrix.dtype))
        
        top2 = torch.topk(masked_sim_matrix, k=2, dim=-1).values
        best_sim = top2[..., 0]
        margin = (best_sim - top2[..., 1]).clamp(min=0) 
        best_idx = masked_sim_matrix.max(dim=-1).indices
        anchor_connected = torch.gather(state.ref_mask, 1, best_idx)
        
        cache_key = f"{block_type}_{block_idx}"
        prev_assign = state.commit_assign.get(cache_key)
        prev_hits = state.commit_hits.get(cache_key)
        
        if prev_assign is None or prev_assign.shape != best_idx.shape:
            hits = torch.ones_like(best_idx, dtype=torch.int16)
        else:
            if prev_hits is None: prev_hits = torch.zeros_like(best_idx, dtype=torch.int16)
            is_neighborhood = torch.abs(prev_assign - best_idx) <= 2
            hits = torch.where(is_neighborhood & (margin > hard_anchor_margin), (prev_hits + 1).clamp(max=100), torch.zeros_like(prev_hits))
                               
        state.commit_assign[cache_key] = best_idx.detach()
        state.commit_hits[cache_key] = hits.detach()
        
        return AnchorContext(
            masked_sim_matrix=masked_sim_matrix,
            best_sim=best_sim,
            margin=margin,
            best_idx=best_idx,
            anchor_connected=anchor_connected,
            hits=hits
        )

# ==============================================================================
# 5. FREQUENCY ROUTER (V1 Mapping + Custom Detailed Option)
# ==============================================================================

class FrequencyRouter:
    @staticmethod
    def split_and_route(uncommitted_delta: torch.Tensor, H: Optional[int], W: Optional[int],
                        contrast_floor: float, block_type: str, fft_cache: dict, routing_mode: str) -> torch.Tensor:
        
        if routing_mode == "Raw (No Filter)":
            return uncommitted_delta

        if H is None or W is None:
            return uncommitted_delta

        B_d, T_d, D_d = uncommitted_delta.shape
        p_h, p_w = None, None
        
        if T_d == H * W: p_h, p_w = H, W
        elif T_d == (H // 2) * (W // 2): p_h, p_w = H // 2, W // 2
        elif T_d == (H // 4) * (W // 4): p_h, p_w = H // 4, W // 4
        else:
            side = int(math.sqrt(T_d))
            if side * side == T_d: p_h, p_w = side, side
            
        if p_h is None or p_w is None:
            return uncommitted_delta

        cache_key = (p_h, p_w, routing_mode)
        if cache_key not in fft_cache:
            u = torch.fft.fftfreq(p_h, device=uncommitted_delta.device)
            v = torch.fft.rfftfreq(p_w, device=uncommitted_delta.device)
            U, V = torch.meshgrid(u, v, indexing='ij')
            R_rad = torch.sqrt(U**2 + V**2) 
            
            if routing_mode == "Smooth (Legacy V1)":
                cutoff = 0.40
                taper_width = 0.10
            else:
                cutoff = 0.55 - (contrast_floor * 0.8)
                taper_width = max(0.01, 0.15 - (contrast_floor * 0.2))
                
            R_scaled = ((R_rad - (cutoff - taper_width)) / taper_width).clamp(0.0, 1.0)
            mask_lf = 0.5 * (1.0 + torch.cos(math.pi * R_scaled))
            mask_hf = 1.0 - mask_lf
            mid_cutoff = 0.60
            mid_taper = 0.10
            R_mid = ((R_rad - (mid_cutoff - mid_taper)) / mid_taper).clamp(0.0, 1.0)
            mask_mid = 0.5 * (1.0 + torch.cos(math.pi * R_mid))

            fft_cache[cache_key] = {
                "mask_lf": mask_lf.unsqueeze(0).unsqueeze(0),
                "mask_hf": mask_hf.unsqueeze(0).unsqueeze(0),
                "mask_mid": mask_mid.unsqueeze(0).unsqueeze(0)
            }

        cached_masks = fft_cache[cache_key]
        spatial_delta = uncommitted_delta.view(B_d, p_h, p_w, D_d).permute(0, 3, 1, 2).float()
        freq = torch.fft.rfft2(spatial_delta, dim=(-2, -1))
        
        if routing_mode == "Smooth (Legacy V1)":
            freq_filtered = freq * cached_masks["mask_lf"]

        elif routing_mode == "Detailed (Split Frequencies)":
            if block_type == "double":
                hf_bleed = 0.15
                layout_mask = cached_masks["mask_lf"] + (hf_bleed * cached_masks["mask_hf"])
                freq_filtered = freq * layout_mask
            elif block_type == "single":
                lf_retention = 0.30 
                asymmetric_mask = cached_masks["mask_hf"] + (lf_retention * cached_masks["mask_lf"])
                freq_filtered = freq * asymmetric_mask
            else:
                return uncommitted_delta
        
        elif routing_mode == "Pure Frequency Split":
            if block_type == "double":
                freq_filtered = freq * cached_masks["mask_lf"] 
            elif block_type == "single":
                freq_filtered = freq * cached_masks["mask_hf"]
            else:
                return uncommitted_delta

        elif routing_mode == "Band-Pass (Identity)":
            if block_type == "double":
                freq_filtered = freq * cached_masks["mask_lf"]
            elif block_type == "single":
                freq_filtered = freq * cached_masks["mask_mid"]
            else:
                return uncommitted_delta
        else:
            return uncommitted_delta
            
        spatial_filtered = torch.fft.irfft2(freq_filtered, s=(p_h, p_w), dim=(-2, -1))
        routed_delta = spatial_filtered.permute(0, 2, 3, 1).reshape(B_d, T_d, D_d).to(uncommitted_delta.dtype)
        
        return routed_delta

# ==============================================================================
# 5b. OVERDRIVE (Reference-Free Residual Amplification)
# ==============================================================================
def _overdrive_residual(delta: torch.Tensor, amp: float, use_gate: bool = True, ceiling_mult: float = 6.0, region: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Amplify a block's residual contribution, focused on salient tokens, with a tanh safety ceiling.
    region: optional [B,T,1] in [0,1] crossfading each token between original (0) and amplified (1)."""
    if abs(amp - 1.0) < 1e-3:
        return delta
    base = delta.float()
    x = base
    boost = amp - 1.0
    if use_gate:
        e = x.abs().mean(dim=-1, keepdim=True)
        z = (e - e.mean(dim=1, keepdim=True)) / e.std(dim=1, keepdim=True).clamp(min=1e-8)
        eff = 1.0 + boost * (1.0 + 0.5 * torch.tanh(z))   
    else:
        eff = 1.0 + boost
    x = x * eff
    ratio = base.abs().mean().clamp(min=1e-6) / x.abs().mean().clamp(min=1e-6)
    x = x * (ratio ** 0.5) 
    ceil = base.abs().max().clamp(min=1e-6) * ceiling_mult
    x = torch.tanh(x / ceil) * ceil
    if region is not None:
        x = base + (x - base) * region.to(x.dtype)
    return x.to(delta.dtype)

# ==============================================================================
# 6. ATTENTION ORCHESTRATOR (The Conductor)
# ==============================================================================

class AttentionOrchestrator:
    def __init__(self, state: DiffusionState, mask_processor: MaskProcessor, params: dict, block_sets: tuple):
        self.state = state
        self.mask_processor = mask_processor
        self.p = params
        self.d_set, self.s_set, self.r_set = block_sets

    def process(self, attn: torch.Tensor, extra_options: dict) -> torch.Tensor:
        block_type = extra_options.get("block_type", "double")
        block_idx = extra_options.get("block_index", -1)
        current_step = max(1, self.state.step_counter)
        
        is_d_target = (block_type == "double" and block_idx in self.d_set)
        is_s_target = (block_type == "single" and block_idx in self.s_set)
        is_radar_block = (block_idx in self.r_set)

        if not (is_d_target or is_s_target or is_radar_block):
            return attn

        img_slice = extra_options.get("img_slice", None)
        ref_tokens_list = extra_options.get("reference_image_num_tokens", [])
        if img_slice is None or not ref_tokens_list: return attn

        txt_len = int(img_slice[0])
        total_seq = int(img_slice[1])
        total_ref = sum(ref_tokens_list)
        gen_start, gen_end, ref_start = txt_len, total_seq - total_ref, total_seq - total_ref
        
        if gen_end <= gen_start or ref_start >= total_seq: return attn
        self.state.od_gen_start = gen_start
        self.state.od_gen_end = gen_end

        gen_f = attn[:, gen_start:gen_end].float()
        ref_f = attn[:, ref_start:total_seq].float()

        target_likeness_metric = self.p["target_likeness_metric"]
        soft_blend_k = self.p["soft_blend_k"]
        dynamic_text_balancing = self.p["dynamic_text_balancing"]

        with torch.no_grad():
            gen_norm = F.normalize(gen_f, dim=-1)
            ref_norm = F.normalize(ref_f, dim=-1)
            sim_matrix = torch.bmm(gen_norm, ref_norm.transpose(1, 2)) 
            B, G, R = sim_matrix.shape
            neg_inf = torch.finfo(sim_matrix.dtype).min
            
            user_mask = self.mask_processor.get_1d_token_mask(R, sim_matrix.device)
            masked_sim_radar = sim_matrix
            if user_mask is not None:
                mask_exp = user_mask.unsqueeze(0).unsqueeze(1).expand(B, G, -1)
                masked_sim_radar = torch.where(mask_exp, sim_matrix, torch.tensor(neg_inf, device=sim_matrix.device, dtype=sim_matrix.dtype))

            if is_radar_block and (current_step <= 2):
                identity_gain_hits = SaliencyRadar.compute_identity_gain(
                    sim_matrix, masked_sim_radar, B, R, current_step, self.state.expected_steps, self.p["contrast_and_texture_floor"]
                )
                if self.state.step_hits_accum is None: self.state.step_hits_accum = identity_gain_hits
                elif self.state.step_hits_accum.shape == identity_gain_hits.shape: self.state.step_hits_accum += identity_gain_hits
                else: self.state.step_hits_accum = identity_gain_hits

            if not (is_d_target or is_s_target):
                return attn

            if current_step == 1:
                ctx = self._process_step_1(
                    sim_matrix, masked_sim_radar, user_mask, B, G, neg_inf, is_radar_block, block_type
                )
            else:
                ctx = self._process_step_n(
                    sim_matrix, user_mask, B, G, R, neg_inf, block_type, block_idx
                )

        boost_r, boost_t = self._calculate_pid_boosts(ctx, target_likeness_metric, current_step, block_type)
        
        attn_out, safe_weight_mean = self._construct_and_apply_deltas(
            attn, gen_f, ref_f, ctx, boost_r, soft_blend_k, block_type, gen_start, gen_end
        )
        
        if (boost_r > 0.0 or boost_t > 0.0) and (dynamic_text_balancing or self.p["background_text_strength"] > 0.0):
            attn_out = self._apply_text_balancing(
                attn_out, attn, txt_len, current_step, block_type, dynamic_text_balancing, boost_t, safe_weight_mean
            )

        return attn_out

    def _process_step_1(self, sim_matrix, masked_sim_radar, user_mask, B, G, neg_inf, is_radar_block, block_type) -> StepContext:
        if block_type == "double" and user_mask is not None:
            soft_subject_mask = F.avg_pool1d(user_mask.float().view(1, 1, -1), kernel_size=17, stride=1, padding=8).squeeze() > 0.05
            mask_exp_soft = soft_subject_mask.unsqueeze(0).unsqueeze(1).expand(B, G, -1)
            masked_sim_matrix = torch.where(mask_exp_soft, sim_matrix, sim_matrix * 0.15)
        elif block_type == "single":
            masked_sim_matrix = masked_sim_radar if is_radar_block else sim_matrix
        else:
            masked_sim_matrix = sim_matrix
            
        best_sim = masked_sim_matrix.max(dim=-1).values
        
        valid_sims = best_sim[best_sim > neg_inf]
        sim = valid_sims.mean().item() if valid_sims.numel() > 0 else 0.0
        
        self.state.last_sim = sim 
        
        return StepContext(sim=sim)

    def _process_step_n(self, sim_matrix, user_mask, B, G, R, neg_inf, block_type, block_idx) -> StepContext:
        anchor_ctx = AnchorTracker.evaluate_and_update(
            self.state, sim_matrix, user_mask, self.p["face_isolation_strictness"], 
            self.p["hard_anchor_margin"], self.p["confidence_gate"], self.p["contrast_and_texture_floor"],
            block_type, block_idx
        )
        
        valid_pull_mask = (anchor_ctx.best_sim > self.p["contrast_and_texture_floor"]) & anchor_ctx.anchor_connected
        face_now = valid_pull_mask.float()
        if self.state.od_face_mask_accum is None or self.state.od_face_mask_accum.shape != face_now.shape:
            self.state.od_face_mask_accum = face_now
        else:
            self.state.od_face_mask_accum = torch.maximum(self.state.od_face_mask_accum, face_now)
        
        temporal_threshold = self.state.temporal_threshold
        is_committed = (anchor_ctx.hits >= temporal_threshold) & (anchor_ctx.margin > self.p["hard_anchor_margin"]) & valid_pull_mask
        margin_conf = torch.sigmoid((anchor_ctx.margin - 0.08) * 12.0)
        
        margin_thresh = 0.75 
        sim_thresh = 0.35
        strong_commit = is_committed & (margin_conf > margin_thresh) & (anchor_ctx.best_sim > sim_thresh)
        
        if strong_commit.any():
            if self.state.persistent_life is None:
                self.state.persistent_life = torch.zeros((B, R), dtype=torch.int8, device=sim_matrix.device)
            
            committed_ref_idx = anchor_ctx.best_idx[strong_commit].unsqueeze(0)
            new_life = torch.full_like(committed_ref_idx, 3, dtype=torch.int8)
            self.state.persistent_life.scatter_reduce_(1, committed_ref_idx, new_life, reduce="amax", include_self=True)
            self.state.persistent_anchors = self.state.persistent_life > 0

        weighted = anchor_ctx.best_sim * margin_conf
        valid_sims = weighted[valid_pull_mask]
        sim = valid_sims.mean().item() if valid_sims.numel() > 0 else 0.0
        self.state.last_sim = sim
        
        return StepContext(
            sim=sim,
            valid_pull_mask=valid_pull_mask,
            is_committed=is_committed,
            best_idx=anchor_ctx.best_idx,
            best_sim=anchor_ctx.best_sim,
            masked_sim_matrix=anchor_ctx.masked_sim_matrix,
            margin_conf=margin_conf
        )

    def _calculate_pid_boosts(self, ctx: StepContext, target_metric: float, step: int, block_type: str) -> Tuple[float, float]:
        gap = target_metric - ctx.sim 
        prev_sim = self.state.prev_sim_for_vel if self.state.prev_sim_for_vel != 0.0 else ctx.sim
        sim_velocity = ctx.sim - prev_sim
        self.state.prev_sim_for_vel = ctx.sim

        f = self._calculate_fade(self.state.progress, self.p["boost_fade_curve"])
        step_attenuator = 1.0 - (f * 0.45)

        boost_r, boost_t = 0.0, 0.0
        if step > 1:
            if gap > 0 and self.p["identity_strength"] > 0.0:
                active_scale = self.p["identity_strength"]
                if block_type == "double":
                    active_scale *= 0.15
                    
                drift_mult = 1.0
                if sim_velocity < 0.0:
                    drift_mult = 1.0 + (abs(sim_velocity) * 15.0)
                boost_r = min(10.0, active_scale * gap * step_attenuator * drift_mult)
            elif gap < 0 and self.p["background_text_strength"] > 0.0:
                boost_t = min(10.0, self.p["background_text_strength"] * abs(gap))
                
        return boost_r, boost_t

    def _construct_and_apply_deltas(self, attn: torch.Tensor, gen_f: torch.Tensor, ref_f: torch.Tensor, 
                                     ctx: StepContext, boost_r: float, soft_k: int, 
                                    block_type: str, gen_start: int, gen_end: int) -> Tuple[torch.Tensor, float]:
        if boost_r <= 0.0:
            return attn, 0.0

        attn_out = attn.clone()
        
        hard_ref = torch.gather(ref_f, 1, ctx.best_idx.unsqueeze(-1).expand(-1, -1, ref_f.shape[-1]))

        if soft_k <= 1:
            soft_ref = hard_ref
        else:
            topk_sims, topk_idx = torch.topk(ctx.masked_sim_matrix, k=soft_k, dim=-1)
            temp = 0.25  
            attn_w = F.softmax(topk_sims / temp, dim=-1)
            B_t, G_t, k_val = topk_idx.shape
            flat_idx_soft = topk_idx.reshape(B_t, G_t * k_val).unsqueeze(-1).expand(-1, -1, ref_f.shape[-1])
            soft_ref = (attn_w.unsqueeze(-1) * torch.gather(ref_f, 1, flat_idx_soft).reshape(B_t, G_t, k_val, ref_f.shape[-1])).sum(dim=2)

       
        confidence = torch.sigmoid((ctx.best_sim - self.p["contrast_and_texture_floor"]) * 8.0)
        surgical_mask = (ctx.valid_pull_mask & (ctx.margin_conf > self.p["confidence_gate"])).to(attn.dtype).unsqueeze(-1)
        safe_weight = torch.clamp(boost_r * confidence.unsqueeze(-1), max=1.0)
        safe_weight_mean = safe_weight.mean().item() if boost_r > 0.0 else 0.0
        
        uncommitted_delta = (soft_ref - gen_f) * safe_weight * surgical_mask
        
        committed_delta = (hard_ref - gen_f) * min(boost_r, 2.5)
        uncommitted_delta = FrequencyRouter.split_and_route(
            uncommitted_delta, self.state.H, self.state.W, 
            self.p["contrast_and_texture_floor"], block_type, self.state.fft_cache,
            self.p["frequency_filtering"]
        )

        delta = torch.where(ctx.is_committed.unsqueeze(-1), committed_delta, uncommitted_delta)
        attn_out[:, gen_start:gen_end] += delta.to(attn.dtype)
            
        return attn_out, safe_weight_mean

    def _apply_text_balancing(self, attn_out: torch.Tensor, original_attn: torch.Tensor, txt_len: int, step: int, block_type: str, dynamic_balancing: bool, boost_t: float, safe_weight_mean: float) -> torch.Tensor:
        is_layout_phase = (block_type == "double")
        is_texture_phase = (block_type == "single" and step >= 3)
        
        if not (is_layout_phase or is_texture_phase):
            return attn_out

        f = self._calculate_fade(self.state.progress, self.p["boost_fade_curve"])
        active_boost = 0.0
        if dynamic_balancing and self.p["background_text_strength"] > 0.0:
            txt_allowance = 1.0 - min(safe_weight_mean, 1.0)
            raw_txt_boost = self.p["background_text_strength"] * txt_allowance
            if raw_txt_boost > 0.0:
                active_boost = raw_txt_boost if is_layout_phase else raw_txt_boost * f
        elif boost_t > 0.0:
            active_boost = boost_t if is_layout_phase else boost_t * f
            
        if active_boost <= 0.0:
            return attn_out
            
        if attn_out.data_ptr() == original_attn.data_ptr():
            attn_out = attn_out.clone()
            
        attn_out[:, :txt_len] += attn_out[:, :txt_len] * active_boost
        return attn_out

    @staticmethod
    def _calculate_fade(progress: float, curve_type: str) -> float:
        if curve_type == "Linear": return progress
        elif curve_type == "Smooth": return (1.0 - math.cos(progress * math.pi)) / 2.0
        elif curve_type == "Ease-In": return progress ** 2
        elif curve_type == "Ease-Out": return 1.0 - (1.0 - progress) ** 2
        return progress

# ==============================================================================
# 7. COMFYUI NODE INTERFACE
# ==============================================================================

class FluxIDAutoAdjusterv2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "ui_mode": (["Basic", "Expert"], {"default": "Basic", "tooltip": "UI only — does not affect generation. Basic = core controls, Expert = all."}),
                "layout_blocks": ("STRING", {"default": "3-7", "tooltip": "Double blocks that receive the identity pull as a low-frequency (layout) signal. Controls head placement, pose and overall composition of the subject — not fine detail. More/earlier blocks = stronger control over where and how the face sits, but can stiffen the composition. The double-block pull is intentionally gentle."}),
                "identity_blocks": ("STRING", {"default": "8-19", "tooltip": "Single blocks that carry the actual likeness. This is where identity is injected; the upper range (~15-19) builds skin and feature detail. Widening or pushing later = sharper resemblance, but overdoing it bakes in a stiff, 'pasted-on' look. Empirically best around 8-19."}),
                "saliency_scan_blocks": ("STRING", {"default": "6-23", "tooltip": "Blocks the radar reads on the first 1-2 steps to LOCATE the face on the canvas (no injection happens here). A wide range gives robust detection; too narrow and the face can be mis-located, weakening every downstream anchor and pull."}),
                "frequency_filtering": (["Raw (No Filter)", "Smooth (Legacy V1)", "Detailed (Split Frequencies)", "Pure Frequency Split", "Band-Pass (Identity)"], {"default": "Band-Pass (Identity)", 
                    "tooltip": "Which frequency bands of the identity signal reach the model. Band-Pass (recommended): LF layout to Double, low+mid structural band to Single, top octave left to FLUX for native pores — best likeness + skin. Smooth = exact V1 low-pass (softer, safest). Detailed / Pure Split = push more high-freq detail (sharper but more artifact-prone). Raw = no filtering (rawest, can look synthetic)."
                }),                
                "total_sampling_steps": ("INT", {"default": 4, "min": 1, "max": 100, "tooltip": "MUST equal your KSampler step count. Drives the fade schedule, the step at which anchors commit, and overdrive's ramp. If it doesn't match, identity injection is mistimed (too early or too late) and likeness suffers."}),
                "boost_fade_curve": (["Linear", "Smooth", "Ease-In", "Ease-Out"], {"default": "Ease-In", "tooltip": "Shape of how the identity pull tapers across steps. Ease-In holds full strength through early/mid steps then releases late (lets identity set, then hands texture to FLUX) — good default. Ease-Out releases early (gentler, more native). Linear = even taper. Smooth = soft S-curve."}),
                "identity_strength": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 3.0, "step": 0.05, "tooltip": "Master gain on the identity pull. Higher = stronger resemblance but risks a stiff, waxy, over-saturated or 'stuck-on' face and color shifts. Lower = subtler and more natural but weaker likeness. ~1.5 is a balanced start."}),
                "background_text_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 10.0, "step": 0.05, "tooltip": "How hard the prompt/background attention is reasserted once the face is satisfied (or throttled in via dynamic balancing). Higher = better scene/prompt adherence and richer background, but competes with identity. 0 disables all text balancing."}),
                "dynamic_text_balancing": ("BOOLEAN", {"default": True, "tooltip": "Auto-throttles background_text_strength down while the face is still working hard, then lets it back up once likeness is locked, so the prompt doesn't fight identity early. No effect if background_text_strength is 0."}),
                "target_likeness_metric": ("FLOAT", {"default": 0.35, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "The cosine-similarity goal the pull aims for (gap = target − current). 0.35 is about ideal for Flux. Higher demands a closer match, forcing aggressive pulling that stiffens and can distort; lower eases off for a softer, more natural result with looser likeness."}),
                "soft_blend_k": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1, "tooltip": "How many top reference matches each generated token blends from. 1 = hard snap to the single best match (crispest, V1 default). 3+ averages several = softer, smoother transfer with fewer hard edges, at some cost to exactness."}),
                "face_isolation_strictness": ("FLOAT", {"default": 1.00, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Fraction of the masked face used as the anchor set. 1.0 uses the whole masked region. Lower values keep only the most salient tokens (eyes/nose/mouth), tightening identity to core features but dropping cheeks/jaw/hair from the lock — sharper core ID, less coverage."}),
                "confidence_gate": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Minimum match confidence (margin) before a token is pulled. Higher = only very certain matches transfer → cleaner, fewer smears, but weaker coverage. Lower = pulls uncertain matches too → stronger, broader transfer but more risk of artifacts/bleed."}),
                "hard_anchor_margin": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.20, "step": 0.01, "tooltip": "How dominant a match must be (best vs runner-up) to register as a hit and eventually commit. Higher = stricter, only unambiguous matches lock → very stable, fewer commits. Lower = more tokens commit → tighter lock but risk of jittery or wrong anchors."}),
                "contrast_and_texture_floor": ("FLOAT", {"default": 0.18, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Similarity floor below which regions are left untouched, and the gate for contrast/texture handling. Higher = only strong-match areas are pulled → protects background and preserves native FLUX texture/contrast. Lower = touches weaker matches too → broader transfer but can flatten texture and wash out contrast."}),
                "apply_overdrive": ("BOOLEAN", {"default": False, "tooltip": "Reference-free amplification of each block's OWN residual on the selected blocks — boosts detail/realism/punch without using the reference. Pulled back inside the committed face via overdrive_face_impact so identity stays coherent. Start with a low strength."}),
                "overdrive_double_blocks": ("STRING", {"default": "", "tooltip": "Which double blocks overdrive amplifies. Leave empty: doubles drive composition, and amplifying them tends to cause ghosting or duplicated features. Only add (e.g. 0-2) if you deliberately want stronger global structure."}),
                "overdrive_single_blocks": ("STRING", {"default": "19-23", "tooltip": "Which single blocks overdrive amplifies. The upper range (~19-23) is where skin, texture and fidelity live — this is where the photorealism gain comes from. Widen for more crunch/detail."}),
                "overdrive_face_impact": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "How much overdrive reaches the committed face. 1.0 = full energy everywhere (incl. face), 0.0 = hard mask (face fully excluded). ~0.2 = light touch keeps skin/light coherent without disturbing identity."}),
                "overdrive_strength": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Peak residual multiplier at step 1, eased to 1.0 by the final step. Higher = punchier detail and contrast, but too high crunches/over-sharpens and can disturb skin. The effective gain is intentionally softened internally, so push gradually; start ~1.1-1.2."}),
            },
            "optional": {
                "subject_mask": ("MASK",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Advanced/Model Bending"

    @staticmethod
    def _parse_to_set(s: str) -> Set[int]:
        if not s or not s.strip(): return set()
        blocks = set()
        for part in s.split(','):
            part = part.strip()
            if not part: continue
            try:
                if '-' in part:
                    start_str, end_str = part.split('-')
                    if not start_str or not end_str: continue 
                    start, end = int(start_str), int(end_str)
                    blocks.update(range(start, end + 1))
                else:  
                    blocks.add(int(part))
            except ValueError:
                continue 
        return blocks

    def apply(self, model, layout_blocks, identity_blocks, saliency_scan_blocks, frequency_filtering, 
              total_sampling_steps, boost_fade_curve, identity_strength, background_text_strength, 
              dynamic_text_balancing, target_likeness_metric, soft_blend_k, face_isolation_strictness, confidence_gate, hard_anchor_margin, contrast_and_texture_floor,
              apply_overdrive=False, overdrive_double_blocks="", overdrive_single_blocks="16-23", overdrive_face_impact=0.20, overdrive_strength=1.15,
              subject_mask=None, ui_mode="Basic"):
        
        m = model.clone()
        
        _src_mask = None
        if subject_mask is not None:
            mk = subject_mask
            if mk.dim() == 4: mk = mk[0, 0]
            elif mk.dim() == 3: mk = mk[0]
            _src_mask = mk.detach().float()
            
        mask_processor = MaskProcessor(_src_mask)
        state = DiffusionState(total_sampling_steps)
        
        d_set = self._parse_to_set(layout_blocks)
        s_set = self._parse_to_set(identity_blocks)
        r_set = self._parse_to_set(saliency_scan_blocks)
        
        params = {
            "frequency_filtering": frequency_filtering,
            "boost_fade_curve": boost_fade_curve,
            "identity_strength": identity_strength,
            "background_text_strength": background_text_strength,
            "dynamic_text_balancing": dynamic_text_balancing,
            "target_likeness_metric": target_likeness_metric,
            "soft_blend_k": soft_blend_k,
            "face_isolation_strictness": face_isolation_strictness,
            "confidence_gate": confidence_gate,
            "hard_anchor_margin": hard_anchor_margin,
            "contrast_and_texture_floor": contrast_and_texture_floor
        }
        orchestrator = AttentionOrchestrator(state, mask_processor, params, (d_set, s_set, r_set))
        prev_wrapper = m.model_options.get("model_function_wrapper", None)

        od_active = bool(apply_overdrive) and overdrive_strength > 1.0
        od_d_set = self._parse_to_set(overdrive_double_blocks)
        od_s_set = self._parse_to_set(overdrive_single_blocks)
        od_buffer: Dict[str, Any] = {}

        def _od_amp() -> float:
            return 1.0 + (overdrive_strength - 1.0) * (1.0 - state.progress)

        def _od_region(ref):
            b_dim, g_dim = ref.shape[0], ref.shape[1]
            ff = state.od_face_factor
            if ff is not None and ff.shape[0] == b_dim and ff.shape[1] == g_dim:
                return ff
            return torch.full((b_dim, g_dim, 1), float(overdrive_face_impact),
                              device=ref.device, dtype=ref.dtype)

        def _od_dpre(module, args, kw):
            img = kw.get("img", args[0] if len(args) > 0 else None)
            if img is not None:
                od_buffer[f"d{module._od_idx}"] = img.detach().clone()

        def _od_dpost(module, args, kw, output):
            base = od_buffer.pop(f"d{module._od_idx}", None)
            if base is None:
                return output
            o = output[0]
            delta = o - base
            amp = _od_amp()
            gs, ge = state.od_gen_start, state.od_gen_end
            if gs is not None and ge is not None and o.shape[1] >= (ge - gs):
                gc = ge - gs
                g_d = delta[:, :gc, :]
                mod = _overdrive_residual(g_d, amp, region=_od_region(g_d))
                o = o.clone()
                o[:, :gc, :] += (mod - g_d)
            else:  
                o = o + (_overdrive_residual(delta, amp) - delta)
            return (o,) + tuple(output[1:])

        def _od_spre(module, args, kw):
            x = kw.get("x", args[0] if len(args) > 0 else None)
            if x is not None:
                od_buffer[f"s{module._od_idx}"] = x.detach().clone()

        def _od_spost(module, args, kw, output):
            base = od_buffer.pop(f"s{module._od_idx}", None)
            if base is None:
                return output
            is_tuple = isinstance(output, tuple)
            o = output[0] if is_tuple else output
            delta = o - base
            amp = _od_amp()
            gs, ge = state.od_gen_start, state.od_gen_end
            if gs is not None and ge is not None and o.shape[1] >= ge:
                g_d = delta[:, gs:ge, :]
                mod = _overdrive_residual(g_d, amp, region=_od_region(g_d))
                o = o.clone()
                o[:, gs:ge, :] += (mod - g_d)
            else:  
                tl = int(od_buffer.get("__txt_len__", 512))
                if o.shape[1] > tl:
                    i_d = delta[:, tl:, :]
                    o = o.clone()
                    o[:, tl:, :] += (_overdrive_residual(i_d, amp) - i_d)
                else:
                    o = o + (_overdrive_residual(delta, amp) - delta)
            return ((o,) + tuple(output[1:])) if is_tuple else o

        def unet_wrapper(apply_model, kwargs):
            state.update_from_kwargs(kwargs, total_sampling_steps)
            hooks = []
            if od_active:
                od_buffer.clear()
                c = kwargs.get("c", {}) or {}
                ck = "c_crossattn" if "c_crossattn" in c else ("context" if "context" in c else "txt")
                od_buffer["__txt_len__"] = c[ck].shape[1] if ck in c and torch.is_tensor(c[ck]) else 512
                fm = state.od_face_mask
                if fm is not None:
                    fm = fm.float()
                    fm = F.avg_pool1d(fm.unsqueeze(1), kernel_size=9, stride=1, padding=4).squeeze(1).clamp(0.0, 1.0)
                    state.od_face_factor = (1.0 - (1.0 - overdrive_face_impact) * fm).unsqueeze(-1)
                else:
                    state.od_face_factor = None
                dm = m.model.diffusion_model
                for i, b in enumerate(getattr(dm, "double_blocks", [])):
                    if i not in od_d_set: continue
                    b._od_idx = i
                    hooks.append(b.register_forward_pre_hook(_od_dpre, with_kwargs=True))
                    hooks.append(b.register_forward_hook(_od_dpost, with_kwargs=True))
                for i, b in enumerate(getattr(dm, "single_blocks", [])):
                    if i not in od_s_set: continue
                    b._od_idx = i
                    hooks.append(b.register_forward_pre_hook(_od_spre, with_kwargs=True))
                    hooks.append(b.register_forward_hook(_od_spost, with_kwargs=True))
            try:
                if prev_wrapper is not None:
                    return prev_wrapper(apply_model, kwargs)
                return apply_model(kwargs["input"], kwargs["timestep"], **kwargs.get("c", {}))
            finally:
                for h in hooks:
                    h.remove()

        def output_patch(attn, extra_options):
            return orchestrator.process(attn, extra_options)

        m.set_model_unet_function_wrapper(unet_wrapper)
        m.set_model_attn1_output_patch(output_patch)
        return (m,)

NODE_CLASS_MAPPINGS = {"FluxIDAutoAdjusterv2": FluxIDAutoAdjusterv2}
NODE_DISPLAY_NAME_MAPPINGS = {"FluxIDAutoAdjusterv2": "FLUX Identity Adjuster (v2)"}