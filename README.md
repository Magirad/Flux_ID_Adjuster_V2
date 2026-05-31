"""# FLUX Identity Adjuster (V2)

The node has been created through vibe coding. The main objective is for identity consistency for FLUX.2 klein 9b models in ComfyUI.

The default values are fixed for the best result, but I have only tested it for limited samples, so look for your own settings.

### Parameter Breakdown

| Parameter | Default | Description & Visual Impact |
| :--- | :---: | :--- |
| **`ui_mode`** | `Basic` | Toggles the UI layout (does not affect generation). Basic = core controls, Expert = reveals all advanced settings. |
| **`layout_blocks`** | `3-7` | **D-Blocks.** Establishes global layout and integrates the subject into the background safely. Double blocks receive the identity pull as a low-frequency signal, controlling head placement and pose without altering fine detail. |
| **`identity_blocks`** | `8-19` | **S-Blocks.** Synthesizes high-frequency identity, facial micro-geometry, and photorealism. Widening or pushing later equals sharper resemblance, but overdoing it creates a stiff, "pasted-on" look. |
| **`saliency_scan_blocks`** | `6-23` | The blocks the radar reads on the first 1-2 steps to dynamically isolate the face from the background. Too narrow of a range can mis-locate the face, weakening downstream anchors. |
| **`frequency_filtering`** | `Band-Pass (Identity)` | **REPLACES `photorealistic_smoothing`.** Controls which frequency bands reach the model.<br>• **Band-Pass (Recommended):** Best likeness + skin.<br>• **Smooth (Legacy V1):** Softer, safest low-pass.<br>• **Detailed / Pure Split:** Pushes high-freq detail (sharper, artifact-prone).<br>• **Raw (No Filter):** Rawest transfer, preserves brushstrokes but can look synthetic. |
| **`total_sampling_steps`** | `4` | **CRITICAL:** Must match your KSampler steps! Syncs the internal fade schedule, anchor commit steps, and overdrive ramp. |
| **`boost_fade_curve`** | `Ease-In` | Shape of how the identity pull tapers across steps. *Ease-In* holds full strength early, then hands texture over to FLUX in late steps. *Ease-Out* releases early for a gentler, more native look. |
| **`identity_strength`** | `1.5` | Master gain on the identity pull. Higher values force a stronger resemblance but risk a stiff, waxy, or stuck-on face. Lower is subtler and more natural. |
| **`background_text_strength`** | `0.6` | Amplifies your text prompt once the face is satisfied. Higher equals better scene adherence but competes with identity. Set to `0` to disable text balancing. |
| **`dynamic_text_balancing`** | `TRUE` | Auto-throttles `background_text_strength` down while the face is working hard, allowing it back up once likeness is locked so the prompt doesn't fight early identity formation. |
| **`target_likeness_metric`** | `0.35` | The cosine-similarity goal the pull aims for. `0.35` is mathematically ideal for Flux. Pushing higher forces aggressive pulling; lower allows a softer, more natural result. |
| **`soft_blend_k`** | `1` | How many top reference matches each generated token blends from. `1` = hard snap to the single best match (crispest). `3+` = softer, smoother transfer with fewer hard edges. |
| **`face_isolation_strictness`** | `1.0` | Fraction of the masked face used as the anchor set. `1.0` uses the whole region. Lower values keep only the most salient tokens (eyes/nose/mouth), tightening identity to core features. |
| **`confidence_gate`** | `0.15` | Minimum match confidence (margin) before a token is pulled. Higher = fewer smears but weaker coverage. Lower = pulls uncertain matches too, increasing risk of artifacts. |
| **`hard_anchor_margin`** | `0.06` | How dominant a match must be (best vs runner-up) to permanently lock. Higher = stricter lock, fewer commits. Lower = tighter lock but risk of jittery anchors. |
| **`contrast_and_texture_floor`** | `0.18` | Similarity floor below which regions are untouched. Higher protects the background and native texture. Lower touches weaker matches but can flatten texture and wash out contrast. |
| **`apply_overdrive`** | `False` | **NEW.** Reference-free amplification of block residuals. Boosts detail, realism, and punch without using the reference. Start with a low strength. |
| **`overdrive_double_blocks`** | *(Empty)* | Which double blocks overdrive amplifies. Usually leave empty to avoid ghosting or duplicated features, unless deliberately seeking stronger global structure. |
| **`overdrive_single_blocks`** | `19-23` | Which single blocks overdrive amplifies. The upper range is where skin, texture, and fidelity live, providing photorealism gains. |
| **`overdrive_face_impact`** | `0.20` | How much overdrive reaches the committed face. `1.0` = full energy everywhere, `0.0` = excluded from face. `0.20` is a light touch keeping skin coherent without disturbing identity. |
| **`overdrive_strength`** | `1.1` | Peak residual multiplier at Step 1, easing to 1.0 by the final step. Push gradually; too high will crunch or over-sharpen skin. |
| **`subject_mask`** | *Optional* | Connect a mask here to restrict the Saliency Radar to only consider reference tokens inside the drawn area. |

### Important Tips
* **Step Matching:** Always remember to match your sampling steps to the KSampler steps to ensure proper anatomy and styling.
* **For Photorealism:** Use `frequency_filtering` set to **Band-Pass (Identity)** or **Smooth (Legacy V1)**, and lower the `contrast_and_texture_floor` slightly (around 0.18 - 0.20) to allow natural noise.
* **Using Overdrive:** If your image looks a bit flat, turn on `apply_overdrive` with an `overdrive_strength` of `1.1` to `1.15`. This significantly boosts micro-contrast and skin texture on the specific single blocks without ruining the likeness.
* **For Anything Artistic:** Set `frequency_filtering` to **Raw (No Filter)**, use the "Ease-In" curve, and bump `background_text_strength` up slightly. For more contrast, increase the `contrast_and_texture_floor` to around 0.30+, though this may result in smoother/waxier skin.

### License
This project uses the **PolyForm Noncommercial License 1.0.0**. The short version is: you are completely free to use this tool, as long as you aren't using it to make money.

✅ **What you CAN do for free:** * Use it for your personal art, hobbies, school projects, academic research, or charity work.
* Tinker with the code, modify the node, build on top of it, and share your tweaks with the community (as long as those are also free).

❌ **What you CANNOT do (without permission):** * Use this node inside a paid AI generation service, a commercial cloud product, a Patreon-gated workflow, or any business where it helps generate revenue.

**Looking to use this for business?**
If you are a company or developer wanting to integrate this into a commercial product or paid service, we can set that up! Just open an issue on this GitHub repository or reach out via the contact info on my GitHub profile to arrange a separate commercial license.
"""
