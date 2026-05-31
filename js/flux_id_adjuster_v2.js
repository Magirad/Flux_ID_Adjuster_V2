// flux_id_adjuster_v2.js

import { app } from "../../scripts/app.js";

const EXPERT_ONLY = new Set([
    "saliency_scan_blocks",
    "target_likeness_metric",
    "soft_blend_k",
    "face_isolation_strictness",
    "confidence_gate",
    "hard_anchor_margin",
    "contrast_and_texture_floor",
]);

const OVERDRIVE_OPTS = new Set([
    "overdrive_double_blocks",
    "overdrive_single_blocks",
    "overdrive_face_impact",
    "overdrive_strength",
]);

// Using standard "hidden" type so ComfyUI/LiteGraph doesn't trip up on serialization
const HIDDEN_TYPE = "hidden";

function isTargetNode(node) {
    return (node.comfyClass || node.type) === "FluxIDAutoAdjusterv2";
}

function toggleWidget(widget, show) {
    if (!widget) return;
    
    if (widget.origType === undefined) {
        widget.origType = widget.type;
        widget.origComputeSize = widget.computeSize;
    }
    
    widget.hidden = !show;
    if (show) {
        widget.type = widget.origType;
        widget.computeSize = widget.origComputeSize;
    } else {
        widget.type = HIDDEN_TYPE;
        widget.computeSize = () => [0, -4];
    }
}

function isOn(widget) {
    if (!widget) return false;
    const v = widget.value;
    return v === true || v === 1 || v === "true";
}

function numVal(widget) {
    if (!widget) return 0;
    const n = Number(widget.value);
    return isNaN(n) ? 0 : n;
}

function updateVisibility(node) {
    if (!node.widgets) return;
    
    // Look up the auto-generated ui_mode widget value
    const modeWidget = node.widgets.find((w) => w.name === "ui_mode");
    const mode = modeWidget ? modeWidget.value : "Basic";
    const odOn = isOn(node.widgets.find((w) => w.name === "apply_overdrive"));
    const bgOn = numVal(node.widgets.find((w) => w.name === "background_text_strength")) > 0;

    for (const w of node.widgets) {
        if (w === modeWidget) continue;
        
        let show = true;
        if (OVERDRIVE_OPTS.has(w.name)) show = odOn;            
        else if (EXPERT_ONLY.has(w.name)) show = mode === "Expert";
        
        if (w.name === "dynamic_text_balancing" && !bgOn) show = false;
        
        toggleWidget(w, show);
    }

    const size = node.computeSize();
    size[0] = Math.max(node.size[0], size[0]); 
    node.setSize(size);
    node.setDirtyCanvas(true, true);
}

function watch(node, name) {
    const w = node.widgets ? node.widgets.find((x) => x.name === name) : null;
    if (!w) return;
    const prev = w.callback;
    w.callback = function () {
        const r = prev ? prev.apply(this, arguments) : undefined;
        updateVisibility(node);
        return r;
    };
}

function setupModeWidget(node) {
    if (node._fluxModeWidget) return;

    // 1. Find the widget Python already created instead of making a new one
    const modeWidget = node.widgets.find(w => w.name === "ui_mode");
    if (!modeWidget) return;

    node._fluxModeWidget = modeWidget;

    // 2. Watch our trigger widgets
    watch(node, "ui_mode");
    watch(node, "apply_overdrive");
    watch(node, "background_text_strength");

    console.log("[FluxID UI] attached to node:", node.comfyClass || node.type);
    updateVisibility(node);
}

app.registerExtension({
    name: "FluxID.Adjuster.UI",
    nodeCreated(node) {
        if (!isTargetNode(node)) return;

        // Defer setup slightly to ensure ComfyUI has populated Python widgets
        setTimeout(() => {
            setupModeWidget(node);
        }, 1);

        const onConfigure = node.onConfigure;
        node.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            // Update visibility right after LiteGraph restores the saved widget values
            updateVisibility(node);
            return r;
        };
    },
});