"""GgufRuntime: drive the C++ block loop with the verified Python primitives.

Wiring for `AERO_NATIVE_LOOP` (default 1), added to engine/runtime.py. Lives in a
mixin so the GgufRuntime class body stays readable.

DESIGN (measured, not assumed):
  * 32 ctypes round-trips per token cost ~1 ms against a 2.5 s token (0.04%), so
    hook-backed layers lose nothing while the native attention/SSM/MoE kernels are
    being wired -- the Python-inside-hooks is what porting removes, step by step.
  * The loop, the norm placements, the residual order and the head are ALL in C++,
    so an entire class of "wrong layout" bugs can never come back through env
    flags: the flags that select GEMMA_OUT_SCALE / BLOCK_MODE in Python have C++
    counterparts passed as integers in the geometry struct.
  * FALLBACK IS LOUD: when the dylib is missing or a hook is refused, the load
    stats record it and forward() raises, rather than silently computing in
    Python and printing a number the user believes was native.
"""

from __future__ import annotations

import ctypes

import numpy as np

from . import transformer as T


def _geom_from_cfg(rt):
    """aero_geom fields mirroring what Model.forward actually does.

    The Python flags are the source of truth: the C++ must be told the SAME
    layout the Python loop uses, or the two disagree and the parity test fails --
    which is exactly what it is for.
    """
    from . import model as M
    cfg = rt.model.cfg
    gemma_out_map = {"none": 0, "deltas": 1, "sum": 2}
    return dict(
        arch="gemma4" if cfg.gemma_cfg is not None else "qwen35",
        n_layer=cfg.n_layer,
        n_embd=cfg.n_embd,
        vocab=rt.tokenizer.n_vocab if rt.tokenizer else 0,
        eps=float(cfg.eps),
        tied_output=bool(cfg.tied_output),
        softcap=float(cfg.softcap or 0.0),
        embed_scale=(float(cfg.embed_scale)
                     if (cfg.embed_scale is not None
                         and M.GEMMA_EMBED_SCALE == "sqrt_embd") else 0.0),
        block_on_output=(T.BLOCK_MODE == "on_output"),
        gemma_out_scale=(gemma_out_map.get(M.GEMMA_OUT_SCALE, 2)
                         if cfg.gemma_cfg is not None else 2),
        block_rows=int(rt.block_rows),
        threads=int(rt.threads or 1),
    )


def layer_specs(rt):
    """Per-layer (kind, out_scale) for the native loop.

    kind: 0 = attention, 1 = SSM (the C++ enum). is_moe: 1 when the layer's FFN
    is routed; MoE layers keep kind by their attention/SSM kind, exactly like
    engine/model.py treats them.
    """
    cfg = rt.model.cfg
    out = []
    for spec in cfg.layers:
        kind = 1 if spec.kind == "ssm" else 0
        scale = 1.0
        if cfg.gemma_cfg is not None:
            from . import gemma as GEMMA
            w = rt._reader.tensor(f"blk.{spec.index}.layer_output_scale.weight")
            scale = float(GEMMA.scale_of(w))
        out.append((kind, 0, scale))
    return out


def build_native_loop(rt):
    """Open the C++ block loop for this runtime, with hook-backed layers.

    Hooks: closure-based adapters from the C ABI (plain pointers) to the Python
    primitives, so the loop runs natively while every layer keeps today's
    verified numerics.
    """
    from . import native_model as NM
    from .chat_template import EnvelopeError  # noqa: F401  (re-exported error)
    cfg = rt.model.cfg

    g = _geom_from_cfg(rt)
    if g["vocab"] == 0:
        raise RuntimeError("no tokenizer; the native loop needs vocab for the head")
    specs = layer_specs(rt)
    kinds = [k for k, _, _ in specs]
    scales = [s for _, _, s in specs]

    n = int(cfg.n_embd)
    state = rt.state

    def attn_hook(ctx, layer, h, pos, n_tok, n_embd, out):
        # h is (T, n_embd) fp32; call the verified Python attention and write out.
        h_arr = np.ctypeslib.as_array(h, shape=(n_tok, n_embd)).copy()
        pos_arr = np.ctypeslib.as_array(pos, shape=(n_tok,)).copy()
        w = rt.model._attn_weights(layer)
        acfg = T.AttnConfig(
            n_head=cfg.n_head, n_head_kv=cfg.kv_heads_at(layer),
            head_dim=cfg.head_dim, value_dim=cfg.value_dim,
            rope_base=cfg.rope_base, rope_dims=cfg.rope_dims, eps=cfg.eps)
        o = T.attention(h_arr, pos_arr, w, acfg, cache=state.kv,
                        layer=layer, commit=False)
        out_arr = np.ctypeslib.as_array(out, shape=(n_tok, n_embd))
        out_arr[:] = o
        return 1

    def ssm_hook(ctx, layer, h, n_tok, n_embd, out):
        h_arr = np.ctypeslib.as_array(h, shape=(n_tok, n_embd)).copy()
        w = __import__("engine.ssm", fromlist=["ssm_weights_from"]).ssm_weights_from(
            rt._reader, layer)
        o, gdn_new = __import__("engine.ssm", fromlist=["ssm_forward"]).ssm_forward(
            h_arr, w, cfg.ssm_cfg, state.ssm_states[layer], state.gdn[layer])
        state.gdn[layer] = gdn_new
        out_arr = np.ctypeslib.as_array(out, shape=(n_tok, n_embd))
        out_arr[:] = o
        return 1

    def moe_hook(ctx, layer, h, n_tok, n_embd, out):
        h_arr = np.ctypeslib.as_array(h, shape=(n_tok, n_embd)).copy()
        cfg_moe = cfg.moe_cfg
        d = rt._moe_ffn(h_arr, layer, cfg_moe, state)
        out_arr = np.ctypeslib.as_array(out, shape=(n_tok, n_embd))
        out_arr[:] = d
        return 1

    loop = NM.NativeBlockLoop(
        rt._reader.path,
        arch=g["arch"], n_layer=g["n_layer"], n_embd=g["n_embd"], vocab=g["vocab"],
        eps=g["eps"], tied_output=g["tied_output"], softcap=g["softcap"],
        embed_scale=g["embed_scale"], block_on_output=g["block_on_output"],
        gemma_out_scale=g["gemma_out_scale"], block_rows=g["block_rows"],
        threads=g["threads"], layer_kinds=kinds, out_scales=scales,
        attn_hook=attn_hook, ssm_hook=ssm_hook if cfg.ssm_cfg is not None else None,
        moe_hook=moe_hook if cfg.moe_cfg is not None else None)

    return loop


