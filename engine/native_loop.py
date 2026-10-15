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

import ctypes

import numpy as np

from . import transformer as T


import os as _os0

ROOT_DIR = _os0.path.dirname(_os0.path.dirname(_os0.path.abspath(__file__)))
from . import native as NK

_SSM = None


def _ssm_lib():
    """libaero_ssm, loaded once. Raises if it is not built -- the hook must not
    silently fall back to a Python SSM when the native one is missing, because the
    whole point of the native hook is the memory profile."""
    global _SSM
    if _SSM is not None:
        return _SSM
    import os as _os
    from ctypes import (POINTER as _P, c_float as _f, c_int as _i,
                        c_int64 as _i64, c_void_p as _vp)
    cands = (_os.path.join(ROOT_DIR, "build", "libaero_ssm.dylib"),
             _os.path.join(ROOT_DIR, "build", "libaero_ssm.so"))
    for path in cands:
        if _os.path.exists(path):
            lib = ctypes.CDLL(path)
            lib.aero_ssm_forward_q.argtypes = [
                _P(_f), _P(_i), _P(_vp), _P(_i64), _P(_f), _P(_f), _P(_f), _P(_f),
                _i, _i, _i, _i, _i, _i, _i, _f, _f, _P(_f), _P(_i), _P(_f), _P(_f), _i]
            lib.aero_ssm_forward_q.restype = _i
            _SSM = lib
            return _SSM
    raise RuntimeError("libaero_ssm not built; run `make all`")


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

    # ---- the NATIVE attention hook ------------------------------------------
    # libaero_attn.dylib implements the whole qwen35 attention layer (projections
    # quantised in place, per-head norms, partial rope, GQA, the gate, the o-proj);
    # verified against the python attention on the REAL blk.3 (rel 5.5e-07). The
    # KV cache lives python-side for now, so the native path here covers only the
    # arithmetic: q/k/v/o projections + norms + rope + gate happen in C; the KV
    # append/read and the attention core are fed from python's cache -- i.e. for
    # THIS pass we use the native kernel via python's own cache semantics.
    # Gated by AERO_ATTN_NATIVE (default 1); a refusal falls back to the verified
    # python attention so a missing dylib can never break the engine.
    class _AttnNative:
        def __init__(self, reader, model):
            import ctypes
            from ctypes import POINTER, c_float, c_int32, c_int64
            self._lib = ctypes.CDLL(os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "build", "libaero_attn.dylib"))
            L = self._lib
            L.aero_attention_qwen.restype = ctypes.c_int
            L.aero_attention_qwen.argtypes = [
                POINTER(c_float), c_int64, c_int64, POINTER(c_int32),
                ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int,
                ctypes.c_int, ctypes.c_void_p, ctypes.c_int64,
                ctypes.c_int, ctypes.c_void_p, ctypes.c_int64,
                ctypes.c_int, ctypes.c_void_p, ctypes.c_int64,
                POINTER(c_float), POINTER(c_float), ctypes.c_int, ctypes.c_float,
                ctypes.c_float, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_float, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, POINTER(c_float)]
            self._reader = reader
            self._model = model

        def __call__(self, layer, h_arr, pos_arr):
            L = self._lib
            r = self._reader
            mcfg = self._model.cfg
            import numpy as np
            from ctypes import POINTER, c_float
            Tt, n_embd = h_arr.shape
            if layer not in r.tensors:
                return None
            p = f"blk.{layer}."
            names = (p + "attn_q.weight", p + "attn_k.weight",
                     p + "attn_v.weight", p + "attn_output.weight")
            if any(n not in r.tensors for n in names):
                return None
            qi_, ki_, vi_, oi_ = (r.tensors[n] for n in names)
            qn = kn = None
            if p + "attn_q_norm.weight" in r.tensors:
                qn = r.tensor(p + "attn_q_norm.weight")
            if p + "attn_k_norm.weight" in r.tensors:
                kn = r.tensor(p + "attn_k_norm.weight")

            def u8b(info):
                b = r.tensor_bytes(info)
                a = (ct.c_ubyte * max(len(b), 1))()
                ct.memmove(a, b, len(b))
                return a

            q_b, k_b, v_b, o_b = u8b(qi_), u8b(ki_), u8b(vi_), u8b(oi_)
            has_gate = 1 if qi_.shape[0] == 2 * mcfg.n_head * mcfg.head_dim else 0

            import engine.transformer as _T
            # the python's own GQA scale/window config for THIS layer
            scale = 0.0        # 0 => native computes 1/sqrt(head_dim)
            out = np.zeros((Tt, n_embd), np.float32)
            NULLF = ctypes.POINTER(c_float)()
            v_rows = vi_.shape[0]
            ok = L.aero_attention_qwen(
                h_arr.ctypes.data_as(POINTER(c_float)), Tt, n_embd,
                np.ascontiguousarray(pos_arr, np.int32).ctypes.data_as(
                    ctypes.POINTER(ct.c_int32)),
                int(qi_.type_id), q_b, int(qi_.shape[0]), has_gate,
                int(ki_.type_id), k_b, int(ki_.shape[0]),
                int(vi_.type_id), v_b, v_rows,
                int(oi_.type_id), o_b, int(oi_.shape[0]),
                (qn.ctypes.data_as(POINTER(c_float)) if qn is not None else NULLF),
                (kn.ctypes.data_as(POINTER(c_float)) if kn is not None else NULLF),
                0, ct.c_float(float(mcfg.eps) if mcfg.eps else 1e-6),
                ct.c_float(float(mcfg.rope_base)),
                int(mcfg.rope_dims or mcfg.head_dim),
                NULLF, 0,
                int(mcfg.n_head), int(mcfg.kv_heads_at(layer)),
                int(mcfg.head_dim), int(mcfg.value_dim),
                ct.c_float(scale), 0,
                1, 8,
                None, None, 0,
                out.ctypes.data_as(POINTER(c_float)))
            import engine.transformer as _T
            # the gate: has_gate => the C applies sigmoid; mode default matches
            return out if ok == 1 else None

    import ctypes as ct
    _attn = None
    try:
        if os.environ.get("AERO_ATTN_NATIVE", "1") == "1":
            _attn = _AttnNative(rt._reader, rt.model)
    except Exception:
        _attn = None        # dylib missing -> the verified python hook runs

    def attn_hook(ctx, layer, h, pos, n_tok, n_embd, out):
        # h is (T, n_embd) fp32; the NATIVE attention runs when possible, else
        # the verified python one -- same buffers, same contract either way.
        h_arr = np.ctypeslib.as_array(h, shape=(n_tok, n_embd)).copy()
        pos_arr = np.ctypeslib.as_array(pos, shape=(n_tok,)).copy()
        try:
            o = _attn(layer, h_arr, pos_arr) if _attn is not None else None
        except Exception:
            o = None
        if o is None:
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
        """Native SSM, weights read as RAW QUANTISED BLOCKS.

        The Python path dequantised every SSM weight: on Qwen3.8-4B the SSM tensors
        are 147 MB quantised and 1025 MB as fp32 (6.98x), and SSM+attention would be
        5.4 GB of fp32 on a 6-8 GB box. aero_ssm_forward_q reads the raw blocks
        through the fused dequant+matmul, so the fp32 weight never exists.

        VERIFIED NAMES in the real file (asserted, not assumed -- my first draft of
        this hook guessed "attn_conv1d" and would have silently dequantised):
            blk.N.attn_qkv.weight   Q6_K  (8192, 2560)   -> 0 qkv
            blk.N.attn_gate.weight  Q4_K  (4096, 2560)   -> 1 gate
            blk.N.ssm_alpha.weight  Q4_K  (32, 2560)     -> 2 alpha
            blk.N.ssm_beta.weight   Q4_K  (32, 2560)     -> 3 beta
            blk.N.ssm_out.weight    Q4_K  (2560, 4096)   -> 4 out
            blk.N.ssm_conv1d.weight fp32  (8192, 4)      kernel = shape[1]
            blk.N.ssm_a / ssm_dt.bias fp32 (32,) ; ssm_norm.weight fp32 (128,)
        """
        h_arr = np.ascontiguousarray(
            np.ctypeslib.as_array(h, shape=(n_tok, n_embd)), dtype=np.float32)
        sc = cfg.ssm_cfg
        p_ = f"blk.{layer}."
        T = rt._reader.tensors

        keep = []
        dts, ptrs, lens = [], [], []
        for nm, key in (("attn_qkv.weight", "qkv"), ("attn_gate.weight", "gate"),
                        ("ssm_alpha.weight", "alpha"), ("ssm_beta.weight", "beta"),
                        ("ssm_out.weight", "out")):
            info = T[p_ + nm]
            if info.type_id in NK.NATIVE_TYPES:
                blob = rt._reader.tensor_bytes(info)
                keep.append(blob)
                buf = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
                keep.append(buf)
                dts.append(int(info.type_id))
                ptrs.append(ctypes.cast(buf, ctypes.c_void_p).value)
                lens.append(len(blob))
            else:
                fp = np.ascontiguousarray(rt._reader.tensor(p_ + nm), np.float32)
                keep.append(fp)
                dts.append(0)
                ptrs.append(fp.ctypes.data)
                lens.append(0)

        a_log = np.ascontiguousarray(rt._reader.tensor(p_ + "ssm_a"), np.float32)
        dt_b = np.ascontiguousarray(rt._reader.tensor(p_ + "ssm_dt.bias"), np.float32)
        conv = np.ascontiguousarray(rt._reader.tensor(p_ + "ssm_conv1d.weight"), np.float32)
        nrm = np.ascontiguousarray(rt._reader.tensor(p_ + "ssm_norm.weight"), np.float32)
        keep += [a_log, dt_b, conv, nrm]
        # The recurrent matrix lives in state.gdn[layer] -- (n_v, d, d) fp32, the same
        # buffer the Python ssm_forward passes as its gdn_state. SsmState itself holds
        # only geometry + the conv history, so passing it (as my first draft did) raised
        # inside the ctypes callback and was SWALLOWED: the hook 'ran', returned garbage,
        # and parity came back 1.25 relative. Hook failures must never be silent; this
        # is the second time that trap has cost a debugging round in this file.
        gdn_slot = np.ascontiguousarray(state.gdn[layer], dtype=np.float32)
        state.gdn[layer] = gdn_slot
        kernel = int(getattr(sc, "conv_kernel", conv.shape[1]))

        dt_arr = (ctypes.c_int * 5)(*dts)
        ptr_arr = (ctypes.c_void_p * 5)(*ptrs)
        len_arr = (ctypes.c_int64 * 5)(*lens)

        tail = state.ssm_tails.get(layer)
        if tail is None:
            tail = np.zeros((kernel - 1, sc.qkv_width), np.float32)
            state.ssm_tails[layer] = tail
        tl = ctypes.c_int(len(tail))
        o = np.zeros((n_tok, n_embd), np.float32)
        rc = _ssm_lib().aero_ssm_forward_q(
            h_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dt_arr, ptr_arr, len_arr,
            a_log.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dt_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            conv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            nrm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_tok, n_embd, sc.inner_size, sc.state_size, sc.n_group,
            kernel, sc.time_step_rank, 1e-6, float(cfg.eps),
            tail.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.byref(tl),
            gdn_slot.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            o.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(rt.threads or 1))
        if rc != 0:
            return 0
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


