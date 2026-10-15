
import ctypes, glob, sys
import numpy as np
sys.path.insert(0, "/Volumes/Backup/Development/aero")
from engine.gguf import GGUFReader
import engine.model as M
P = "/Users/cyberdeth/.cache/aero-models/Qwen3.8-4B-Q4_K_M.gguf"
L = ctypes.CDLL("/Volumes/Backup/Development/aero/build/libaero_ssm.dylib")
F = ctypes.POINTER(ctypes.c_float)
L.aero_ssm_forward_q.restype = ctypes.c_int
L.aero_ssm_forward_q.argtypes = [F, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int64), F, F, F, F, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
    F, ctypes.POINTER(ctypes.c_int), F, F, ctypes.c_int]
with GGUFReader(P) as r:
    m = M.Model(r, block_rows=256, matmul_threads=8)
    sc = m.cfg.ssm_cfg
    NE, IN, DS, NG, K, TR = m.cfg.n_embd, sc.inner_size, sc.state_size, sc.n_group, sc.conv_kernel, sc.time_step_rank
    QW = IN + 2*NG*DS; p_ = "blk.0."
    dts, ptrs, lens, keep = [], [], [], []
    for nm in ("attn_qkv.weight","attn_gate.weight","ssm_alpha.weight","ssm_beta.weight","ssm_out.weight"):
        blob = r.tensor_bytes(r.tensors[p_+nm]); buf = ctypes.create_string_buffer(blob, len(blob))
        keep.append(buf); dts.append(r.tensors[p_+nm].type_id)
        ptrs.append(ctypes.cast(buf, ctypes.c_void_p).value); lens.append(len(blob))
    a_log = np.ascontiguousarray(r.tensor(p_+"ssm_a"), np.float32)
    dt_b  = np.ascontiguousarray(r.tensor(p_+"ssm_dt.bias"), np.float32)
    conv  = np.ascontiguousarray(r.tensor(p_+"ssm_conv1d.weight"), np.float32)
    nrm   = np.ascontiguousarray(r.tensor(p_+"ssm_norm.weight"), np.float32)
    keep += [a_log, dt_b, conv, nrm]
    dt_arr=(ctypes.c_int*5)(*dts); ptr_arr=(ctypes.c_void_p*5)(*ptrs); len_arr=(ctypes.c_int64*5)(*lens)
    rng = np.random.default_rng(7)
    H5 = np.ascontiguousarray(rng.standard_normal((5,NE)).astype(np.float32)*0.5)
    def bad(a): return int((~np.isfinite(a)).sum())
    def fresh(): return np.zeros((K-1,QW),np.float32), np.zeros((TR,DS,DS),np.float32)
    def call(h,tail,tl,gdn,T):
        o = np.zeros((T,NE),np.float32)
        rc = L.aero_ssm_forward_q(h.ctypes.data_as(F),dt_arr,ptr_arr,len_arr,
            a_log.ctypes.data_as(F),dt_b.ctypes.data_as(F),conv.ctypes.data_as(F),nrm.ctypes.data_as(F),
            T,NE,IN,DS,NG,K,TR,ctypes.c_float(1e-6),ctypes.c_float(sc.eps),
            tail.ctypes.data_as(F),ctypes.byref(tl),gdn.ctypes.data_as(F),o.ctypes.data_as(F),8)
        return rc,o
    tA,gA = fresh(); tlA = ctypes.c_int(K-1)
    rcA,oA = call(H5,tA,tlA,gA,5)
    print("one-shot T=5 : rc=%d  o nonfinite=%d/%d  gdn nonfinite=%d/%d  tail nonfinite=%d"
          % (rcA, bad(oA), oA.size, bad(gA), gA.size, bad(tA)))
    tB,gB = fresh(); tlB = ctypes.c_int(K-1)
    rcB,oB = call(np.ascontiguousarray(H5[:4]),tB,tlB,gB,4)
    print("prefill  T=4 : rc=%d  o nonfinite=%d/%d  gdn nonfinite=%d/%d  tail nonfinite=%d  tl=%d"
          % (rcB, bad(oB), oB.size, bad(gB), gB.size, bad(tB), tlB.value))
    rcC,oC = call(np.ascontiguousarray(H5[4:5]),tB,tlB,gB,1)
    print("decode   T=1 : rc=%d  o nonfinite=%d/%d  gdn nonfinite=%d/%d  tl=%d"
          % (rcC, bad(oC), oC.size, bad(gB), gB.size, tlB.value))
    d = float(np.abs(oA[4] - oC[0]).max())
    rel = d / max(float(np.abs(oA[4]).max()), 1e-9)
    print("\nINVARIANT one-shot[4] vs carried-decode[0]: max|d|=%.4e rel=%.3e  -> %s"
          % (d, rel, "PASS" if rel < 1e-4 else "FAIL"))
    print("          one-shot[4] max|o|=%.5f   decode[0] max|o|=%.5f"
          % (float(np.abs(oA[4]).max()), float(np.abs(oC[0]).max())))
    # is the carried path even CLOSE? compare against a no-history T=1 (tl=0)
    tN,gN = fresh(); tlN = ctypes.c_int(0)
    rcN,oN = call(np.ascontiguousarray(H5[4:5]),tN,tlN,gN,1)
    print("          vs a no-history decode: max|one-shot[4]-that|=%.4e"
          % float(np.abs(oA[4]-oN[0]).max()))
    # control: does a T=1 call work at all with a ZERO tail and zero gdn?
    tZ,gZ = fresh(); tlZ = ctypes.c_int(K-1)
    rcZ,oZ = call(np.ascontiguousarray(H5[4:5]),tZ,tlZ,gZ,1)
    print("control  T=1 (zero tail+gdn): rc=%d  o nonfinite=%d/%d" % (rcZ, bad(oZ), oZ.size))
    t0,g0 = fresh(); tl0 = ctypes.c_int(0)
    rc0,o0 = call(np.ascontiguousarray(H5[4:5]),t0,tl0,g0,1)
    print("control  T=1 (tl=0):           rc=%d  o nonfinite=%d/%d" % (rc0, bad(o0), o0.size))
