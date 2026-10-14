"""Compare aero's Q4_K (type 12) dequant against llama.cpp's gguf-py reference.

token_embd.weight in gemma-4-12b-it-IQ4_XS.gguf is the ONLY Q4_K tensor in the
file, and it feeds layer 0 and the (tied) lm_head. If its dequant is wrong the
whole model is semantically dead while still numerically finite.

Run from the repo root:
    AERO_MAX_RAM=5G .venv-cml/bin/python tools/diag_q4k.py
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '/Volumes/Backup/Development/llama.cpp/gguf-py')
import numpy as np

P = '/Users/cyberdeth/.cache/aero-models/gemma-4-12b-it-IQ4_XS.gguf'
NAME = 'token_embd.weight'
NROW = 4

import gguf
from gguf import quants
from gguf.constants import GGML_QUANT_SIZES

ref_reader = gguf.GGUFReader(P)
fld = next(t for t in ref_reader.tensors if t.name == NAME)
ncols = int(fld.shape[0])      # gguf-py reports (row_len, n_rows)
nrows = int(fld.shape[1])
block, tsize = GGML_QUANT_SIZES[fld.tensor_type]
row_bytes = (ncols // block) * tsize
print('gguf-py: type %s shape %s block %d tsize %d row_bytes %d'
      % (fld.tensor_type, (ncols, nrows), block, tsize, row_bytes))
print('gguf-py data memmap', fld.data.shape, fld.data.dtype)

raw_ref = None
with open(P, 'rb') as _f:
    _f.seek(int(fld.data_offset))          # already absolute in gguf-py
    raw_ref = _f.read(row_bytes * NROW)
tb = np.frombuffer(raw_ref, dtype=np.uint8).reshape(-1, tsize)
ref = np.asarray(quants.dequantize(tb, fld.tensor_type),
                 dtype=np.float32).reshape(NROW, ncols)
print('ref shape', ref.shape)
print('ref[0,:6]', ref[0, :5])
print('ref rms[0] %.6f max %.6f' % (float(np.sqrt(np.mean(ref[0] ** 2))),
                                    float(np.max(np.abs(ref[0])))))

from engine.gguf import GGUFReader
from engine import quant as Q
try:
    from engine import native
except Exception as e:                                     # pragma: no cover
    native = None
    print('native import failed:', e)
have_native = bool(native and native.available())
print('native available:', have_native,
      'types:', sorted(native.NATIVE_TYPES) if have_native else None)


def rel(a, b):
    d = float(np.abs(a - b).max())
    s = max(float(np.abs(b).max()), 1e-12)
    return d, d / s


with GGUFReader(P) as r:
    info = r.tensors[NAME]
    print('aero: type_id %s shape %s' % (info.type_id, tuple(info.shape)))
    rb = r.row_bytes(info)
    assert rb == row_bytes, (rb, row_bytes)
    # raw slice, exactly the bytes gguf-py got
    r._fh.seek(r.data_offset + info.offset)
    raw = r._read(row_bytes * NROW)
    assert raw == raw_ref, 'byte slices differ'

    py = np.asarray(Q.dequantize(raw, info.type_id, (NROW, ncols)),
                    dtype=np.float32).reshape(NROW, ncols)
    d, rl = rel(py, ref)
    print('PYTHON q4_k dequant  vs gguf-py: max abs %.3e  rel %.3e' % (d, rl))

    if have_native and info.type_id in native.NATIVE_TYPES:
        nat = np.asarray(native.dequant(info.type_id, raw, (NROW, ncols)),
                         dtype=np.float32).reshape(NROW, ncols)
        d, rl = rel(nat, ref)
        print('NATIVE q4_k dequant  vs gguf-py: max abs %.3e  rel %.3e' % (d, rl))
        d, rl = rel(nat, py)
        print('NATIVE vs PYTHON                 : max abs %.3e  rel %.3e' % (d, rl))
    else:
        print('NATIVE q4_k: NOT in NATIVE_TYPES -> Python path is used')

    # what the engine actually feeds layer 0 / the lm_head with
    t_rows = np.asarray(r.tensor_rows(info, 0, NROW), dtype=np.float32)
    d, rl = rel(t_rows, ref)
    print('aero tensor_rows()   vs gguf-py: max abs %.3e  rel %.3e' % (d, rl))
    for i in (0, 1, 2):
        print('  row %d rms %.6f' % (i, float(np.sqrt(np.mean(t_rows[i] ** 2)))))
