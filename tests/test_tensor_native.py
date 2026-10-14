"""Native fp32 primitives must match the numpy reference.

Three separate traps are encoded here, all of which produced a WRONG verdict while
this suite was being written. They are the reason this is a test and not a script:

  1. DIVIDING BY A NEAR-ZERO DENOMINATOR. Comparing with max|a-b| / |a| reported a
     2% 'error' in matvec, because the reference element happened to be ~0. Error
     must be measured against an exact (float64) result, or against the result's own
     SCALE -- never element-by-element against the value being compared.

  2. REUSING ONE BUFFER FOR BOTH SIDES. These primitives are in-place, so handing
     the same array to the Python call and then the native call feeds the native
     side already-transformed data. It reported 10% errors in softmax and gelu. Each
     side gets its own copy.

  3. TRUSTING A TOLERANCE WITHOUT A REFERENCE. 'Matches the Python' is weak if the
     Python could be wrong. Where a closed form exists (gelu/erf, softmax sum-to-one,
     l2 unit norm) this checks against THAT, not just against the other port.

What this suite deliberately does NOT do is assert bit-exactness: the C++ and numpy
sum in different orders, so ~1e-6 relative is the floor. Asserting equality would
make the test wrong, not the code.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import math

import numpy as np

from engine import ssm as SSM
from engine import tensor_native as TN
from engine import transformer as T

_fail = 0
_n = 0


def check(cond, desc):
    global _fail, _n
    _n += 1
    if cond:
        print("  ok   " + desc)
    else:
        _fail += 1
        print("  FAIL " + desc)


if not TN.available():
    print(f"\n=== 0 checks, 0 failures (native tensor library not built) ===")
    print(f"    reason: {TN.unavailable_reason()}")
    sys.exit(0)

rng = np.random.default_rng(7)
EPS = 1e-6
N = 2560


def rel_to_scale(got, ref):
    """Error relative to the reference's own magnitude.

    NOT elementwise |a-b|/|a| -- that explodes wherever the reference crosses zero
    and reports large 'errors' for numerically perfect output.
    """
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    scale = max(float(np.abs(ref).max()), 1e-30)
    return float(np.abs(got - ref).max()), float(np.abs(got - ref).max() / scale)


print("\n[1] the library is reachable at all")
print(f"    reason-if-any: {TN.unavailable_reason()!r}")
check(TN.available(), "libaero_tensor loaded")
check(TN.threads() > 0, f"native thread count positive ({TN.threads()})")

print("\n[2] matvec / matmul against an exact float64 result")
x = rng.standard_normal(N).astype(np.float32)
w = rng.standard_normal((9216, N)).astype(np.float32)
exact = w.astype(np.float64) @ x.astype(np.float64)
ma, mr = rel_to_scale(TN.matvec(w, x), exact)
check(mr < 1e-5, f"matvec 9216x2560 rel={mr:.3e} abs={ma:.3e}")

X = rng.standard_normal((4, N)).astype(np.float32)
exact_m = X.astype(np.float64) @ w.astype(np.float64).T
ma, mr = rel_to_scale(TN.matmul(w, X), exact_m)
check(mr < 1e-5, f"matmul 4x9216x2560 rel={mr:.3e} abs={ma:.3e}")

# a row-sum probe: with x all ones the answer must be the row sums, which is an
# independent closed form (no accumulation-order ambiguity to hide behind)
ones = np.ones(N, np.float32)
rowsums = w.astype(np.float64).sum(axis=1)
ma, mr = rel_to_scale(TN.matvec(w, ones), rowsums)
check(mr < 1e-5, f"matvec(all-ones) == row sums, rel={mr:.3e}")

print("\n[3] elementwise ops, each side given its OWN copy")
xs = rng.standard_normal(N).astype(np.float32)
up = rng.standard_normal(N).astype(np.float32)
g = (rng.standard_normal(N) + 1.5).astype(np.float32)

for name, pyf, nf in (("gelu", T.gelu, TN.gelu), ("silu", T.silu, TN.silu),
                      ("softmax", T.softmax, TN.softmax)):
    a = pyf(xs.copy())            # fresh copy: these are IN-PLACE
    b = nf(xs.copy())             # fresh copy again
    ma, mr = rel_to_scale(b, a)
    check(ma < 1e-5, f"{name} abs={ma:.3e} (py and native agree)")

a = T.swiglu(xs.copy(), up.copy())
b = TN.swiglu(xs.copy(), up.copy())
ma, mr = rel_to_scale(b, a)
check(ma < 1e-5, f"swiglu abs={ma:.3e}")

a = TN.mul(xs.copy(), g.copy())
check(np.allclose(a, xs * g, atol=1e-6), "mul matches elementwise product")

print("\n[4] closed-form checks (do not just trust the other port)")
# gelu must be the ERF form, not the tanh approximation -- the two differ by ~1e-4
# at x=3, which is far above the fp32 floor and would be a real model bug.
for xv in (0.0, 1.0, 3.0, -2.5):
    got = float(TN.gelu(np.array([xv], np.float32))[0])
    want = 0.5 * xv * (1.0 + math.erf(xv / math.sqrt(2.0)))
    tanh = 0.5 * xv * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (xv + 0.044715 * xv ** 3)))
    check(abs(got - want) < 1e-6 and (xv == 0.0 or abs(got - tanh) > 1e-7),
          f"gelu({xv}) = {got:.7f} matches erf form ({want:.7f}), not tanh ({tanh:.7f})")

# silu(x) = x * sigmoid(x)
for xv in (0.0, 1.0, -1.0, 5.0):
    got = float(TN.silu(np.array([xv], np.float32))[0])
    want = xv / (1.0 + math.exp(-xv))
    check(abs(got - want) < 1e-6, f"silu({xv}) = {got:.7f} matches x*sigmoid(x)")

# softmax must sum to exactly 1 (within fp32) and be order-preserving
sm = TN.softmax(xs.copy())
check(abs(sm.sum() - 1.0) < 1e-5, f"softmax sums to 1 ({sm.sum():.7f})")
check(bool(np.all(np.diff(sm[np.argsort(xs)]) >= -1e-7)),
      "softmax is monotonic in its input")

# l2norm: eps is a FLOOR ON THE NORM, not sqrt(sum+eps). These differ consistently.
v = rng.standard_normal(64).astype(np.float32)
out = TN.l2norm(v, None, eps=1e-6)
check(abs(float(np.linalg.norm(out)) - 1.0) < 1e-5,
      f"weightless l2norm yields a unit vector (|out|={np.linalg.norm(out):.7f})")

# the discriminating case: a vector whose norm is BELOW eps. The floor must bite,
# so the output is NOT unit length. Two traps here, both of which produced a wrong
# verdict while writing this:
#   * rng.standard_normal(64) * 1e-9 does NOT have norm ~8e-9 in float32. Squaring
#     ~1e-9-scale fp32 values loses all significance in the float32 sum, and the
#     measured norm comes back ~7e-9 only because the values stay representable --
#     the point is that the SCALE is what matters, so state it explicitly and use a
#     vector whose norm is unambiguous.
#   * asserting "< 1e-3" was arbitrary. For x = 1e-9 everywhere, norm = 8e-9 and
#     dividing by the eps floor (1e-6) gives 8e-3, so the correct assertion is
#     |out| == ||x|| / eps, or simply "not unit" with the value printed.
v_sub = np.full(64, 1e-9, np.float32)
norm_v = float(np.linalg.norm(v_sub.astype(np.float64)))
out_tiny = TN.l2norm(v_sub.copy(), None, eps=1e-6)
got_norm = float(np.linalg.norm(out_tiny.astype(np.float64)))
want_norm = norm_v / 1e-6
print(f"    ||x||={norm_v:.3e} < eps=1e-6 -> |out|={got_norm:.3e} (expect ||x||/eps={want_norm:.3e})")
check(abs(got_norm - want_norm) / want_norm < 1e-3,
      "l2norm floors at eps: |out| == ||x||/eps, so NOT unit length")
check(got_norm < 0.5, "a floored l2norm is far from unit length (floor really bites)")
# the native and Python floors must agree, which is the property that actually matters
py_norm = float(np.linalg.norm(SSM.l2_normalize(v_sub.copy(), 1e-6).astype(np.float64)))
check(abs(got_norm - py_norm) / max(py_norm, 1e-30) < 1e-5,
      f"floor agrees with ssm.l2_normalize ({got_norm:.3e} vs {py_norm:.3e})")

# and it must match the Python reference exactly on the normal case
ma, mr = rel_to_scale(TN.l2norm(xs.copy(), None), SSM.l2_normalize(xs.copy()))
check(ma < 1e-6, f"l2norm matches ssm.l2_normalize abs={ma:.3e}")

print("\n[5] rmsnorm vs the reference, and its own definition")
out = TN.rmsnorm(xs.copy(), g.copy(), EPS)
want = xs / np.sqrt((xs.astype(np.float64) ** 2).mean() + EPS) * g
ma, mr = rel_to_scale(out, want)
check(mr < 1e-6, f"rmsnorm matches its formula rel={mr:.3e}")

print("\n[6] argmax agrees with numpy")
check(TN.argmax(xs) == int(np.argmax(xs)), "argmax agrees")

print("\n[7] bad arguments are refused, not silently accepted")
for fn, args, label in (
        (TN.matvec, (np.zeros((4, 8), np.float32), np.zeros(3, np.float32)), "matvec x-size mismatch"),
        (TN.matmul, (np.zeros((4, 8), np.float32), np.zeros((2, 3), np.float32)), "matmul X cols mismatch"),
        (TN.mul, (np.zeros(4, np.float32), np.zeros(5, np.float32)), "mul length mismatch"),
        (TN.rmsnorm, (np.zeros(4, np.float32), np.zeros(5, np.float32), 1e-6), "rmsnorm size mismatch")):
    try:
        fn(*args)
        check(False, f"{label} raises")
    except (ValueError, TypeError) as e:
        check(True, f"{label} raises ({type(e).__name__})")

# float64 is COERCED to float32, deliberately -- the kernel is fp32-only and a
# silent up-cast would be the alternative. Assert the coercion, and assert the
# result dtype, rather than demanding a rejection the binding never promised.
r64 = TN.matvec(np.eye(8, dtype=np.float64), np.arange(8, dtype=np.float64))
check(r64.dtype == np.float32, f"float64 input is coerced to float32 (got {r64.dtype})")
check(np.allclose(r64, np.arange(8, dtype=np.float32), atol=1e-5),
      "the coerced float64 result is still numerically correct")
# non-numeric input must be refused rather than coerced into nonsense
try:
    TN.matvec("not an array", np.zeros(8, np.float32))
    check(False, "a non-array is rejected")
except (TypeError, ValueError):
    check(True, "a non-array is rejected")

print(f"\n=== {_n} checks, {_fail} failures ===")
sys.exit(1 if _fail else 0)
