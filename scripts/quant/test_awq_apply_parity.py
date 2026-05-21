"""FP-parity test for AWQ scale absorption.

Verifies that applying AWQ transforms to (RMSNorm γ + consumer W) and
(up_proj + down_proj) pairs leaves the forward output unchanged up to
floating-point rounding.  This is the contract awq_apply.py must satisfy.

Run:
    python scripts/quant/test_awq_apply_parity.py
"""

import sys
import pathlib

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from awq_apply import _apply_member_scale, _apply_absorb


def _rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x, weight, eps)


def _make_scale(in_dim: int, seed: int = 0) -> mx.array:
    """A non-trivial diagonal scale: half columns > 1, half < 1, geometric mean ≈ 1."""
    mx.random.seed(seed)
    s = mx.exp(mx.random.normal((in_dim,)) * 0.3)
    return s / mx.sqrt(s.max() * s.min())


# ---------------------------------------------------------------------------
# Case 1: norm_weight absorption (G1 qkv_input, G2 mlp_input pattern)
# ---------------------------------------------------------------------------

def test_norm_weight_absorption() -> None:
    """RMSNorm γ → Linear W   ==   RMSNorm (γ/s) → Linear (W·diag(s))."""
    mx.random.seed(0)
    B, L, in_dim, out_dim = 2, 5, 96, 48

    x = mx.random.normal((B, L, in_dim)) * 0.5
    gamma = mx.random.normal((in_dim,)) * 0.1 + 1.0  # ≈ centered at 1
    W = mx.random.normal((out_dim, in_dim)) * 0.1
    s = _make_scale(in_dim, seed=42)

    # Original forward
    h0 = _rms_norm(x, gamma)
    y0 = h0 @ W.T
    mx.eval(y0)

    # Build mock modules with .weight attribute, mutate, re-forward
    class _Carrier:
        def __init__(self, w):
            self.weight = w

    norm_mod = _Carrier(gamma)
    lin_mod = _Carrier(W)

    _apply_member_scale(lin_mod, s)             # W ← W · diag(s)
    _apply_absorb(norm_mod, s, "norm_weight")   # γ ← γ / s

    h1 = _rms_norm(x, norm_mod.weight)
    y1 = h1 @ lin_mod.weight.T
    mx.eval(y1)

    diff = mx.max(mx.abs(y1 - y0))
    mx.eval(diff)
    diff_f = float(diff)
    rel = diff_f / max(float(mx.max(mx.abs(y0))), 1e-12)
    print(f"  norm_weight: max|Δ|={diff_f:.3e} rel={rel:.3e}")
    # rms_norm in f32 + matmul: ~5e-6 absolute, ~1e-5 relative is the rounding floor.
    assert rel < 1e-4, f"norm_weight absorption broke parity: rel={rel}"


# ---------------------------------------------------------------------------
# Case 2: linear_out_axis absorption (G3 mlp_down pattern)
# ---------------------------------------------------------------------------

def test_linear_out_axis_absorption() -> None:
    """down(gelu(gate) · up(x))  ==  down·diag(s)( gelu(gate) · (up·diag(1/s_out)(x)) ).

    Concretely: up_proj's OUT axis divided by s, down_proj's IN axis multiplied
    by s.  The gelu(gate) factor is unaffected (same shape, no axis scaling).
    """
    mx.random.seed(1)
    B, L, hidden, inter = 2, 4, 32, 80  # inter must be % 64 for AFFINE quant later

    x_norm = mx.random.normal((B, L, hidden)) * 0.3
    W_gate = mx.random.normal((inter, hidden)) * 0.1
    W_up = mx.random.normal((inter, hidden)) * 0.1
    W_down = mx.random.normal((hidden, inter)) * 0.1

    # s has shape [inter] — that's down_proj's IN axis == up_proj's OUT axis.
    s = _make_scale(inter, seed=99)

    # Original forward
    gate0 = x_norm @ W_gate.T
    up0 = x_norm @ W_up.T
    h0 = nn.gelu_approx(gate0) * up0
    y0 = h0 @ W_down.T
    mx.eval(y0)

    class _Carrier:
        def __init__(self, w):
            self.weight = w

    up_mod = _Carrier(W_up)
    down_mod = _Carrier(W_down)

    _apply_member_scale(down_mod, s)               # W_down ← W_down · diag(s)  (in axis)
    _apply_absorb(up_mod, s, "linear_out_axis")    # W_up   ← W_up / s[:, None]  (out axis)

    gate1 = x_norm @ W_gate.T  # gate untouched
    up1 = x_norm @ up_mod.weight.T
    h1 = nn.gelu_approx(gate1) * up1
    y1 = h1 @ down_mod.weight.T
    mx.eval(y1)

    diff = mx.max(mx.abs(y1 - y0))
    mx.eval(diff)
    diff_f = float(diff)
    rel = diff_f / max(float(mx.max(mx.abs(y0))), 1e-12)
    print(f"  linear_out_axis: max|Δ|={diff_f:.3e} rel={rel:.3e}")
    assert rel < 1e-4, f"linear_out_axis absorption broke parity: rel={rel}"


# ---------------------------------------------------------------------------
# Case 3: SwitchLinear member scaling (Phase B — applies to expert groups)
# ---------------------------------------------------------------------------

def test_switchlinear_member_scale() -> None:
    """SwitchLinear weight [E, out, in] · diag(s)   ==   per-input-channel scale."""
    mx.random.seed(2)
    E, out_dim, in_dim = 4, 32, 64
    W = mx.random.normal((E, out_dim, in_dim)) * 0.1
    s = _make_scale(in_dim, seed=7)

    # Simulate SWITCH_CLASSES: use a fake class that isinstance-matches via patch
    class _FakeSwitch:
        pass
    # _apply_member_scale uses isinstance(module, SWITCH_CLASSES) — patch globally
    import awq_apply
    orig_switch = awq_apply.SWITCH_CLASSES
    awq_apply.SWITCH_CLASSES = (_FakeSwitch,)

    class _Mod(_FakeSwitch):
        def __init__(self, w):
            self.weight = w
    try:
        mod = _Mod(W)
        _apply_member_scale(mod, s)
        # Expected: W_new[e, o, c] == W[e, o, c] * s[c]
        expected = W.astype(mx.float32) * s[None, None, :]
        diff = mx.max(mx.abs(mod.weight.astype(mx.float32) - expected))
        mx.eval(diff)
        diff_f = float(diff)
        print(f"  switchlinear: max|Δ|={diff_f:.3e}")
        assert diff_f < 1e-5, f"SwitchLinear scaling broke: {diff_f}"
    finally:
        awq_apply.SWITCH_CLASSES = orig_switch


# ---------------------------------------------------------------------------

def main() -> int:
    print("[test] norm_weight absorption (G1 qkv_input, G2 mlp_input)")
    test_norm_weight_absorption()
    print("[test] linear_out_axis absorption (G3 mlp_down)")
    test_linear_out_axis_absorption()
    print("[test] switchlinear member scaling (Phase B groups)")
    test_switchlinear_member_scale()
    print("OK — all AWQ absorption identities hold within fp rounding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
