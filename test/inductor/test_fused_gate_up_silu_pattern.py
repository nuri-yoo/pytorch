# Owner(s): ["module: inductor"]

import unittest

import torch
import torch.nn.functional as F
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code


class LlamaMLP(torch.nn.Module):
    """Minimal Llama MLP for pattern matching test."""

    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
class TestPatternMatch(TestCase):
    def test_pattern_fires(self):
        """Verify the pattern matcher replaces silu(mm)*mm with fused op."""
        model = LlamaMLP(512, 1384).half().to("xpu")
        x = torch.randn(32, 512, device="xpu", dtype=torch.float16)

        out, codes = run_and_get_code(torch.compile(model, backend="inductor"), x)

        # Verify the fused kernel appears in the generated code
        code = codes[0] if len(codes) == 1 else "\n".join(codes)
        self.assertIn(
            "fused_gate_up_silu",
            code,
            "fused_gate_up_silu not found in generated inductor code",
        )

        # Verify correctness
        ref = model(x)
        self.assertTrue(torch.allclose(ref, out, rtol=2e-3, atol=0.5))

    def test_fallback_on_cpu(self):
        """Pattern should NOT fire on CPU — output must still be correct."""
        model = LlamaMLP(512, 1384).half()
        x = torch.randn(32, 512, dtype=torch.float16)

        out, codes = run_and_get_code(torch.compile(model, backend="inductor"), x)

        code = codes[0] if len(codes) == 1 else "\n".join(codes)
        self.assertNotIn("fused_gate_up_silu", code)

        ref = model(x)
        self.assertTrue(torch.allclose(ref, out, rtol=2e-3, atol=0.5))

    def test_fallback_on_fp32(self):
        """Pattern should NOT fire on fp32 — output must still be correct."""
        model = LlamaMLP(512, 1384).to("xpu")
        x = torch.randn(32, 512, device="xpu")

        out, codes = run_and_get_code(torch.compile(model, backend="inductor"), x)

        code = codes[0] if len(codes) == 1 else "\n".join(codes)
        self.assertNotIn("fused_gate_up_silu", code)

        ref = model(x)
        self.assertTrue(torch.allclose(ref, out, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    run_tests()
