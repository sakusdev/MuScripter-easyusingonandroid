#!/usr/bin/env python3
"""Runtime parity test for the production dynamic block-prefill decoder."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from executorch.runtime import Runtime

from export_muscriptor import ModelConfig
from export_muscriptor_dynamic import DynamicDecoder, export_dynamic_decoder
from test_exporter_runtime import synthetic_state


def main() -> None:
    cfg = ModelConfig("tiny-dynamic", dim=16, num_heads=4, num_layers=2, card=1393)
    state = synthetic_state(cfg)
    # Production export allows S<=504, so the synthetic cache must be at least that long.
    max_context = 520

    torch.manual_seed(17)
    prefill = torch.randn(1, 5, cfg.dim)
    decode1 = torch.randn(1, 1, cfg.dim)
    decode2 = torch.randn(1, 1, cfg.dim)

    eager = DynamicDecoder(state, cfg, max_context).eval()
    with torch.inference_mode():
        expected_prefill = eager(prefill, torch.arange(5, dtype=torch.long))
        expected_decode1 = eager(decode1, torch.tensor([5], dtype=torch.long))
        expected_decode2 = eager(decode2, torch.tensor([6], dtype=torch.long))

    with tempfile.TemporaryDirectory() as td:
        pte = Path(td) / "dynamic_decoder.pte"
        export_dynamic_decoder(
            DynamicDecoder(state, cfg, max_context),
            pte,
            cfg,
            xnnpack=True,
        )
        assert pte.stat().st_size > 0

        method = Runtime.get().load_program(str(pte)).load_method("forward")
        actual_prefill = method.execute([prefill, torch.arange(5, dtype=torch.long)])[0]
        actual_decode1 = method.execute([decode1, torch.tensor([5], dtype=torch.long)])[0]
        actual_decode2 = method.execute([decode2, torch.tensor([6], dtype=torch.long)])[0]

        torch.testing.assert_close(actual_prefill, expected_prefill, rtol=2e-3, atol=2e-4)
        torch.testing.assert_close(actual_decode1, expected_decode1, rtol=2e-3, atol=2e-4)
        torch.testing.assert_close(actual_decode2, expected_decode2, rtol=2e-3, atol=2e-4)

        # Reusing the same method at position 0 starts a logically fresh chunk.
        restart = torch.randn(1, 4, cfg.dim)
        fresh = DynamicDecoder(state, cfg, max_context).eval()
        with torch.inference_mode():
            expected_restart = fresh(restart, torch.arange(4, dtype=torch.long))
        actual_restart = method.execute([restart, torch.arange(4, dtype=torch.long)])[0]
        torch.testing.assert_close(actual_restart, expected_restart, rtol=2e-3, atol=2e-4)

    print("Production ExecuTorch XNNPACK dynamic prefill/decode parity: OK")


if __name__ == "__main__":
    main()
