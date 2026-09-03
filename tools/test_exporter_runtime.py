#!/usr/bin/env python3
"""Smoke-test the Android decoder ABI through the real ExecuTorch Python runtime.

This deliberately uses a tiny synthetic MuScriptor-shaped transformer so CI can
verify mutable KV-cache export without downloading licensed model weights.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from executorch.runtime import Runtime

from export_muscriptor import ModelConfig, OneTokenDecoder, export_pte


def synthetic_state(cfg: ModelConfig) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(20260903)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g) * 0.08

    state: dict[str, torch.Tensor] = {
        "linear.weight": randn(cfg.card, cfg.dim),
        "out_norm.weight": torch.ones(cfg.dim),
        "out_norm.bias": torch.zeros(cfg.dim),
    }
    ff = cfg.dim * 4
    for layer in range(cfg.num_layers):
        p = f"transformer.layers.{layer}."
        state[p + "self_attn.in_proj_weight"] = randn(3 * cfg.dim, cfg.dim)
        state[p + "self_attn.out_proj.weight"] = randn(cfg.dim, cfg.dim)
        state[p + "norm1.weight"] = torch.ones(cfg.dim)
        state[p + "norm1.bias"] = torch.zeros(cfg.dim)
        state[p + "norm2.weight"] = torch.ones(cfg.dim)
        state[p + "norm2.bias"] = torch.zeros(cfg.dim)
        state[p + "linear1.weight"] = randn(ff, cfg.dim)
        state[p + "linear2.weight"] = randn(cfg.dim, ff)
    return state


def run_eager(model: OneTokenDecoder, embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
    outputs = []
    with torch.inference_mode():
        for pos, embedding in enumerate(embeddings):
            outputs.append(model(embedding, torch.tensor([pos], dtype=torch.long)).clone())
    return outputs


def main() -> None:
    cfg = ModelConfig("tiny", dim=16, num_heads=4, num_layers=2, card=1393)
    state = synthetic_state(cfg)
    max_context = 12

    torch.manual_seed(7)
    embeddings = [torch.randn(1, 1, cfg.dim) for _ in range(4)]

    eager = OneTokenDecoder(state, cfg, max_context).eval()
    eager_outputs = run_eager(eager, embeddings)

    with tempfile.TemporaryDirectory() as td:
        pte = Path(td) / "decoder.pte"
        exported = OneTokenDecoder(state, cfg, max_context).eval()
        export_pte(
            exported,
            (torch.zeros(1, 1, cfg.dim), torch.zeros(1, dtype=torch.long)),
            pte,
            xnnpack=True,
        )
        assert pte.stat().st_size > 0

        runtime = Runtime.get()
        program = runtime.load_program(str(pte))
        method = program.load_method("forward")

        for pos, (embedding, expected) in enumerate(zip(embeddings, eager_outputs)):
            actual = method.execute([embedding, torch.tensor([pos], dtype=torch.long)])[0]
            torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-4)

        # A new chunk starts at position 0. The runtime module still physically
        # contains previous cache bytes, but the mask must make them invisible.
        restart_embedding = torch.randn(1, 1, cfg.dim)
        fresh_eager = OneTokenDecoder(state, cfg, max_context).eval()
        with torch.inference_mode():
            expected_restart = fresh_eager(
                restart_embedding,
                torch.tensor([0], dtype=torch.long),
            )
        actual_restart = method.execute(
            [restart_embedding, torch.tensor([0], dtype=torch.long)]
        )[0]
        torch.testing.assert_close(
            actual_restart,
            expected_restart,
            rtol=2e-3,
            atol=2e-4,
        )

    print("ExecuTorch XNNPACK decoder ABI parity: OK")


if __name__ == "__main__":
    main()
