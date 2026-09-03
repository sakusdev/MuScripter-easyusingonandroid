#!/usr/bin/env python3
"""Prove one ExecuTorch method can serve block prefill and 1-token decode.

Android's current ExecuTorch Module wrapper does not opt into cross-method shared
memory arenas, so the production ABI should keep KV state inside one `forward`
method. This test exports that method with a dynamic sequence dimension, calls
it once with a block, then repeatedly with S=1, and checks the real XNNPACK
runtime against eager PyTorch.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.runtime import Runtime

from export_muscriptor import ModelConfig
from test_exporter_runtime import synthetic_state


class DynamicLayer(nn.Module):
    def __init__(
        self,
        state: dict[str, torch.Tensor],
        layer: int,
        cfg: ModelConfig,
        max_context: int,
    ) -> None:
        super().__init__()
        p = f"transformer.layers.{layer}."
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.max_context = max_context

        self.in_proj_weight = nn.Parameter(state[p + "self_attn.in_proj_weight"].clone())
        self.out_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.out_proj.weight = nn.Parameter(state[p + "self_attn.out_proj.weight"].clone())

        self.norm1 = nn.LayerNorm(cfg.dim, eps=1e-5)
        self.norm1.weight = nn.Parameter(state[p + "norm1.weight"].clone())
        self.norm1.bias = nn.Parameter(state[p + "norm1.bias"].clone())
        self.norm2 = nn.LayerNorm(cfg.dim, eps=1e-5)
        self.norm2.weight = nn.Parameter(state[p + "norm2.weight"].clone())
        self.norm2.bias = nn.Parameter(state[p + "norm2.bias"].clone())

        ff = state[p + "linear1.weight"].shape[0]
        self.linear1 = nn.Linear(cfg.dim, ff, bias=False)
        self.linear1.weight = nn.Parameter(state[p + "linear1.weight"].clone())
        self.linear2 = nn.Linear(ff, cfg.dim, bias=False)
        self.linear2.weight = nn.Parameter(state[p + "linear2.weight"].clone())

        cache_shape = (1, cfg.num_heads, max_context, cfg.head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(cache_shape), persistent=False)
        self.register_buffer(
            "cache_indices", torch.arange(max_context, dtype=torch.long), persistent=False
        )

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        # x: [1,S,D], input_pos: [S]
        qkv = F.linear(self.norm1(x), self.in_proj_weight)
        b, s, _ = qkv.shape
        packed = qkv.view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = packed.unbind(dim=2)
        q = q.transpose(1, 2)  # [1,H,S,head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        self.k_cache.index_copy_(2, input_pos, k)
        self.v_cache.index_copy_(2, input_pos, v)

        # Every query row may attend up to its own absolute position. This is
        # square causal attention during block prefill and all-history attention
        # when S=1 during decode.
        allowed = self.cache_indices.view(1, -1) <= input_pos.view(-1, 1)
        mask = torch.where(
            allowed,
            torch.zeros((), dtype=q.dtype, device=q.device),
            torch.full((), float("-inf"), dtype=q.dtype, device=q.device),
        ).view(1, 1, s, self.max_context)
        attn = F.scaled_dot_product_attention(
            q,
            self.k_cache,
            self.v_cache,
            attn_mask=mask,
            dropout_p=0.0,
        )
        attn = attn.transpose(1, 2).reshape(b, s, -1)
        x = x + self.out_proj(attn)
        x = x + self.linear2(F.gelu(self.linear1(self.norm2(x))))
        return x


class DynamicDecoder(nn.Module):
    def __init__(
        self,
        state: dict[str, torch.Tensor],
        cfg: ModelConfig,
        max_context: int,
    ) -> None:
        super().__init__()
        self.dim = cfg.dim
        self.max_context = max_context
        self.layers = nn.ModuleList(
            [DynamicLayer(state, i, cfg, max_context) for i in range(cfg.num_layers)]
        )
        self.out_norm = nn.LayerNorm(cfg.dim, eps=1e-5)
        self.out_norm.weight = nn.Parameter(state["out_norm.weight"].clone())
        self.out_norm.bias = nn.Parameter(state["out_norm.bias"].clone())
        self.linear = nn.Linear(cfg.dim, cfg.card, bias=False)
        self.linear.weight = nn.Parameter(state["linear.weight"].clone())

    def forward(self, embedding: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        # Exact upstream sinusoidal positions, generalized from one position to S.
        half = self.dim // 2
        positions = input_pos.to(torch.float32).view(1, -1, 1)
        adim = torch.arange(half, dtype=torch.float32).view(1, 1, -1)
        phase = positions / (10_000.0 ** (adim / (half - 1)))
        x = embedding + torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1).to(
            embedding.dtype
        )
        for layer in self.layers:
            x = layer(x, input_pos)
        return self.linear(self.out_norm(x)).float()


def export_dynamic(model: nn.Module, out: Path, cfg: ModelConfig, max_seq: int) -> None:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
    from executorch.exir import to_edge_transform_and_lower

    example_s = 3
    seq = torch.export.Dim("seq", min=1, max=max_seq)
    ep = torch.export.export(
        model.eval(),
        (
            torch.zeros(1, example_s, cfg.dim),
            torch.arange(example_s, dtype=torch.long),
        ),
        dynamic_shapes=(
            {1: seq},
            {0: seq},
        ),
        strict=True,
    )
    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[XnnpackPartitioner()],
        compile_config=get_xnnpack_edge_compile_config(),
    )
    out.write_bytes(edge.to_executorch().buffer)


def main() -> None:
    cfg = ModelConfig("tiny-dynamic", dim=16, num_heads=4, num_layers=2, card=1393)
    state = synthetic_state(cfg)
    max_context = 12
    max_dynamic_seq = 6

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
        export_dynamic(DynamicDecoder(state, cfg, max_context), pte, cfg, max_dynamic_seq)
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

    print("ExecuTorch XNNPACK dynamic prefill/decode ABI parity: OK")


if __name__ == "__main__":
    main()
