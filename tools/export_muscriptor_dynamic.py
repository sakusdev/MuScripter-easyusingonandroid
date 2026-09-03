#!/usr/bin/env python3
"""Export MuScriptor to the optimized Android ABI: block prefill + 1-token decode.

This keeps exactly one stateful ExecuTorch `forward` method. The same method
accepts either S=1 or any sequence up to 504 embeddings, so Android can prefill
503 conditioning embeddings + the initial token in one call and then continue
with ordinary one-token autoregressive decoding. Keeping one method is
important because ExecuTorch Android 1.4 does not opt its Module wrapper into
cross-method shared memory arenas.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

import export_muscriptor as base


# 501 self_wav embeddings + dataset + instrument + initial model token.
MAX_PREFILL_SEQUENCE = base.MEL_FRAMES_5S + 3  # 504
DECODER_SEQUENCE_MODE = "dynamic_block_v1"


class DynamicDecoderLayer(base.DecoderLayer):
    """The ABI-v1 layer generalized from S=1 to a dynamic sequence S."""

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        # x: [1,S,D], input_pos: [S], positions are absolute and contiguous.
        qkv = F.linear(self.norm1(x), self.in_proj_weight)
        b, s, _ = qkv.shape
        packed = qkv.view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = packed.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        self.k_cache.index_copy_(2, input_pos, k)
        self.v_cache.index_copy_(2, input_pos, v)

        # Each query row attends through its own absolute position. Slots above
        # that point are invisible. This simultaneously provides causal block
        # prefill and all-history S=1 decode, and makes position-0 restart
        # logically equivalent to a freshly allocated upstream model_state.
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
        state: Mapping[str, torch.Tensor],
        cfg: base.ModelConfig,
        max_context: int,
    ) -> None:
        super().__init__()
        self.dim = cfg.dim
        self.card = cfg.card
        self.max_context = max_context
        self.layers = nn.ModuleList(
            [DynamicDecoderLayer(state, i, cfg, max_context) for i in range(cfg.num_layers)]
        )
        self.out_norm = nn.LayerNorm(cfg.dim, eps=1e-5)
        self.out_norm.weight = nn.Parameter(base._require(state, "out_norm.weight").clone())
        self.out_norm.bias = nn.Parameter(base._require(state, "out_norm.bias").clone())
        self.linear = nn.Linear(cfg.dim, cfg.card, bias=False)
        self.linear.weight = nn.Parameter(base._require(state, "linear.weight").clone())
        self.register_buffer(
            "vocab_ids",
            torch.arange(cfg.card, dtype=torch.long),
            persistent=False,
        )

    def forward(self, embedding: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        # Upstream positions are always computed in fp32. Generalize the exact
        # sinusoid from one position to S positions.
        half = self.dim // 2
        positions = input_pos.to(torch.float32).view(1, -1, 1)
        adim = torch.arange(
            half,
            dtype=torch.float32,
            device=input_pos.device,
        ).view(1, 1, -1)
        phase = positions / (
            torch.tensor(10_000.0, device=input_pos.device) ** (adim / (half - 1))
        )
        pos = torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)

        x = embedding + pos.to(embedding.dtype)
        for layer in self.layers:
            x = layer(x, input_pos)

        # Only the last row is useful for greedy generation. Keeping the output
        # fixed at [1,card] avoids copying 504xcard logits back through JNI.
        logits = self.linear(self.out_norm(x))[:, -1, :].float()
        if self.card > base.GENERATED_VOCAB_SIZE:
            logits = torch.where(
                self.vocab_ids.view(1, -1) < base.GENERATED_VOCAB_SIZE,
                logits,
                torch.full_like(logits, float("-inf")),
            )
        return logits


def export_dynamic_decoder(
    model: nn.Module,
    out: Path,
    cfg: base.ModelConfig,
    xnnpack: bool,
) -> None:
    from executorch.exir import to_edge_transform_and_lower

    # Use a non-trivial example so S is actually exercised as a sequence.
    example_s = 3
    seq = torch.export.Dim("seq", min=1, max=MAX_PREFILL_SEQUENCE)
    ep = torch.export.export(
        model.eval(),
        (
            torch.zeros(
                1,
                example_s,
                cfg.dim,
                dtype=model.linear.weight.dtype,
            ),
            torch.arange(example_s, dtype=torch.long),
        ),
        dynamic_shapes=(
            {1: seq},
            {0: seq},
        ),
        strict=True,
    )

    if xnnpack:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        from executorch.backends.xnnpack.utils.configs import (
            get_xnnpack_edge_compile_config,
        )

        edge = to_edge_transform_and_lower(
            ep,
            partitioner=[XnnpackPartitioner()],
            compile_config=get_xnnpack_edge_compile_config(),
        )
    else:
        edge = to_edge_transform_and_lower(ep)
    out.write_bytes(edge.to_executorch().buffer)


def build_bundle(
    weights: Path,
    output: Path,
    config_path: Path | None,
    max_context: int,
    xnnpack: bool,
) -> Path:
    state = base.remap_legacy_keys(load_file(weights, device="cpu"))
    cfg = base.infer_config(state, config_path)
    if max_context < MAX_PREFILL_SEQUENCE:
        raise ValueError(
            f"max_context={max_context} is smaller than prefill {MAX_PREFILL_SEQUENCE}"
        )

    output.mkdir(parents=True, exist_ok=True)
    conditioner_path = output / "conditioner.pte"
    embedder_path = output / "embedder.pte"
    decoder_path = output / "decoder.pte"

    conditioner = base.Conditioner(state)
    embedder = base.TokenEmbedder(state)
    decoder = DynamicDecoder(state, cfg, max_context)

    base.export_pte(
        conditioner,
        (
            torch.zeros(1, base.MEL_FRAMES_5S, base.MEL_BINS),
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
        ),
        conditioner_path,
        xnnpack,
    )
    base.export_pte(
        embedder,
        (torch.full((1, 1), cfg.card, dtype=torch.long),),
        embedder_path,
        xnnpack,
    )
    export_dynamic_decoder(decoder, decoder_path, cfg, xnnpack)

    file_hashes = {
        conditioner_path.name: base.sha256_file(conditioner_path),
        embedder_path.name: base.sha256_file(embedder_path),
        decoder_path.name: base.sha256_file(decoder_path),
    }
    manifest = {
        "format": "muscriptor-android-bundle",
        "abi_version": base.ABI_VERSION,
        "model": asdict(cfg),
        "source": {
            "weights_name": weights.name,
            "weights_sha256": base.sha256_file(weights),
        },
        "runtime": {
            "executorch": "1.4.x",
            "backend": "xnnpack" if xnnpack else "portable",
            "max_context": max_context,
            "decode_step": 1,
            "decoder_sequence_mode": DECODER_SEQUENCE_MODE,
            "max_prefill_sequence": MAX_PREFILL_SEQUENCE,
        },
        "frontend": {
            "sample_rate": 16000,
            "segment_seconds": 5.0,
            "n_fft": 2048,
            "win_length": 2048,
            "window": "hann_periodic",
            "hop_length": 160,
            "frame_rate": 100,
            "center": True,
            "pad_mode": "reflect",
            "onesided": True,
            "normalized": False,
            "power": 1.0,
            "mel_scale": "htk",
            "mel_norm": None,
            "f_min": 0.0,
            "f_max": 8000.0,
            "mel_bins": base.MEL_BINS,
            "log_eps": 1e-6,
            "frames_for_full_segment": base.MEL_FRAMES_5S,
        },
        "conditioning": {
            "instrument_token_count": 1,
            "dataset_token_count": 1,
            "default_instrument_token": 0,
            "default_dataset_token": 0,
            "prefix_order": ["self_wav", "dataset_name", "instrument_group"],
        },
        "vocabulary": {
            "generated_size": base.GENERATED_VOCAB_SIZE,
            "model_cardinality": cfg.card,
            "initial_token_id": cfg.card,
            "eos_token_id": 1,
        },
        "files": {
            "conditioner": conditioner_path.name,
            "embedder": embedder_path.name,
            "decoder": decoder_path.name,
        },
        "sha256": file_hashes,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    bundle = output.with_suffix(".msa")
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in (
            "manifest.json",
            "conditioner.pte",
            "embedder.pte",
            "decoder.pte",
        ):
            zf.write(output / name, arcname=name)
    return bundle


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights", type=Path, help="MuScriptor model.safetensors")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("build/muscriptor-android"),
        help="Output directory; <output>.msa is also created",
    )
    p.add_argument("--config", type=Path, help="Optional upstream config.json")
    p.add_argument("--max-context", type=int, default=base.DEFAULT_MAX_CONTEXT)
    p.add_argument("--portable", action="store_true", help="Skip XNNPACK partitioning")
    args = p.parse_args()

    config = args.config
    if config is None:
        candidate = args.weights.with_name("config.json")
        config = candidate if candidate.exists() else None

    bundle = build_bundle(
        args.weights,
        args.output,
        config,
        max_context=args.max_context,
        xnnpack=not args.portable,
    )
    print(bundle)


if __name__ == "__main__":
    main()
