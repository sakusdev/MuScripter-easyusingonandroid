#!/usr/bin/env python3
"""Export MuScriptor checkpoints into an Android/ExecuTorch model bundle.

Bring-up ABI v1 intentionally favors correctness over speed:

* Android computes the exact 512-bin log-mel frontend.
* conditioner.pte projects log-mel + class conditions into prefix embeddings.
* embedder.pte maps one MT3 token id to a transformer embedding.
* decoder.pte consumes exactly one embedding at a time and owns its KV caches.

Feeding the 503-condition-token prefix one position at a time is slower than a
block prefill, but keeps the decoder shape static and makes the first parity
milestone substantially easier. A block-prefill ABI can replace this later
without changing weights or MT3 decoding.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


ABI_VERSION = 1
GENERATED_VOCAB_SIZE = 1393
MEL_BINS = 512
MEL_FRAMES_5S = 501
DEFAULT_MAX_GENERATED = 2000
DEFAULT_MAX_CONTEXT = MEL_FRAMES_5S + 2 + DEFAULT_MAX_GENERATED


@dataclass(frozen=True)
class ModelConfig:
    variant: str
    dim: int
    num_heads: int
    num_layers: int
    card: int

    @property
    def head_dim(self) -> int:
        if self.dim % self.num_heads:
            raise ValueError(f"dim={self.dim} is not divisible by num_heads={self.num_heads}")
        return self.dim // self.num_heads


KNOWN_CONFIGS = {
    "small": ModelConfig("small", dim=768, num_heads=12, num_layers=14, card=1393),
    "medium": ModelConfig("medium", dim=1024, num_heads=16, num_layers=24, card=1395),
    "large": ModelConfig("large", dim=1536, num_heads=24, num_layers=48, card=1395),
}
HEADS_BY_DIM = {cfg.dim: cfg.num_heads for cfg in KNOWN_CONFIGS.values()}


def remap_legacy_keys(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map old single-codebook key names to current upstream MuScriptor names."""
    if any(k.startswith(("emb.1.", "linears.1.")) for k in state):
        raise ValueError("Multi-codebook checkpoints are not supported")
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("emb.0."):
            key = "emb." + key[len("emb.0.") :]
        elif key.startswith("linears.0."):
            key = "linear." + key[len("linears.0.") :]
        out[key] = value
    return out


def infer_config(state: Mapping[str, torch.Tensor], config_path: Path | None) -> ModelConfig:
    if config_path and config_path.exists():
        raw = json.loads(config_path.read_text())
        return ModelConfig(
            variant=str(raw.get("variant", "custom")),
            dim=int(raw["dim"]),
            num_heads=int(raw["num_heads"]),
            num_layers=int(raw["num_layers"]),
            card=int(raw["card"]),
        )

    linear = state.get("linear.weight")
    emb = state.get("emb.weight")
    if linear is None or emb is None:
        raise ValueError("Checkpoint must contain linear.weight and emb.weight")
    card, dim = map(int, linear.shape)
    if emb.shape[1] != dim:
        raise ValueError("emb.weight and linear.weight disagree on model dimension")

    layer_ids = {
        int(m.group(1))
        for key in state
        if (m := re.match(r"transformer\.layers\.(\d+)\.", key))
    }
    if not layer_ids:
        raise ValueError("Could not infer transformer layer count")
    num_layers = max(layer_ids) + 1
    num_heads = HEADS_BY_DIM.get(dim)
    if num_heads is None:
        raise ValueError(
            f"Unknown model dimension {dim}; pass a config.json containing num_heads"
        )
    variant = next(
        (
            name
            for name, cfg in KNOWN_CONFIGS.items()
            if (cfg.dim, cfg.num_layers, cfg.card) == (dim, num_layers, card)
        ),
        "custom",
    )
    return ModelConfig(variant, dim, num_heads, num_layers, card)


def _require(state: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    try:
        return state[key]
    except KeyError as exc:
        raise KeyError(f"Checkpoint is missing required tensor: {key}") from exc


class Conditioner(nn.Module):
    """Build the exact upstream condition prefix from precomputed log-mel.

    Class inputs use upstream's tokenized representation: 0 is null/pad and a
    real class N is passed as N + 1. ClassConditioner.forward then indexes the
    embedding with token + 1; that extra +1 is intentionally reproduced here.
    """

    def __init__(self, state: Mapping[str, torch.Tensor]):
        super().__init__()
        self.mel_weight = nn.Parameter(
            _require(state, "condition_provider.conditioners.self_wav.output_proj.weight").clone()
        )
        self.mel_bias = nn.Parameter(
            _require(state, "condition_provider.conditioners.self_wav.output_proj.bias").clone()
        )
        self.instrument_weight = nn.Parameter(
            _require(state, "condition_provider.conditioners.instrument_group.embed.weight").clone()
        )
        self.dataset_weight = nn.Parameter(
            _require(state, "condition_provider.conditioners.dataset_name.embed.weight").clone()
        )

    def forward(
        self,
        log_mel: torch.Tensor,           # [1, 501, 512]
        instrument_tokens: torch.Tensor, # [1, 1]
        dataset_tokens: torch.Tensor,    # [1, 1]
    ) -> torch.Tensor:
        mel = F.linear(log_mel, self.mel_weight, self.mel_bias)
        instrument = F.embedding(instrument_tokens + 1, self.instrument_weight)
        dataset = F.embedding(dataset_tokens + 1, self.dataset_weight)
        # LMModel iterates instrument, dataset, wav and prepends each one.
        # Repeated prepend reverses the final transformer prefix order.
        return torch.cat((mel, dataset, instrument), dim=1)


class TokenEmbedder(nn.Module):
    def __init__(self, state: Mapping[str, torch.Tensor]):
        super().__init__()
        self.weight = nn.Parameter(_require(state, "emb.weight").clone())

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Android ABI normally never sends the upstream -1 zero token; retain
        # ScaledEmbedding parity anyway so the module is independently exact.
        is_zero = token_ids < 0
        ids = torch.clamp(token_ids, min=0)
        y = F.embedding(ids, self.weight)
        return torch.where(is_zero.unsqueeze(-1), torch.zeros_like(y), y)


def sinusoidal_position(
    input_pos: torch.Tensor,
    dim: int,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """Exact fp32 positional embedding used by StreamingTransformer."""
    half = dim // 2
    pos = input_pos.to(torch.float32).view(1, 1, 1)
    adim = torch.arange(
        half, dtype=torch.float32, device=input_pos.device
    ).view(1, 1, -1)
    phase = pos / (
        torch.tensor(max_period, device=input_pos.device) ** (adim / (half - 1))
    )
    return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        state: Mapping[str, torch.Tensor],
        layer: int,
        cfg: ModelConfig,
        max_context: int,
    ):
        super().__init__()
        p = f"transformer.layers.{layer}."
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.max_context = max_context

        self.in_proj_weight = nn.Parameter(
            _require(state, p + "self_attn.in_proj_weight").clone()
        )
        self.out_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.out_proj.weight = nn.Parameter(
            _require(state, p + "self_attn.out_proj.weight").clone()
        )

        self.norm1 = nn.LayerNorm(cfg.dim, eps=1e-5)
        self.norm1.weight = nn.Parameter(_require(state, p + "norm1.weight").clone())
        self.norm1.bias = nn.Parameter(_require(state, p + "norm1.bias").clone())
        self.norm2 = nn.LayerNorm(cfg.dim, eps=1e-5)
        self.norm2.weight = nn.Parameter(_require(state, p + "norm2.weight").clone())
        self.norm2.bias = nn.Parameter(_require(state, p + "norm2.bias").clone())

        ff_dim = int(_require(state, p + "linear1.weight").shape[0])
        self.linear1 = nn.Linear(cfg.dim, ff_dim, bias=False)
        self.linear1.weight = nn.Parameter(
            _require(state, p + "linear1.weight").clone()
        )
        self.linear2 = nn.Linear(ff_dim, cfg.dim, bias=False)
        self.linear2.weight = nn.Parameter(
            _require(state, p + "linear2.weight").clone()
        )

        cache_shape = (1, cfg.num_heads, max_context, cfg.head_dim)
        cache_dtype = self.in_proj_weight.dtype
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=cache_dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=cache_dtype), persistent=False
        )
        self.register_buffer(
            "cache_indices", torch.arange(max_context, dtype=torch.long), persistent=False
        )

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        qkv = F.linear(self.norm1(x), self.in_proj_weight)
        b, s, _ = qkv.shape
        packed = qkv.view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = packed.unbind(dim=2)
        q = q.transpose(1, 2)  # [B,H,1,D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Static one-token decode ABI. index_copy_ mirrors ExecuTorch's Llama
        # KVCache implementation and is captured as mutable module buffer state.
        self.k_cache.index_copy_(2, input_pos, k)
        self.v_cache.index_copy_(2, input_pos, v)

        # A fixed-size cache avoids dynamic slicing. Old values above input_pos
        # are masked, so a new chunk can restart at position 0 without clearing.
        allowed = self.cache_indices <= input_pos[0]
        mask = torch.where(
            allowed,
            torch.zeros((), dtype=q.dtype, device=q.device),
            torch.full((), float("-inf"), dtype=q.dtype, device=q.device),
        ).view(1, 1, 1, self.max_context)
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


class OneTokenDecoder(nn.Module):
    """MuScriptor transformer with model-owned KV cache, one position per call."""

    def __init__(
        self,
        state: Mapping[str, torch.Tensor],
        cfg: ModelConfig,
        max_context: int,
    ):
        super().__init__()
        self.dim = cfg.dim
        self.card = cfg.card
        self.max_context = max_context
        self.layers = nn.ModuleList(
            [DecoderLayer(state, i, cfg, max_context) for i in range(cfg.num_layers)]
        )
        self.out_norm = nn.LayerNorm(cfg.dim, eps=1e-5)
        self.out_norm.weight = nn.Parameter(_require(state, "out_norm.weight").clone())
        self.out_norm.bias = nn.Parameter(_require(state, "out_norm.bias").clone())
        self.linear = nn.Linear(cfg.dim, cfg.card, bias=False)
        self.linear.weight = nn.Parameter(_require(state, "linear.weight").clone())
        self.register_buffer(
            "vocab_ids", torch.arange(cfg.card, dtype=torch.long), persistent=False
        )

    def forward(
        self,
        embedding: torch.Tensor, # [1, 1, D]
        input_pos: torch.Tensor, # [1]
    ) -> torch.Tensor:
        pos = sinusoidal_position(input_pos, self.dim).to(embedding.dtype)
        x = embedding + pos
        for layer in self.layers:
            x = layer(x, input_pos)
        logits = self.linear(self.out_norm(x))[:, -1, :].float()
        if self.card > GENERATED_VOCAB_SIZE:
            logits = torch.where(
                self.vocab_ids.view(1, -1) < GENERATED_VOCAB_SIZE,
                logits,
                torch.full_like(logits, float("-inf")),
            )
        return logits


def export_pte(
    model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    out: Path,
    xnnpack: bool,
) -> None:
    """torch.export -> Edge -> ExecuTorch .pte, optionally using XNNPACK."""
    from executorch.exir import to_edge_transform_and_lower

    ep = torch.export.export(model.eval(), example_inputs, strict=True)
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
    program = edge.to_executorch()
    out.write_bytes(program.buffer)


def build_bundle(
    weights: Path,
    output: Path,
    config_path: Path | None,
    max_context: int,
    xnnpack: bool,
) -> Path:
    state = remap_legacy_keys(load_file(weights, device="cpu"))
    cfg = infer_config(state, config_path)
    if max_context < MEL_FRAMES_5S + 3:
        raise ValueError("max_context is too small for the 5-second condition prefix")

    output.mkdir(parents=True, exist_ok=True)
    conditioner_path = output / "conditioner.pte"
    embedder_path = output / "embedder.pte"
    decoder_path = output / "decoder.pte"

    conditioner = Conditioner(state)
    embedder = TokenEmbedder(state)
    decoder = OneTokenDecoder(state, cfg, max_context)

    export_pte(
        conditioner,
        (
            torch.zeros(1, MEL_FRAMES_5S, MEL_BINS),
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
        ),
        conditioner_path,
        xnnpack,
    )
    export_pte(
        embedder,
        (torch.full((1, 1), cfg.card, dtype=torch.long),),
        embedder_path,
        xnnpack,
    )
    export_pte(
        decoder,
        (
            torch.zeros(
                1,
                1,
                cfg.dim,
                dtype=_require(state, "linear.weight").dtype,
            ),
            torch.zeros(1, dtype=torch.long),
        ),
        decoder_path,
        xnnpack,
    )

    manifest = {
        "format": "muscriptor-android-bundle",
        "abi_version": ABI_VERSION,
        "model": asdict(cfg),
        "runtime": {
            "executorch": "1.4.x",
            "backend": "xnnpack" if xnnpack else "portable",
            "max_context": max_context,
            "decode_step": 1,
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
            "mel_bins": MEL_BINS,
            "log_eps": 1e-6,
            "frames_for_full_segment": MEL_FRAMES_5S,
        },
        "conditioning": {
            "instrument_token_count": 1,
            "dataset_token_count": 1,
            "default_instrument_token": 0,
            "default_dataset_token": 0,
            "prefix_order": ["self_wav", "dataset_name", "instrument_group"],
        },
        "vocabulary": {
            "generated_size": GENERATED_VOCAB_SIZE,
            "model_cardinality": cfg.card,
            "initial_token_id": cfg.card,
            "eos_token_id": 1,
        },
        "files": {
            "conditioner": conditioner_path.name,
            "embedder": embedder_path.name,
            "decoder": decoder_path.name,
        },
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
    p.add_argument("--max-context", type=int, default=DEFAULT_MAX_CONTEXT)
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
