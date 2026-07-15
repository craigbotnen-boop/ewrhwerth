#!/usr/bin/env python3
"""
BITNET EH-1 ARCHITECTURE SMOKE CORE
===================================

Reference model-construction layer for a matched four-arm native-training test:

    B_full     : full width D
    B_narrow   : narrow width N, standard RMS divisor N
    B_ambient  : narrow width N, fixed full-reference divisor D, E=0
    E_EH1      : narrow width N, full-reference divisor D, learned scalar E_hat(x)

Pinned Transformers source convention:
    096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4

The pinned BitNet MLP computes:
    h = relu2(gate_proj(x)) * up_proj(x)
    h = ffn_sub_norm(h)
    y = down_proj(h)

This module replaces only the FFN SubLN behavior for B_ambient / E_EH1 and
converts ordinary Linear projections to AutoBitLinear online QAT after matched
initialization has been established.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BitNetConfig, BitNetForCausalLM
from transformers.activations import ACT2FN
from transformers.integrations.bitnet import AutoBitLinear


EnergyMode = Literal[
    "normal",
    "zero",
    "mean",
    "shuffle_within_sequence",
    "shuffle_cross_sequence",
]


@dataclass(frozen=True)
class SmokeShape:
    vocab_size: int = 4096
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    d_ff_full: int = 2816
    d_ff_narrow: int = 2784
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0

    @property
    def removed_width(self) -> int:
        return self.d_ff_full - self.d_ff_narrow


class ReferenceWidthFFNSubLN(nn.Module):
    """Learned affine FFN RMSNorm with an explicit denominator reference width."""

    def __init__(
        self,
        transport_width: int,
        reference_width: int,
        eps: float,
    ) -> None:
        super().__init__()
        if transport_width <= 0:
            raise ValueError("transport_width must be positive.")
        if reference_width < transport_width:
            raise ValueError("reference_width must be >= transport_width.")
        self.transport_width = int(transport_width)
        self.reference_width = int(reference_width)
        self.variance_epsilon = float(eps)
        self.weight = nn.Parameter(torch.ones(transport_width))

    def forward(
        self,
        hidden_states: torch.Tensor,
        omitted_energy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[-1] != self.transport_width:
            raise ValueError(
                f"Expected last dim {self.transport_width}, "
                f"got {hidden_states.shape[-1]}."
            )

        input_dtype = hidden_states.dtype
        h = hidden_states.float()
        transport_energy = h.square().sum(dim=-1, keepdim=True)

        if omitted_energy is None:
            e = torch.zeros_like(transport_energy)
        else:
            if omitted_energy.shape != (*hidden_states.shape[:-1], 1):
                raise ValueError(
                    "omitted_energy must have shape hidden_states.shape[:-1] + (1,)."
                )
            e = omitted_energy.float()
            if bool((e < 0).any()):
                raise ValueError("omitted_energy must be non-negative.")

        mean_square = (transport_energy + e) / self.reference_width
        h = h * torch.rsqrt(mean_square + self.variance_epsilon)
        return self.weight * h.to(input_dtype)


def _valid_mask(
    x: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones(
            x.shape[:2],
            dtype=torch.bool,
            device=x.device,
        )
    if attention_mask.shape != x.shape[:2]:
        raise ValueError(
            f"attention_mask {tuple(attention_mask.shape)} "
            f"does not match token shape {tuple(x.shape[:2])}."
        )
    return attention_mask.to(device=x.device, dtype=torch.bool)


def intervene_on_energy(
    energy: torch.Tensor,
    mode: EnergyMode,
    attention_mask: torch.Tensor | None,
    *,
    seed: int,
) -> torch.Tensor:
    """Apply frozen evaluation-only interventions to [B,S,1] energy scalars."""
    if mode == "normal":
        return energy

    valid = _valid_mask(energy, attention_mask)
    out = energy.clone()
    values_2d = out[..., 0]

    if mode == "zero":
        values_2d[valid] = 0
        return out

    valid_values = values_2d[valid]
    if valid_values.numel() == 0:
        raise ValueError("No valid tokens available for energy intervention.")

    if mode == "mean":
        values_2d[valid] = valid_values.float().mean().to(valid_values.dtype)
        return out

    generator = torch.Generator(device=energy.device)
    generator.manual_seed(int(seed))

    if mode == "shuffle_cross_sequence":
        perm = torch.randperm(
            valid_values.numel(),
            generator=generator,
            device=energy.device,
        )
        values_2d[valid] = valid_values[perm]
        return out

    if mode == "shuffle_within_sequence":
        for batch_idx in range(values_2d.shape[0]):
            row_valid = valid[batch_idx]
            row_values = values_2d[batch_idx, row_valid]
            if row_values.numel() <= 1:
                continue
            perm = torch.randperm(
                row_values.numel(),
                generator=generator,
                device=energy.device,
            )
            values_2d[batch_idx, row_valid] = row_values[perm]
        return out

    raise ValueError(f"Unknown energy intervention mode: {mode}")


class AmbientDenominatorBitNetMLP(nn.Module):
    """Narrow gated BitNet MLP with full reference-width FFN SubLN and E=0."""

    def __init__(
        self,
        config: BitNetConfig,
        *,
        reference_width: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.ffn_sub_norm = ReferenceWidthFFNSubLN(
            transport_width=self.intermediate_size,
            reference_width=reference_width,
            eps=config.rms_norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        h = self.ffn_sub_norm(h, omitted_energy=None)
        return self.down_proj(h)


class EnergyAwareBitNetMLP(nn.Module):
    """Narrow gated BitNet MLP with a learned one-scalar omitted-energy head."""

    def __init__(
        self,
        config: BitNetConfig,
        *,
        reference_width: int,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.reference_width = int(reference_width)
        self.removed_width = self.reference_width - self.intermediate_size
        self.layer_idx = int(layer_idx)

        if self.removed_width <= 0:
            raise ValueError("Energy arm requires reference_width > intermediate_size.")

        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
        )
        self.energy_proj = nn.Linear(self.hidden_size, 1, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]
        self.ffn_sub_norm = ReferenceWidthFFNSubLN(
            transport_width=self.intermediate_size,
            reference_width=self.reference_width,
            eps=config.rms_norm_eps,
        )

        self.energy_mode: EnergyMode = "normal"
        self.energy_attention_mask: torch.Tensor | None = None
        self.energy_seed: int = 0

    def set_energy_intervention(
        self,
        mode: EnergyMode,
        *,
        attention_mask: torch.Tensor | None = None,
        seed: int = 0,
    ) -> None:
        self.energy_mode = mode
        self.energy_attention_mask = attention_mask
        self.energy_seed = int(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        # Positive aggregate omitted-energy estimate.
        e_hat = self.removed_width * F.softplus(self.energy_proj(x))
        e_hat = intervene_on_energy(
            e_hat,
            self.energy_mode,
            self.energy_attention_mask,
            seed=self.energy_seed + self.layer_idx,
        )

        h = self.ffn_sub_norm(h, omitted_energy=e_hat)
        return self.down_proj(h)


def make_config(shape: SmokeShape, intermediate_size: int) -> BitNetConfig:
    return BitNetConfig(
        vocab_size=shape.vocab_size,
        hidden_size=shape.hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=shape.num_hidden_layers,
        num_attention_heads=shape.num_attention_heads,
        num_key_value_heads=shape.num_key_value_heads,
        hidden_act="relu2",
        max_position_embeddings=shape.max_position_embeddings,
        rms_norm_eps=shape.rms_norm_eps,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=shape.rope_theta,
        attention_bias=False,
        attention_dropout=0.0,
    )


def _replace_mlp_modules(
    model: BitNetForCausalLM,
    *,
    arm_name: str,
    reference_width: int,
) -> None:
    if arm_name not in {"B_ambient", "E_EH1"}:
        return

    for layer_idx, layer in enumerate(model.model.layers):
        if arm_name == "B_ambient":
            layer.mlp = AmbientDenominatorBitNetMLP(
                model.config,
                reference_width=reference_width,
            )
        else:
            layer.mlp = EnergyAwareBitNetMLP(
                model.config,
                reference_width=reference_width,
                layer_idx=layer_idx,
            )


def _slice_from_full(
    name: str,
    source: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor | None:
    if name.endswith("mlp.gate_proj.weight") or name.endswith("mlp.up_proj.weight"):
        if source.ndim == 2 and tuple(target_shape[1:]) == tuple(source.shape[1:]):
            return source[: target_shape[0], :]
    if name.endswith("mlp.down_proj.weight"):
        if source.ndim == 2 and target_shape[0] == source.shape[0]:
            return source[:, : target_shape[1]]
    if name.endswith("mlp.ffn_sub_norm.weight"):
        if source.ndim == 1:
            return source[: target_shape[0]]
    return None


@torch.no_grad()
def copy_matched_initialization(
    full: BitNetForCausalLM,
    target: BitNetForCausalLM,
    *,
    energy_seed_base: int,
) -> None:
    full_state = full.state_dict()
    target_state = target.state_dict()

    for name, target_tensor in target_state.items():
        if name.endswith("mlp.energy_proj.weight"):
            continue

        source = full_state.get(name)
        if source is None:
            raise KeyError(f"Target parameter has no full-arm source: {name}")

        if source.shape == target_tensor.shape:
            target_tensor.copy_(source)
            continue

        sliced = _slice_from_full(name, source, target_tensor.shape)
        if sliced is None or sliced.shape != target_tensor.shape:
            raise ValueError(
                f"No declared matched-init rule for {name}: "
                f"source={tuple(source.shape)}, target={tuple(target_tensor.shape)}"
            )
        target_tensor.copy_(sliced)

    target.load_state_dict(target_state, strict=True)

    # Deterministic, arm-specific EH-1 initialization after all common tensors match.
    for layer_idx, layer in enumerate(target.model.layers):
        if isinstance(layer.mlp, EnergyAwareBitNetMLP):
            generator = torch.Generator(device=layer.mlp.energy_proj.weight.device)
            generator.manual_seed(int(energy_seed_base + layer_idx))
            init = torch.empty_like(layer.mlp.energy_proj.weight)
            init.normal_(
                mean=0.0,
                std=target.config.initializer_range,
                generator=generator,
            )
            layer.mlp.energy_proj.weight.copy_(init)


@torch.no_grad()
def assert_matched_initialization(
    full: BitNetForCausalLM,
    target: BitNetForCausalLM,
) -> None:
    full_state = full.state_dict()
    target_state = target.state_dict()

    for name, target_tensor in target_state.items():
        if name.endswith("mlp.energy_proj.weight"):
            continue
        source = full_state[name]
        expected = (
            source
            if source.shape == target_tensor.shape
            else _slice_from_full(name, source, target_tensor.shape)
        )
        if expected is None:
            raise AssertionError(f"No match assertion rule for {name}")
        if not torch.equal(expected, target_tensor):
            raise AssertionError(f"Matched initialization failed for {name}")


def convert_to_online_bitlinear(
    module: nn.Module,
    *,
    prefix: str = "",
    skip_fragments: Iterable[str] = ("lm_head", "energy_proj"),
) -> None:
    """
    Recursively replace nn.Linear with AutoBitLinear(online_quant=True), copying
    master weights exactly. The EH-1 energy row and LM head remain ordinary Linear.
    """
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if isinstance(child, AutoBitLinear):
            continue

        if isinstance(child, nn.Linear) and not any(
            fragment in full_name for fragment in skip_fragments
        ):
            replacement = AutoBitLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
                online_quant=True,
            )
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                if child.bias is not None and replacement.bias is not None:
                    replacement.bias.copy_(child.bias)
            setattr(module, child_name, replacement)
            continue

        convert_to_online_bitlinear(
            child,
            prefix=full_name,
            skip_fragments=skip_fragments,
        )


def build_matched_arms(
    *,
    shape: SmokeShape = SmokeShape(),
    initialization_seed: int = 42,
    energy_seed_base: int = 42000,
) -> Dict[str, BitNetForCausalLM]:
    torch.manual_seed(initialization_seed)

    full = BitNetForCausalLM(make_config(shape, shape.d_ff_full))
    narrow = BitNetForCausalLM(make_config(shape, shape.d_ff_narrow))
    ambient = BitNetForCausalLM(make_config(shape, shape.d_ff_narrow))
    energy = BitNetForCausalLM(make_config(shape, shape.d_ff_narrow))

    _replace_mlp_modules(
        ambient,
        arm_name="B_ambient",
        reference_width=shape.d_ff_full,
    )
    _replace_mlp_modules(
        energy,
        arm_name="E_EH1",
        reference_width=shape.d_ff_full,
    )

    copy_matched_initialization(
        full,
        narrow,
        energy_seed_base=energy_seed_base,
    )
    copy_matched_initialization(
        full,
        ambient,
        energy_seed_base=energy_seed_base,
    )
    copy_matched_initialization(
        full,
        energy,
        energy_seed_base=energy_seed_base,
    )

    assert_matched_initialization(full, narrow)
    assert_matched_initialization(full, ambient)
    assert_matched_initialization(full, energy)

    arms = {
        "B_full": full,
        "B_narrow": narrow,
        "B_ambient": ambient,
        "E_EH1": energy,
    }

    for model in arms.values():
        convert_to_online_bitlinear(model)

    return arms


def set_energy_intervention(
    model: BitNetForCausalLM,
    mode: EnergyMode,
    *,
    attention_mask: torch.Tensor | None = None,
    seed: int = 0,
) -> None:
    for layer in model.model.layers:
        if isinstance(layer.mlp, EnergyAwareBitNetMLP):
            layer.mlp.set_energy_intervention(
                mode,
                attention_mask=attention_mask,
                seed=seed,
            )


def projection_dividend(shape: SmokeShape) -> dict[str, int]:
    """
    Net parameterized projection connections removed by N=D-k:
        gate: d_model*k
        up:   d_model*k
        down: d_model*k
        EH1 cost: d_model
    """
    per_layer = (
        3 * shape.hidden_size * shape.removed_width
        - shape.hidden_size
    )
    return {
        "per_layer_net_projection_connections_removed": per_layer,
        "all_layers_net_projection_connections_removed": (
            per_layer * shape.num_hidden_layers
        ),
    }


if __name__ == "__main__":
    shape = SmokeShape()
    arms = build_matched_arms(shape=shape)
    print("BUILT_ARMS", sorted(arms))
    print("DIVIDEND", projection_dividend(shape))
    print("MATCHED_INITIALIZATION_PASS")
