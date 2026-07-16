#!/usr/bin/env python3
"""Fail-closed four-arm end-to-end trainer implementation smoke.

This is deliberately tiny and scientifically ineligible. It verifies that the
current matched four-arm core can execute repeated forward/backward/gradient
scan/clip/optimizer steps on identical structured autoregressive batches and
that the E_EH1 intervention path remains wired after training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from bitnet_energy_head_smoke_core import (
    SmokeShape,
    build_matched_arms,
    initialization_receipt,
    set_energy_intervention,
)

CAMPAIGN = "BITNET_EH1_ARCHITECTURE_SMOKE_V1"
SMOKE_ID = "BITNET_EH1_FOUR_ARM_TINY_TRAINER_SMOKE_V1"
ARMS = ("B_full", "B_narrow", "B_ambient", "E_EH1")


def structured_windows(
    *, batch_size: int, sequence_length: int, vocab_size: int, offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic context-dependent source; explicitly not IID tokens."""
    raw = torch.empty(batch_size, sequence_length + 1, dtype=torch.long)
    batch_index = torch.arange(batch_size, dtype=torch.long)
    raw[:, 0] = (17 * batch_index + 3 + offset) % vocab_size
    for t in range(1, sequence_length + 1):
        mode = (batch_index + (t // 7)) % 3
        multiplier = 3 + 2 * mode
        raw[:, t] = (
            multiplier * raw[:, t - 1]
            + 7 * (t % 5)
            + 11 * batch_index
            + offset
        ) % vocab_size
    return raw[:, :-1].contiguous(), raw[:, 1:].contiguous()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def make_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "model.embed_tokens.weight" or ".energy_proj." in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1.0e-3,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        foreach=False,
        fused=False,
    )


def explicit_next_token_nll(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    attention_mask = torch.ones_like(inputs)
    logits = model(
        input_ids=inputs,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )
    return loss, logits


def scan_gradients(model: torch.nn.Module) -> dict[str, float | int | bool]:
    total_nonfinite = 0
    tensor_count = 0
    finite_value_count = 0
    max_abs_finite = 0.0
    sum_squares = 0.0
    for parameter in model.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        tensor_count += 1
        grad_float = grad.detach().float()
        finite = torch.isfinite(grad_float)
        total_nonfinite += int((~finite).sum().item())
        if bool(finite.any()):
            finite_values = grad_float[finite]
            finite_value_count += int(finite_values.numel())
            max_abs_finite = max(
                max_abs_finite,
                float(finite_values.abs().max().item()),
            )
            sum_squares += float(finite_values.double().square().sum().item())
    grad_norm = math.sqrt(sum_squares)
    return {
        "gradient_tensor_count": tensor_count,
        "finite_gradient_value_count": finite_value_count,
        "total_nonfinite_gradient_values": total_nonfinite,
        "max_abs_finite_gradient": max_abs_finite,
        "grad_norm_preclip": grad_norm,
        "grad_norm_preclip_finite": math.isfinite(grad_norm),
    }


def parameter_delta_norm(
    model: torch.nn.Module,
    initial: dict[str, torch.Tensor],
) -> float:
    sum_squares = 0.0
    for name, parameter in model.named_parameters():
        delta = parameter.detach().float() - initial[name].float()
        sum_squares += float(delta.double().square().sum().item())
    return math.sqrt(sum_squares)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("BITNET_EH1_FOUR_ARM_TINY_TRAINER_SMOKE_V1.json"),
    )
    args = parser.parse_args()

    receipt: dict[str, object] = {
        "campaign": CAMPAIGN,
        "smoke_id": SMOKE_ID,
        "scope": "IMPLEMENTATION_ONLY_NON_VERDICT",
        "device": "cpu",
        "torch_version": torch.__version__,
    }

    try:
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        torch.manual_seed(20260715)

        shape = SmokeShape(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,
            d_ff_full=96,
            d_ff_narrow=88,
            max_position_embeddings=64,
        )
        batch_size = 4
        sequence_length = 48
        update_count = 3
        batches = [
            structured_windows(
                batch_size=batch_size,
                sequence_length=sequence_length,
                vocab_size=shape.vocab_size,
                offset=offset,
            )
            for offset in (0, 13, 29)
        ]
        batch_receipts = [
            {
                "update": index + 1,
                "input_sha256": tensor_sha256(inputs),
                "target_sha256": tensor_sha256(targets),
            }
            for index, (inputs, targets) in enumerate(batches)
        ]

        arms = build_matched_arms(
            shape=shape,
            initialization_seed=17,
            energy_seed_base=17000,
        )
        if tuple(arms) != ARMS:
            raise RuntimeError(f"Unexpected arm order: {tuple(arms)}")

        arm_results: list[dict[str, object]] = []
        trained_energy_model = None

        for arm_name in ARMS:
            model = arms[arm_name].cpu()
            model.train()
            if arm_name == "E_EH1":
                set_energy_intervention(model, "normal")
            initial = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
            }
            optimizer = make_optimizer(model)
            update_rows = []

            for update_index, (inputs, targets) in enumerate(batches, start=1):
                optimizer.zero_grad(set_to_none=True)
                loss, _ = explicit_next_token_nll(model, inputs, targets)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"{arm_name} update {update_index}: non-finite loss {loss.item()}"
                    )
                loss.backward()
                gradient_receipt = scan_gradients(model)
                if gradient_receipt["total_nonfinite_gradient_values"] != 0:
                    raise FloatingPointError(
                        f"{arm_name} update {update_index}: non-finite gradient values"
                    )
                if not gradient_receipt["grad_norm_preclip_finite"]:
                    raise FloatingPointError(
                        f"{arm_name} update {update_index}: non-finite gradient norm"
                    )
                clipped_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                update_rows.append(
                    {
                        "update": update_index,
                        "nll": float(loss.item()),
                        "clip_grad_norm_return": float(clipped_norm.item()),
                        **gradient_receipt,
                    }
                )

            delta_norm = parameter_delta_norm(model, initial)
            if not math.isfinite(delta_norm) or delta_norm <= 0.0:
                raise RuntimeError(
                    f"{arm_name}: parameters did not move finitely; delta={delta_norm}"
                )
            model.eval()
            with torch.inference_mode():
                final_loss, final_logits = explicit_next_token_nll(
                    model, batches[-1][0], batches[-1][1]
                )
            if not bool(torch.isfinite(final_logits).all()):
                raise FloatingPointError(f"{arm_name}: non-finite final logits")
            arm_results.append(
                {
                    "arm": arm_name,
                    "updates": update_rows,
                    "parameter_delta_l2": delta_norm,
                    "final_batch_nll": float(final_loss.item()),
                    "final_logits_finite": True,
                }
            )
            if arm_name == "E_EH1":
                trained_energy_model = model

        if trained_energy_model is None:
            raise RuntimeError("E_EH1 model was not retained.")

        intervention_input, intervention_target = batches[-1]
        attention_mask = torch.ones_like(intervention_input)
        intervention_rows = []
        with torch.inference_mode():
            set_energy_intervention(
                trained_energy_model,
                "normal",
                attention_mask=attention_mask,
                seed=8009,
            )
            normal_loss, normal_logits = explicit_next_token_nll(
                trained_energy_model,
                intervention_input,
                intervention_target,
            )
            for mode in (
                "zero",
                "mean",
                "shuffle_within_sequence",
                "shuffle_cross_sequence",
            ):
                set_energy_intervention(
                    trained_energy_model,
                    mode,
                    attention_mask=attention_mask,
                    seed=8009,
                )
                alt_loss, alt_logits = explicit_next_token_nll(
                    trained_energy_model,
                    intervention_input,
                    intervention_target,
                )
                if not bool(torch.isfinite(alt_logits).all()):
                    raise FloatingPointError(
                        f"E_EH1 intervention {mode}: non-finite logits"
                    )
                intervention_rows.append(
                    {
                        "mode": mode,
                        "nll": float(alt_loss.item()),
                        "delta_nll_vs_normal": float(
                            alt_loss.item() - normal_loss.item()
                        ),
                        "max_abs_logit_delta_vs_normal": float(
                            (alt_logits - normal_logits).abs().max().item()
                        ),
                    }
                )

        if not any(
            row["max_abs_logit_delta_vs_normal"] > 0.0
            for row in intervention_rows
        ):
            raise RuntimeError("EH-1 intervention path produced no logit change.")

        receipt.update(
            {
                "shape": {
                    "vocab_size": shape.vocab_size,
                    "hidden_size": shape.hidden_size,
                    "num_hidden_layers": shape.num_hidden_layers,
                    "num_attention_heads": shape.num_attention_heads,
                    "num_key_value_heads": shape.num_key_value_heads,
                    "d_ff_full": shape.d_ff_full,
                    "d_ff_narrow": shape.d_ff_narrow,
                    "sequence_length": sequence_length,
                    "batch_size": batch_size,
                },
                "initialization": initialization_receipt(shape),
                "matched_initialization_assertions": "PASS_IN_CORE_BEFORE_QAT_CONVERSION",
                "input_process": "deterministic structured context-dependent recurrence; not IID",
                "shared_batch_receipts": batch_receipts,
                "optimizer": {
                    "type": "AdamW",
                    "learning_rate": 0.001,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                    "embedding_and_energy_head_weight_decay": 0.0,
                    "gradient_clip_norm": 1.0,
                },
                "arm_results": arm_results,
                "eh1_interventions": {
                    "normal_nll": float(normal_loss.item()),
                    "rows": intervention_rows,
                },
                "verdict": "FOUR_ARM_TINY_END_TO_END_TRAINER_SMOKE_PASS_NON_VERDICT",
                "claim_ceiling": "IMPLEMENTATION ONLY; no WIDTH_PENALTY, HEAD_USE, TOKEN_CONDITIONING, HEAD_INCREMENT, or RECOVERY inference",
            }
        )
    except Exception as exc:
        receipt.update(
            {
                "verdict": "FOUR_ARM_TINY_END_TO_END_TRAINER_SMOKE_FAIL",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "claim_ceiling": "IMPLEMENTATION FAILURE ONLY; no scientific inference",
            }
        )
        args.output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        raise

    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
