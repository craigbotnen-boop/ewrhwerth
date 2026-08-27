#!/usr/bin/env python3
"""Fail-closed entry point for BITNET_EH1_ARCHITECTURE_SMOKE_V1.

This wrapper is the only authorized scientific-training entry point. It keeps
`train_architecture_smoke.py` as the implementation runner, but inserts an
explicit pre-optimizer gradient-finiteness firewall around its clipping call.
No non-finite gradient may be clipped, zeroed, ignored, or stepped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import torch

import train_architecture_smoke as runner
from bitnet_energy_head_smoke_core import SmokeShape, initialization_receipt


PARAMETER_NAMES: dict[int, str] = {}
ORIGINAL_MAKE_OPTIMIZER = runner.make_optimizer
ORIGINAL_CLIP_GRAD_NORM = torch.nn.utils.clip_grad_norm_


def _output_dir_from_argv() -> Path | None:
    try:
        index = sys.argv.index("--output-dir")
        return Path(sys.argv[index + 1])
    except (ValueError, IndexError):
        return None


def make_optimizer_with_name_registry(
    model: torch.nn.Module,
) -> tuple[torch.optim.Optimizer, dict[str, list[str]]]:
    PARAMETER_NAMES.clear()
    PARAMETER_NAMES.update({id(parameter): name for name, parameter in model.named_parameters()})
    return ORIGINAL_MAKE_OPTIMIZER(model)


def fail_closed_clip_grad_norm(
    parameters: Iterable[torch.Tensor],
    max_norm: float,
    *args,
    **kwargs,
) -> torch.Tensor:
    params = list(parameters)
    bad = []
    total_nonfinite = 0
    for parameter in params:
        gradient = parameter.grad
        if gradient is None:
            continue
        finite = torch.isfinite(gradient)
        nonfinite = int((~finite).sum().item())
        if nonfinite:
            total_nonfinite += nonfinite
            bad.append(
                {
                    "name": PARAMETER_NAMES.get(id(parameter), "<unregistered>"),
                    "nonfinite_gradient_values": nonfinite,
                    "gradient_values": int(gradient.numel()),
                    "gradient_dtype": str(gradient.dtype),
                }
            )

    if bad:
        event = {
            "event": "NONFINITE_GRADIENT_BLOCKER",
            "bad_parameter_gradient_tensor_count": len(bad),
            "total_nonfinite_gradient_values": total_nonfinite,
            "bad_parameters": bad,
            "disposition": "FAIL_CLOSED_BEFORE_CLIP_AND_OPTIMIZER_STEP",
        }
        print(json.dumps(event, sort_keys=True), flush=True)
        raise RuntimeError(json.dumps(event, sort_keys=True))

    safe_kwargs = dict(kwargs)
    safe_kwargs["error_if_nonfinite"] = True
    grad_norm = ORIGINAL_CLIP_GRAD_NORM(
        params,
        max_norm,
        *args,
        **safe_kwargs,
    )
    if not bool(torch.isfinite(grad_norm)):
        event = {
            "event": "NONFINITE_GRAD_NORM_BLOCKER",
            "grad_norm": float(grad_norm),
            "disposition": "FAIL_CLOSED_BEFORE_OPTIMIZER_STEP",
        }
        print(json.dumps(event, sort_keys=True), flush=True)
        raise RuntimeError(json.dumps(event, sort_keys=True))
    return grad_norm


def main() -> None:
    output_dir = _output_dir_from_argv()
    init_receipt = initialization_receipt(SmokeShape())
    safety_receipt = {
        "campaign": runner.CAMPAIGN,
        "entry_point": "run_architecture_smoke_failclosed.py",
        "gradient_firewall": "all parameter gradients scanned with torch.isfinite immediately before clip_grad_norm_; any non-finite value raises before clipping and optimizer.step",
        "clip_grad_norm_error_if_nonfinite": True,
        "initialization": init_receipt,
        "status": "ARMED",
    }
    print(
        json.dumps(
            {"event": "EXECUTION_SAFETY_FIREWALL_ARMED", **safety_receipt},
            sort_keys=True,
        ),
        flush=True,
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "EXECUTION_SAFETY_RECEIPT.json").write_text(
            json.dumps(safety_receipt, indent=2, sort_keys=True)
        )

    runner.make_optimizer = make_optimizer_with_name_registry
    torch.nn.utils.clip_grad_norm_ = fail_closed_clip_grad_norm
    runner.main()

    safety_receipt["status"] = "COMPLETED_WITHOUT_GRADIENT_FIREWALL_TRIGGER"
    if output_dir is not None:
        (output_dir / "EXECUTION_SAFETY_RECEIPT.json").write_text(
            json.dumps(safety_receipt, indent=2, sort_keys=True)
        )
    print(
        json.dumps(
            {"event": "EXECUTION_SAFETY_FIREWALL_CLOSEOUT", **safety_receipt},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
