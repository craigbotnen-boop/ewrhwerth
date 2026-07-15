#!/usr/bin/env python3
"""Execute one arm/seed of BITNET_EH1_ARCHITECTURE_SMOKE_V1.

The scientific run uses the frozen uint16 token streams and frozen schedules.
A shortened `--max-updates` run is explicitly non-verdict and is intended only
for execution preflight.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from bitnet_energy_head_smoke_core import (
    EnergyAwareBitNetMLP,
    SmokeShape,
    build_matched_arms,
    projection_dividend,
    set_energy_intervention,
)

CAMPAIGN = "BITNET_EH1_ARCHITECTURE_SMOKE_V1"
TRANSFORMERS_COMMIT = "096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4"
ARMS = ("B_full", "B_narrow", "B_ambient", "E_EH1")
PAIRED_SEEDS = (17, 29, 41)
SEQ = 2048
WINDOW = 2049
GLOBAL_BATCH_TOKENS = 32768
SEQUENCES_PER_UPDATE = 16
TOTAL_UPDATES = 1526
REALIZED_TRAINING_TOKENS = TOTAL_UPDATES * GLOBAL_BATCH_TOKENS
WARMUP_FRACTION = 0.05
WARMUP_TOKENS = WARMUP_FRACTION * REALIZED_TRAINING_TOKENS
MAX_LR = 1.5e-3
MIN_LR = 1.5e-4
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
GRAD_CLIP = 1.0
VALIDATION_TOKEN_INTERVAL = 1_000_000
SHUFFLE_REPLICATE_SEEDS = (1009, 2017, 3011, 4001, 5003, 6007, 7001, 8009)
SOURCE_FREEZE_VERDICT = "CORPUS_TOKENIZER_AND_SCHEDULE_FREEZE_PASS"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_code_commit() -> str | None:
    env_commit = os.environ.get("BITNET_EH1_CODE_COMMIT")
    if env_commit:
        return env_commit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def lr_at_tokens(tokens_after_update: int) -> float:
    """Token-clock schedule frozen at the post-batch/pre-optimizer boundary."""
    t = float(tokens_after_update)
    total = float(REALIZED_TRAINING_TOKENS)
    if t <= WARMUP_TOKENS:
        return MAX_LR * (t / WARMUP_TOKENS)
    progress = (t - WARMUP_TOKENS) / (total - WARMUP_TOKENS)
    progress = min(max(progress, 0.0), 1.0)
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))


def validation_updates(max_updates: int) -> set[int]:
    checkpoints: set[int] = set()
    threshold = VALIDATION_TOKEN_INTERVAL
    while threshold <= max_updates * GLOBAL_BATCH_TOKENS:
        checkpoints.add(min(max_updates, math.ceil(threshold / GLOBAL_BATCH_TOKENS)))
        threshold += VALIDATION_TOKEN_INTERVAL
    checkpoints.add(max_updates)
    return checkpoints


def verify_frozen_inputs(root: Path, seed: int) -> dict[str, object]:
    closeout_path = root / "FREEZE_CLOSEOUT_RECEIPT.json"
    corpus_path = root / "tokenized" / "CORPUS_FREEZE_MANIFEST.json"
    schedule_path = root / "schedules" / "BATCH_SCHEDULE_MANIFEST.json"
    amendment_path = root / "source_jsonl" / "SOURCE_SUBSTITUTION_AMENDMENT.json"
    for path in (closeout_path, corpus_path, schedule_path, amendment_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    closeout = json_load(closeout_path)
    corpus = json_load(corpus_path)
    schedules = json_load(schedule_path)
    amendment = json_load(amendment_path)
    if closeout.get("verdict") != SOURCE_FREEZE_VERDICT:
        raise RuntimeError(f"Frozen input verdict is not PASS: {closeout.get('verdict')}")
    if schedules.get("global_batch_tokens") != GLOBAL_BATCH_TOKENS:
        raise RuntimeError("Global batch-token contract mismatch.")
    if schedules.get("sequence_length") != SEQ:
        raise RuntimeError("Sequence-length contract mismatch.")
    if schedules.get("optimizer_updates") != TOTAL_UPDATES:
        raise RuntimeError("Optimizer-update contract mismatch.")
    if schedules.get("realized_training_tokens") != REALIZED_TRAINING_TOKENS:
        raise RuntimeError("Training-token contract mismatch.")
    if schedules.get("allow_train_window_reuse") is not False:
        raise RuntimeError("Train window reuse must be false.")

    train_bin = root / "tokenized" / "train.tokens.uint16"
    validation_bin = root / "tokenized" / "validation.tokens.uint16"
    train_schedule = root / "schedules" / f"train_schedule_seed_{seed}.npy"
    validation_schedule = root / "schedules" / "validation_schedule.npy"
    expected = {
        train_bin: corpus["train"]["token_bin_sha256"],
        validation_bin: corpus["validation"]["token_bin_sha256"],
        validation_schedule: schedules["validation"]["sha256"],
    }
    seed_schedule = next(
        entry for entry in schedules["train_schedules"] if entry["seed"] == seed
    )
    expected[train_schedule] = seed_schedule["sha256"]
    actual_hashes = {}
    for path, expected_sha in expected.items():
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"SHA256 mismatch for {path}: {actual_sha} != {expected_sha}")
        actual_hashes[str(path)] = actual_sha

    train_offsets = np.load(train_schedule, allow_pickle=False)
    validation_offsets = np.load(validation_schedule, allow_pickle=False)
    if train_offsets.shape != (TOTAL_UPDATES, SEQUENCES_PER_UPDATE):
        raise RuntimeError(f"Unexpected train schedule shape: {train_offsets.shape}")
    if validation_offsets.shape != (2048,):
        raise RuntimeError(f"Unexpected validation schedule shape: {validation_offsets.shape}")

    return {
        "closeout": closeout,
        "corpus": corpus,
        "schedules": schedules,
        "amendment": amendment,
        "actual_hashes": actual_hashes,
        "train_bin": train_bin,
        "validation_bin": validation_bin,
        "train_schedule": train_schedule,
        "validation_schedule": validation_schedule,
    }


def windows_from_offsets(
    token_memmap: np.memmap,
    offsets: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    windows = []
    for offset in offsets:
        start = int(offset)
        raw = np.asarray(token_memmap[start : start + WINDOW], dtype=np.int64)
        if raw.shape[0] != WINDOW:
            raise RuntimeError(f"Short source window at offset {start}.")
        windows.append(raw)
    batch = np.stack(windows, axis=0)
    return torch.from_numpy(batch[:, :-1].copy()), torch.from_numpy(batch[:, 1:].copy())


def chunked(values: Sequence[int], size: int) -> Iterator[Sequence[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def energy_modules(model: torch.nn.Module) -> list[EnergyAwareBitNetMLP]:
    modules: list[EnergyAwareBitNetMLP] = []
    for layer in model.model.layers:
        if isinstance(layer.mlp, EnergyAwareBitNetMLP):
            modules.append(layer.mlp)
    return modules


def make_optimizer(model: torch.nn.Module) -> tuple[torch.optim.Optimizer, dict[str, list[str]]]:
    decay_params = []
    no_decay_params = []
    decay_names = []
    no_decay_names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        exempt = name == "model.embed_tokens.weight" or ".energy_proj." in name
        if exempt:
            no_decay_params.append(parameter)
            no_decay_names.append(name)
        else:
            decay_params.append(parameter)
            decay_names.append(name)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=0.0,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        foreach=False,
        fused=False,
    )
    return optimizer, {"decay": decay_names, "no_decay": no_decay_names}


def cross_entropy_sum(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="sum",
    )


@torch.inference_mode()
def evaluate_nll(
    model: torch.nn.Module,
    token_memmap: np.memmap,
    offsets: np.ndarray,
    *,
    device: torch.device,
    batch_sequences: int,
    external_provider: Callable[[int, int, int, torch.device], torch.Tensor] | None = None,
) -> float:
    model.eval()
    modules = energy_modules(model)
    state = {"start": 0, "end": 0}
    handles = []
    if external_provider is not None:
        if not modules:
            raise RuntimeError("External energy override requires E_EH1.")
        set_energy_intervention(model, "normal")
        for layer_idx, mlp in enumerate(modules):
            def hook(module, args, kwargs, layer_idx=layer_idx):
                replacement = external_provider(
                    layer_idx,
                    state["start"],
                    state["end"],
                    args[0].device,
                )
                updated = dict(kwargs)
                updated["omitted_energy"] = replacement
                return args, updated
            handles.append(
                mlp.ffn_sub_norm.register_forward_pre_hook(hook, with_kwargs=True)
            )

    loss_sum = 0.0
    token_count = 0
    try:
        for start in range(0, len(offsets), batch_sequences):
            end = min(start + batch_sequences, len(offsets))
            state["start"], state["end"] = start, end
            inputs, targets = windows_from_offsets(token_memmap, offsets[start:end])
            inputs = inputs.to(device=device, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=inputs, use_cache=False).logits
            batch_loss = cross_entropy_sum(logits, targets)
            loss_sum += float(batch_loss.item())
            token_count += int(targets.numel())
    finally:
        for handle in handles:
            handle.remove()
    return loss_sum / token_count


@torch.inference_mode()
def evaluate_and_capture_energy(
    model: torch.nn.Module,
    token_memmap: np.memmap,
    offsets: np.ndarray,
    *,
    device: torch.device,
    batch_sequences: int,
) -> tuple[float, list[torch.Tensor]]:
    model.eval()
    set_energy_intervention(model, "normal")
    modules = energy_modules(model)
    if not modules:
        raise RuntimeError("Energy capture requires E_EH1.")
    chunks: list[list[torch.Tensor]] = [[] for _ in modules]
    handles = []
    for layer_idx, mlp in enumerate(modules):
        def hook(module, args, kwargs, layer_idx=layer_idx):
            energy = kwargs.get("omitted_energy")
            if energy is None:
                raise RuntimeError("Normal E_EH1 forward supplied no omitted_energy.")
            chunks[layer_idx].append(energy.detach().float().cpu())
        handles.append(mlp.ffn_sub_norm.register_forward_pre_hook(hook, with_kwargs=True))

    loss_sum = 0.0
    token_count = 0
    try:
        for start in range(0, len(offsets), batch_sequences):
            end = min(start + batch_sequences, len(offsets))
            inputs, targets = windows_from_offsets(token_memmap, offsets[start:end])
            inputs = inputs.to(device=device, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=inputs, use_cache=False).logits
            batch_loss = cross_entropy_sum(logits, targets)
            loss_sum += float(batch_loss.item())
            token_count += int(targets.numel())
    finally:
        for handle in handles:
            handle.remove()
    banks = [torch.cat(layer_chunks, dim=0).contiguous() for layer_chunks in chunks]
    expected_shape = (len(offsets), SEQ, 1)
    if any(tuple(bank.shape) != expected_shape for bank in banks):
        raise RuntimeError(f"Captured energy-bank shape mismatch; expected {expected_shape}.")
    return loss_sum / token_count, banks


def provider_zero(
    normal_banks: list[torch.Tensor],
) -> Callable[[int, int, int, torch.device], torch.Tensor]:
    def provider(layer_idx: int, start: int, end: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((end - start, SEQ, 1), dtype=torch.float32, device=device)
    return provider


def provider_mean(
    normal_banks: list[torch.Tensor],
) -> Callable[[int, int, int, torch.device], torch.Tensor]:
    means = [float(bank.mean().item()) for bank in normal_banks]
    def provider(layer_idx: int, start: int, end: int, device: torch.device) -> torch.Tensor:
        return torch.full(
            (end - start, SEQ, 1),
            means[layer_idx],
            dtype=torch.float32,
            device=device,
        )
    return provider


def provider_bank(
    banks: list[torch.Tensor],
) -> Callable[[int, int, int, torch.device], torch.Tensor]:
    def provider(layer_idx: int, start: int, end: int, device: torch.device) -> torch.Tensor:
        return banks[layer_idx][start:end].to(device=device, non_blocking=True)
    return provider


def make_cross_sequence_shuffle(
    normal_banks: list[torch.Tensor], replicate_seed: int
) -> list[torch.Tensor]:
    shuffled = []
    for layer_idx, bank in enumerate(normal_banks):
        values = bank.reshape(-1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(replicate_seed + layer_idx))
        permutation = torch.randperm(values.numel(), generator=generator)
        shuffled.append(values[permutation].reshape_as(bank).contiguous())
    return shuffled


def make_within_sequence_shuffle(
    normal_banks: list[torch.Tensor], replicate_seed: int
) -> list[torch.Tensor]:
    shuffled = []
    sequence_index = torch.arange(normal_banks[0].shape[0])[:, None]
    for layer_idx, bank in enumerate(normal_banks):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(replicate_seed + layer_idx))
        permutation = torch.rand(
            (bank.shape[0], bank.shape[1]), generator=generator
        ).argsort(dim=1)
        values = bank[..., 0]
        permuted = values[sequence_index, permutation].unsqueeze(-1).contiguous()
        shuffled.append(permuted)
    return shuffled


def energy_bank_stats(normal_banks: list[torch.Tensor]) -> list[dict[str, float]]:
    result = []
    for layer_idx, bank in enumerate(normal_banks):
        values = bank.reshape(-1)
        quantiles = torch.quantile(values, torch.tensor([0.01, 0.5, 0.99]))
        result.append(
            {
                "layer_idx": layer_idx,
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "min": float(values.min().item()),
                "q01": float(quantiles[0].item()),
                "median": float(quantiles[1].item()),
                "q99": float(quantiles[2].item()),
                "max": float(values.max().item()),
            }
        )
    return result


def save_checkpoint(model: torch.nn.Module, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    save_file(state, str(path))
    del state
    gc.collect()
    return sha256_file(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=PAIRED_SEEDS, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=TOTAL_UPDATES)
    parser.add_argument("--microbatch-sequences", type=int, default=1)
    parser.add_argument("--eval-batch-sequences", type=int, default=4)
    parser.add_argument("--skip-periodic-eval", action="store_true")
    parser.add_argument("--skip-final-interventions", action="store_true")
    parser.add_argument("--upload-repo", default=None)
    parser.add_argument("--upload-prefix", default="runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Scientific runner requires CUDA with bf16 support.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not report bf16 support.")
    if not 1 <= args.max_updates <= TOTAL_UPDATES:
        raise ValueError("max_updates must be in [1, 1526].")
    if SEQUENCES_PER_UPDATE % args.microbatch_sequences != 0:
        raise ValueError("microbatch_sequences must divide 16 exactly.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    frozen = verify_frozen_inputs(args.frozen_root, args.seed)
    train_tokens = np.memmap(frozen["train_bin"], mode="r", dtype=np.uint16)
    validation_tokens = np.memmap(frozen["validation_bin"], mode="r", dtype=np.uint16)
    train_offsets = np.load(frozen["train_schedule"], allow_pickle=False)
    validation_offsets = np.load(frozen["validation_schedule"], allow_pickle=False)

    shape = SmokeShape()
    energy_seed_base = 42_000 + 100 * args.seed
    arms = build_matched_arms(
        shape=shape,
        initialization_seed=args.seed,
        energy_seed_base=energy_seed_base,
    )
    model = arms.pop(args.arm)
    del arms
    gc.collect()

    device = torch.device("cuda")
    model.gradient_checkpointing_enable()
    model.to(device)
    optimizer, optimizer_groups = make_optimizer(model)

    run_start = time.time()
    periodic_metrics = []
    eval_updates = validation_updates(args.max_updates)
    model.train()

    for update_idx in range(args.max_updates):
        optimizer.zero_grad(set_to_none=True)
        update_loss_sum = 0.0
        offsets = train_offsets[update_idx]
        for micro_offsets in chunked(offsets, args.microbatch_sequences):
            inputs, targets = windows_from_offsets(train_tokens, micro_offsets)
            inputs = inputs.to(device=device, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=inputs, use_cache=False).logits
            loss_sum = cross_entropy_sum(logits, targets)
            (loss_sum / GLOBAL_BATCH_TOKENS).backward()
            update_loss_sum += float(loss_sum.detach().item())
            del inputs, targets, logits, loss_sum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        tokens_after = (update_idx + 1) * GLOBAL_BATCH_TOKENS
        learning_rate = lr_at_tokens(tokens_after)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()

        update_number = update_idx + 1
        train_nll = update_loss_sum / GLOBAL_BATCH_TOKENS
        print(
            json.dumps(
                {
                    "event": "train_update",
                    "arm": args.arm,
                    "seed": args.seed,
                    "update": update_number,
                    "tokens_after": tokens_after,
                    "lr": learning_rate,
                    "train_nll": train_nll,
                    "grad_norm_preclip": float(grad_norm),
                    "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        if update_number in eval_updates and not args.skip_periodic_eval:
            normal_nll = evaluate_nll(
                model,
                validation_tokens,
                validation_offsets,
                device=device,
                batch_sequences=args.eval_batch_sequences,
            )
            metric = {
                "update": update_number,
                "tokens_after": tokens_after,
                "validation_nll": normal_nll,
                "validation_perplexity": math.exp(normal_nll),
            }
            periodic_metrics.append(metric)
            print(json.dumps({"event": "validation", **metric}, sort_keys=True), flush=True)
            model.train()

    final_normal_nll = evaluate_nll(
        model,
        validation_tokens,
        validation_offsets,
        device=device,
        batch_sequences=args.eval_batch_sequences,
    )
    final_metrics: dict[str, object] = {
        "normal_nll": final_normal_nll,
        "normal_perplexity": math.exp(final_normal_nll),
    }

    if args.arm == "E_EH1" and not args.skip_final_interventions:
        captured_normal_nll, normal_banks = evaluate_and_capture_energy(
            model,
            validation_tokens,
            validation_offsets,
            device=device,
            batch_sequences=args.eval_batch_sequences,
        )
        if abs(captured_normal_nll - final_normal_nll) > 1e-8:
            raise RuntimeError(
                f"Energy capture changed normal NLL: {captured_normal_nll} vs {final_normal_nll}"
            )
        zero_nll = evaluate_nll(
            model,
            validation_tokens,
            validation_offsets,
            device=device,
            batch_sequences=args.eval_batch_sequences,
            external_provider=provider_zero(normal_banks),
        )
        mean_nll = evaluate_nll(
            model,
            validation_tokens,
            validation_offsets,
            device=device,
            batch_sequences=args.eval_batch_sequences,
            external_provider=provider_mean(normal_banks),
        )
        cross_nlls = []
        within_nlls = []
        for replicate_seed in SHUFFLE_REPLICATE_SEEDS:
            cross_bank = make_cross_sequence_shuffle(normal_banks, replicate_seed)
            cross_nlls.append(
                evaluate_nll(
                    model,
                    validation_tokens,
                    validation_offsets,
                    device=device,
                    batch_sequences=args.eval_batch_sequences,
                    external_provider=provider_bank(cross_bank),
                )
            )
            del cross_bank
            gc.collect()

            within_bank = make_within_sequence_shuffle(normal_banks, replicate_seed)
            within_nlls.append(
                evaluate_nll(
                    model,
                    validation_tokens,
                    validation_offsets,
                    device=device,
                    batch_sequences=args.eval_batch_sequences,
                    external_provider=provider_bank(within_bank),
                )
            )
            del within_bank
            gc.collect()

        final_metrics.update(
            {
                "zero_nll": zero_nll,
                "mean_nll": mean_nll,
                "delta_zero_nll": zero_nll - final_normal_nll,
                "delta_mean_nll": mean_nll - final_normal_nll,
                "shuffle_cross_sequence_nlls": cross_nlls,
                "shuffle_cross_sequence_delta_nlls": [value - final_normal_nll for value in cross_nlls],
                "shuffle_cross_sequence_mean_delta_nll": float(np.mean(cross_nlls) - final_normal_nll),
                "shuffle_cross_sequence_std_delta_nll": float(np.std(np.asarray(cross_nlls) - final_normal_nll)),
                "shuffle_within_sequence_nlls": within_nlls,
                "shuffle_within_sequence_delta_nlls": [value - final_normal_nll for value in within_nlls],
                "shuffle_within_sequence_mean_delta_nll": float(np.mean(within_nlls) - final_normal_nll),
                "shuffle_within_sequence_std_delta_nll": float(np.std(np.asarray(within_nlls) - final_normal_nll)),
                "shuffle_replicate_seeds": list(SHUFFLE_REPLICATE_SEEDS),
                "energy_bank_stats": energy_bank_stats(normal_banks),
                "intervention_semantics": "Normal-run per-layer EH1 scalars captured over the entire frozen validation schedule. Zero and layer-global mean are externally clamped. Cross-sequence and within-sequence shuffles permute the captured scalar assignments with frozen seeds, then clamp those values during intervened forwards; downstream EH1 values are not recomputed from intervened hidden states.",
            }
        )
        del normal_banks
        gc.collect()

    checkpoint_path = args.output_dir / "model.safetensors"
    checkpoint_sha = save_checkpoint(model, checkpoint_path)

    code_commit = resolve_code_commit()
    receipt = {
        "campaign": CAMPAIGN,
        "arm": args.arm,
        "seed": args.seed,
        "verdict_eligible": bool(args.max_updates == TOTAL_UPDATES and not args.skip_periodic_eval and not args.skip_final_interventions),
        "max_updates_executed": args.max_updates,
        "realized_training_tokens_executed": args.max_updates * GLOBAL_BATCH_TOKENS,
        "paired_seed_contract": list(PAIRED_SEEDS),
        "architecture": asdict(shape),
        "projection_dividend": projection_dividend(shape),
        "matched_initialization_seed": args.seed,
        "energy_seed_base": energy_seed_base,
        "transformers_commit": TRANSFORMERS_COMMIT,
        "code_commit": code_commit,
        "precision": "bf16-mixed CUDA autocast; FP32 model parameters and optimizer states",
        "gradient_checkpointing": True,
        "microbatch_sequences": args.microbatch_sequences,
        "gradient_accumulation_microbatches": SEQUENCES_PER_UPDATE // args.microbatch_sequences,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "sequence_length": SEQ,
        "target_tokens_per_sequence": SEQ,
        "loss": "explicit next-token cross entropy: logits(input=window[:-1]) against target=window[1:], reduction=sum/global_batch_tokens",
        "optimizer": {
            "name": "AdamW",
            "betas": list(ADAM_BETAS),
            "eps": ADAM_EPS,
            "weight_decay": WEIGHT_DECAY,
            "foreach": False,
            "fused": False,
            "weight_decay_exempt_parameter_names": optimizer_groups["no_decay"],
        },
        "scheduler": {
            "clock": "cumulative processed training tokens",
            "boundary_convention": "learning rate for an optimizer step is evaluated at tokens_after_update, after the batch gradient is accumulated and immediately before optimizer.step",
            "max_lr": MAX_LR,
            "min_lr": MIN_LR,
            "warmup_fraction": WARMUP_FRACTION,
            "warmup_tokens_continuous": WARMUP_TOKENS,
            "total_tokens": REALIZED_TRAINING_TOKENS,
            "schedule": "linear warmup then cosine to min_lr",
        },
        "gradient_clip_norm": GRAD_CLIP,
        "periodic_validation": {
            "token_interval": VALIDATION_TOKEN_INTERVAL,
            "checkpoint_update_rule": "ceil(k*1,000,000/global_batch_tokens), plus final update; duplicate update indices deduplicated",
            "validation_sequences": int(len(validation_offsets)),
            "validation_target_tokens": int(len(validation_offsets) * SEQ),
            "metrics": periodic_metrics,
        },
        "final_metrics": final_metrics,
        "frozen_input_closeout": frozen["closeout"],
        "frozen_input_hashes_reverified": frozen["actual_hashes"],
        "source_amendment": frozen["amendment"],
        "checkpoint": {"path": checkpoint_path.name, "sha256": checkpoint_sha},
        "runtime": {
            "seconds": time.time() - run_start,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "claim_ceiling": "Architecture Smoke only. No production parity, physical energy/MAC, or Microsoft 2B4T corpus claim.",
    }
    receipt_path = args.output_dir / "RUN_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    sha_ledger = args.output_dir / "SHA256SUMS.txt"
    sha_ledger.write_text(
        f"{sha256_file(receipt_path)}  {receipt_path.name}\n"
        f"{checkpoint_sha}  {checkpoint_path.name}\n"
    )
    print(json.dumps({"event": "run_complete", "receipt": receipt}, sort_keys=True), flush=True)

    if args.upload_repo:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for --upload-repo.")
        api = HfApi(token=token)
        path_in_repo = f"{args.upload_prefix}/{args.arm}/seed_{args.seed}"
        if not receipt["verdict_eligible"]:
            path_in_repo = f"preflight/{args.arm}/seed_{args.seed}/updates_{args.max_updates}"
        api.upload_folder(
            repo_id=args.upload_repo,
            repo_type="dataset",
            folder_path=str(args.output_dir),
            path_in_repo=path_in_repo,
            commit_message=f"Upload {CAMPAIGN} {args.arm} seed {args.seed} updates {args.max_updates}",
        )
        print("UPLOAD_PASS", args.upload_repo, path_in_repo, flush=True)


if __name__ == "__main__":
    main()
