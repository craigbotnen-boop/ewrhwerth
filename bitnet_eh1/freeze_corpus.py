#!/usr/bin/env python3
"""Freeze the pre-training corpus/tokenizer/schedules for BITNET_EH1_ARCHITECTURE_SMOKE_V1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import datasets
import huggingface_hub
import numpy as np
import tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

ROOT = Path("bitnet_eh1/frozen_corpus")
SRC = ROOT / "source_jsonl"
TOK = ROOT / "tokenized"
SCH = ROOT / "schedules"
for path in (SRC, TOK, SCH):
    path.mkdir(parents=True, exist_ok=True)

DATASET_ID = "gmongaras/SlimPajama-627B_Reupload"
DATASET_REVISION = "c34c22dbb10ae6b264a2f357a909d1a537141b36"
UPSTREAM_DATASET_ID = "cerebras/SlimPajama-627B"
FIRST32_CANON_SHA256 = {
    "train": "18291e33adc67674291390b4b445f3c0d3da8e6add91f8c9c52fbabee975377b",
    "validation": "e56046fffef07ab782c1f3144176113c6b97d5c1d196f6dec67241e4ef75e1d1",
}
EXPECTED_FIRST = {
    "train": (
        "RedPajamaCommonCrawl",
        "J.J. Abrams Returns To Write And Direct 'Star Wars: Episode IX'",
    ),
    "validation": ("RedPajamaGithub", "namespace base {"),
}
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
SID = {token: idx for idx, token in enumerate(SPECIAL)}
SEQ = 2048
BATCH_TOK = 32768
UPDATES = 1526
SEEDS = [17, 29, 41]
VAL_SEQ = 2048


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sampled_order(split: str) -> dict[str, object]:
    stream = load_dataset(
        DATASET_ID,
        split=split,
        streaming=True,
        revision=DATASET_REVISION,
    )
    digest = hashlib.sha256()
    first = None
    rows = 0
    for idx, row in enumerate(stream):
        canonical = json.dumps(
            {"text": row.get("text"), "meta": row.get("meta")},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
        if idx == 0:
            first = row
        rows += 1
        if idx == 31:
            break
    if rows != 32 or first is None:
        raise RuntimeError(f"Expected 32 sampled rows for {split}, got {rows}.")
    meta = (first.get("meta") or {}).get("redpajama_set_name")
    text = first.get("text", "")
    expected_meta, expected_prefix = EXPECTED_FIRST[split]
    if meta != expected_meta or not text.startswith(expected_prefix):
        raise RuntimeError(f"Sampled source-order compatibility failed for {split}.")
    sampled_sha = digest.hexdigest()
    if sampled_sha != FIRST32_CANON_SHA256[split]:
        raise RuntimeError(f"First-32 canonical SHA mismatch for {split}.")
    return {
        "first32_canonical_sha256": sampled_sha,
        "first_row_meta_set_name": meta,
        "first_row_text_prefix": expected_prefix,
    }


def materialize(split: str, target_mib: int, output: Path) -> dict[str, object]:
    stream = load_dataset(
        DATASET_ID,
        split=split,
        streaming=True,
        revision=DATASET_REVISION,
    )
    target = target_mib * 1024 * 1024
    selected = source_rows_seen = text_bytes = 0
    first_selected = last_selected = None
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for idx, row in enumerate(stream):
            source_rows_seen = idx + 1
            text = row.get("text")
            if not isinstance(text, str) or len(text) == 0:
                continue
            if first_selected is None:
                first_selected = idx
            last_selected = idx
            handle.write(
                json.dumps(
                    {"text": text, "meta": row.get("meta")},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
            selected += 1
            text_bytes += len(text.encode("utf-8"))
            if text_bytes >= target:
                break
    if text_bytes < target:
        raise RuntimeError(f"Insufficient {split} text: {text_bytes} < {target}.")
    return {
        "split": split,
        "target_text_bytes": target,
        "realized_text_bytes": text_bytes,
        "selected_document_count": selected,
        "source_rows_seen": source_rows_seen,
        "first_selected_source_row": first_selected,
        "last_selected_source_row": last_selected,
        "output_path": str(output),
        "output_sha256": sha256_file(output),
        "output_size_bytes": output.stat().st_size,
        "selection_rule": "native streaming order; nonempty string text only; no normalization/strip/truncate/shuffle/resample; stop when cumulative UTF-8 text bytes first reach/exceed target",
    }


def iter_texts(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                text = json.loads(line)["text"]
                if not isinstance(text, str):
                    raise TypeError("Frozen text must be a string.")
                yield text


def encode(tokenizer: Tokenizer, source: Path, output: Path) -> tuple[int, int]:
    token_count = document_count = 0
    with output.open("wb") as writer:
        for text in iter_texts(source):
            ids = [SID["<bos>"], *tokenizer.encode(text).ids, SID["<eos>"]]
            if max(ids) >= 65536:
                raise RuntimeError("Token ID does not fit uint16.")
            np.asarray(ids, dtype=np.uint16).tofile(writer)
            token_count += len(ids)
            document_count += 1
    return token_count, document_count


def main() -> None:
    sampled_compatibility = {
        split: verify_sampled_order(split) for split in ("train", "validation")
    }
    amendment = {
        "campaign": "BITNET_EH1_ARCHITECTURE_SMOKE_SOURCE_AMENDMENT_V1",
        "timing": "PRE_TRAINING",
        "original_requested_dataset_id": UPSTREAM_DATASET_ID,
        "substituted_dataset_id": DATASET_ID,
        "substituted_dataset_revision": DATASET_REVISION,
        "reason": "Original Cerebras Hub dataset API/resolve endpoint was not accessible during 2026-07-15 preflight; GitHub Actions failed closed before corpus materialization.",
        "mirror_declaration": "Mirror README declares a reupload of the original SlimPajama-627B corpus into larger Parquet chunks.",
        "sampled_order_compatibility": sampled_compatibility,
        "identity_status": "PINNED_PUBLIC_REUPLOAD_WITH_SAMPLED_ORDER_COMPATIBILITY; NOT_BYTEWISE_UPSTREAM_CERTIFIED",
        "scientific_effect": "Architecture Smoke source freeze amended before any training verdict. Claim ceiling remains Architecture Smoke only.",
    }
    amendment_path = SRC / "SOURCE_SUBSTITUTION_AMENDMENT.json"
    amendment_path.write_text(json.dumps(amendment, indent=2, sort_keys=True))

    train_jsonl = SRC / "train.jsonl"
    validation_jsonl = SRC / "validation.jsonl"
    source_manifest = {
        "campaign": "BITNET_EH1_ARCHITECTURE_SMOKE_SLIMPAJAMA_SOURCE_FREEZE_V2",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "upstream_declared_original": UPSTREAM_DATASET_ID,
        "source_identity_status": amendment["identity_status"],
        "source_amendment_path": str(amendment_path),
        "source_amendment_sha256": sha256_file(amendment_path),
        "datasets_version": datasets.__version__,
        "huggingface_hub_version": huggingface_hub.__version__,
        "train": materialize("train", 256, train_jsonl),
        "validation": materialize("validation", 32, validation_jsonl),
        "claim_ceiling": "Deterministic ordered pinned-mirror SlimPajama subsample for Architecture Smoke; source is not bytewise upstream-certified and is not the Microsoft 2B4T production corpus.",
    }
    (SRC / "SOURCE_JSONL_FREEZE_MANIFEST.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True)
    )

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=4096,
        min_frequency=2,
        special_tokens=SPECIAL,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_texts(train_jsonl), trainer=trainer)
    vocabulary = tokenizer.get_vocab()
    if tokenizer.get_vocab_size() != 4096 or not all(
        vocabulary[token] == idx for token, idx in SID.items()
    ):
        raise RuntimeError("Tokenizer freeze contract failed.")
    tokenizer_path = TOK / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    train_bin = TOK / "train.tokens.uint16"
    validation_bin = TOK / "validation.tokens.uint16"
    train_tokens, train_docs = encode(tokenizer, train_jsonl, train_bin)
    validation_tokens, validation_docs = encode(
        tokenizer, validation_jsonl, validation_bin
    )
    corpus_manifest = {
        "campaign": "BITNET_EH1_ARCHITECTURE_SMOKE_CORPUS_FREEZE_V1",
        "text_field": "text",
        "tokenizers_version": tokenizers.__version__,
        "tokenizer_type": "ByteLevel BPE",
        "vocab_size": 4096,
        "special_token_ids": SID,
        "document_encoding": "[BOS] + encode(text) + [EOS]",
        "train": {
            "source_path": str(train_jsonl),
            "source_sha256": sha256_file(train_jsonl),
            "document_count": train_docs,
            "token_bin": str(train_bin),
            "token_bin_sha256": sha256_file(train_bin),
            "token_count": train_tokens,
        },
        "validation": {
            "source_path": str(validation_jsonl),
            "source_sha256": sha256_file(validation_jsonl),
            "document_count": validation_docs,
            "token_bin": str(validation_bin),
            "token_bin_sha256": sha256_file(validation_bin),
            "token_count": validation_tokens,
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": sha256_file(tokenizer_path),
            "trained_on": "train split only",
        },
        "claim_ceiling": "4096-vocab Architecture Smoke tokenizer; not Microsoft 2B4T Llama-3 tokenizer.",
    }
    (TOK / "CORPUS_FREEZE_MANIFEST.json").write_text(
        json.dumps(corpus_manifest, indent=2, sort_keys=True)
    )

    sequences_per_batch = BATCH_TOK // SEQ
    window = SEQ + 1
    total_sequences = UPDATES * sequences_per_batch
    train_windows = train_tokens // window
    validation_windows = validation_tokens // window
    if BATCH_TOK % SEQ != 0 or train_windows < total_sequences or validation_windows < VAL_SEQ:
        raise RuntimeError("Frozen token streams do not satisfy the schedule contract.")

    schedules = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        ids = rng.permutation(train_windows)[:total_sequences]
        if np.unique(ids).size != total_sequences:
            raise RuntimeError("Train source-window reuse detected.")
        offsets = (ids.astype(np.int64) * window).reshape(
            UPDATES, sequences_per_batch
        )
        path = SCH / f"train_schedule_seed_{seed}.npy"
        np.save(path, offsets, allow_pickle=False)
        schedules.append(
            {
                "seed": seed,
                "path": str(path),
                "sha256": sha256_file(path),
                "shape": list(offsets.shape),
                "sequences_per_update": sequences_per_batch,
                "input_tokens_per_update": BATCH_TOK,
                "source_window_tokens": window,
                "source_window_reuse_occurred": False,
                "unique_source_windows": int(np.unique(ids).size),
            }
        )

    validation_offsets = np.arange(VAL_SEQ, dtype=np.int64) * window
    validation_schedule_path = SCH / "validation_schedule.npy"
    np.save(validation_schedule_path, validation_offsets, allow_pickle=False)
    schedule_manifest = {
        "campaign": "BITNET_EH1_ARCHITECTURE_SMOKE_BATCH_SCHEDULE_V2",
        "sequence_length": SEQ,
        "source_window_tokens": window,
        "global_batch_tokens": BATCH_TOK,
        "optimizer_updates": UPDATES,
        "realized_training_tokens": UPDATES * BATCH_TOK,
        "train_token_count": train_tokens,
        "train_nonoverlapping_source_window_count": train_windows,
        "allow_train_window_reuse": False,
        "train_schedules": schedules,
        "validation": {
            "token_count": validation_tokens,
            "nonoverlapping_source_window_count": validation_windows,
            "selected_sequence_count": VAL_SEQ,
            "selection_rule": "first N non-overlapping source windows in frozen validation token order",
            "path": str(validation_schedule_path),
            "sha256": sha256_file(validation_schedule_path),
            "shape": list(validation_offsets.shape),
        },
        "pairing_rule": "Within seed all four arms consume identical train offsets; all arms/interventions use identical validation offsets.",
    }
    (SCH / "BATCH_SCHEDULE_MANIFEST.json").write_text(
        json.dumps(schedule_manifest, indent=2, sort_keys=True)
    )

    closeout = {
        "campaign": "BITNET_EH1_ARCHITECTURE_SMOKE_FREEZE_CLOSEOUT_V2",
        "source_dataset_id": DATASET_ID,
        "source_dataset_revision": DATASET_REVISION,
        "source_identity_status": amendment["identity_status"],
        "source_amendment_sha256": source_manifest["source_amendment_sha256"],
        "source_train_jsonl_sha256": source_manifest["train"]["output_sha256"],
        "source_validation_jsonl_sha256": source_manifest["validation"]["output_sha256"],
        "tokenizer_sha256": corpus_manifest["tokenizer"]["sha256"],
        "train_token_bin_sha256": corpus_manifest["train"]["token_bin_sha256"],
        "validation_token_bin_sha256": corpus_manifest["validation"]["token_bin_sha256"],
        "train_token_count": train_tokens,
        "validation_token_count": validation_tokens,
        "realized_training_tokens": UPDATES * BATCH_TOK,
        "paired_seeds": SEEDS,
        "validation_sequence_count": VAL_SEQ,
        "train_window_reuse": False,
        "verdict": "CORPUS_TOKENIZER_AND_SCHEDULE_FREEZE_PASS",
        "claim_ceiling": "Architecture Smoke data freeze only; pinned public reupload is not bytewise upstream-certified and the corpus is not the Microsoft 2B4T production corpus.",
    }
    (ROOT / "FREEZE_CLOSEOUT_RECEIPT.json").write_text(
        json.dumps(closeout, indent=2, sort_keys=True)
    )
    print(json.dumps(closeout, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
