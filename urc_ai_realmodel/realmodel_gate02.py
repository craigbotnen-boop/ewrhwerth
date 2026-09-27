#!/usr/bin/env python3
"""
URC-AI Gate 02 real-model experiment (frozen confirmatory run).

Models:
  - Qwen/Qwen2.5-0.5B-Instruct (GQA)
  - EleutherAI/pythia-160m (GPT-NeoX family)

The script does NOT fit or tune any target spectral dimension.
It implements the frozen v0.5 reject-option protocol:
  raw + top-p {0.85,0.90,0.95}, native + token-0 puncture,
  stationary-mode-subtracted heat trace,
  >=1.5 decades, R^2>=0.995, local d_S range<=0.15,
  IR cap tau <= 0.25/lambda_1.

It writes only compact CSV/JSON summaries; raw N x N attentions are not
uploaded as artifacts.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.set_num_interop_threads(1)

MODELS = {
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
    "pythia": "EleutherAI/pythia-160m",
}

TASKS = {
    "arithmetic": (
        "Solve carefully. A warehouse starts with 4,120 pallets. "
        "On Monday 14 shipments of 65 pallets arrive and 380 pallets are dispatched. "
        "On Tuesday 22 shipments of 45 pallets arrive and 515 pallets are dispatched. "
        "On Wednesday one-fourth of the current stock is transferred out. "
        "How many pallets remain? Show the arithmetic logic explicitly."
    ),
    "retrieval": (
        "Use this reference log. [001 Alpha=Operational] [002 Beta=Degraded] "
        "[003 Gamma=Offline] [004 Delta=Operational] [005 Epsilon=Standby] "
        "[006 Zeta=Maintenance] [007 Eta=Operational] [008 Theta=Offline]. "
        "Identify which systems are either Degraded or Maintenance and explain how the "
        "answer follows from the log."
    ),
}

FILLER = (
    "Archived neutral context. This sentence is deliberately irrelevant to the task, "
    "contains no answer-bearing quantities or system labels, and exists only to control "
    "context length. "
)


def exact_length_ids(tokenizer, core: str, target_n: int) -> torch.Tensor:
    core_ids = tokenizer.encode(core, add_special_tokens=False)
    filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
    if len(core_ids) >= target_n:
        ids = core_ids[-target_n:]
    else:
        need = target_n - len(core_ids)
        reps = (need + len(filler_ids) - 1) // len(filler_ids)
        prefix = (filler_ids * reps)[:need]
        ids = prefix + core_ids
    assert len(ids) == target_n
    return torch.tensor([ids], dtype=torch.long)


def row_top_p(A: np.ndarray, p: float) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    B = np.zeros_like(A)
    for i in range(n):
        x = A[i, : i + 1]
        order = np.argsort(-x)
        cs = np.cumsum(x[order])
        k = int(np.searchsorted(cs, p, side="left")) + 1
        keep = order[: max(1, k)]
        B[i, keep] = x[keep]
        s = B[i].sum()
        if s > 0:
            B[i] /= s
    return B


def above_uniform(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    B = np.zeros_like(A)
    for i in range(n):
        x = A[i, : i + 1]
        th = 1.0 / (i + 1)
        keep = np.flatnonzero(x > th)
        if len(keep) == 0:
            keep = np.array([int(np.argmax(x))])
        B[i, keep] = x[keep]
        s = B[i].sum()
        if s > 0:
            B[i] /= s
    return B


def puncture(A: np.ndarray, idx: int) -> np.ndarray:
    keep = np.ones(len(A), dtype=bool)
    keep[idx] = False
    return A[np.ix_(keep, keep)]


def weak_mass(A: np.ndarray) -> dict:
    n = len(A)
    weak = []
    long_weak = []
    token0 = []
    for i in range(n):
        row = A[i, : i + 1]
        th = 1.0 / (i + 1)
        weak.append(float(row[row <= th].sum()))
        if i > 0:
            token0.append(float(row[0]))
        if i >= 4:
            cut = max(1, (i + 1) // 2)
            old = row[:cut]
            long_weak.append(float(old[old <= th].sum()))
    return {
        "weak_mass_mean": float(np.mean(weak)),
        "weak_mass_p95": float(np.quantile(weak, 0.95)),
        "long_weak_mass_mean": float(np.mean(long_weak)) if long_weak else 0.0,
        "token0_mean_attention": float(np.mean(token0)) if token0 else np.nan,
        "token0_median_attention": float(np.median(token0)) if token0 else np.nan,
    }


def graph_stats(A: np.ndarray, token0_present: bool = True):
    W = (A + A.T) / 2.0
    deg = W.sum(axis=1)
    keep = deg > 1e-14
    W = W[np.ix_(keep, keep)]
    deg = W.sum(axis=1)
    if len(W) < 3:
        return None

    mean_deg = float(deg.mean())
    top = int(np.argmax(deg))
    pi = deg / deg.sum()
    inv = 1.0 / np.sqrt(np.maximum(deg, 1e-300))
    S = (inv[:, None] * W) * inv[None, :]
    L = np.eye(len(S)) - S
    L = (L + L.T) / 2.0
    eig = scipy.linalg.eigvalsh(L, overwrite_a=True, check_finite=False)
    eig[np.abs(eig) < 1e-11] = 0.0
    eig = np.maximum(eig, 0.0)
    pos = eig[eig > 1e-11]
    lam1 = float(pos[0]) if len(pos) else np.nan
    trel = 1.0 / lam1 if np.isfinite(lam1) and lam1 > 0 else np.inf

    return {
        "eig": eig,
        "lambda1": lam1,
        "t_relax": trel,
        "hub_index": float(deg[top] / mean_deg) if mean_deg > 0 else np.nan,
        "max_degree_node": top,
        "stationary_pi_max": float(pi[top]),
        "stationary_ipr": float(np.sum(pi * pi)),
        "stationary_effective_nodes": float(1.0 / np.sum(pi * pi)),
        "token0_hub_index": float(deg[0] / mean_deg)
        if token0_present and mean_deg > 0
        else np.nan,
        "token0_stationary_pi": float(pi[0]) if token0_present else np.nan,
    }


def linfit(x, y):
    b, a = np.polyfit(x, y, 1)
    yh = a + b * x
    den = np.sum((y - y.mean()) ** 2)
    r2 = np.nan if den <= 0 else 1.0 - np.sum((y - yh) ** 2) / den
    return float(-2.0 * b), float(r2)


def plateau(eig, min_decades=1.5, min_r2=0.995, max_local_range=0.15):
    pos = eig[eig > 1e-11]
    if len(pos) == 0:
        return {
            "status": "NO_STABLE_PLATEAU",
            "d_s": None,
            "r2": None,
            "t1": None,
            "t2": None,
            "local_ds_range": None,
        }

    lam1 = float(pos[0])
    ir_cap = 0.25 / lam1
    tau = np.geomspace(5.0, 1e4, 220)
    c0 = int(np.sum(eig <= 1e-11))
    # stationary-subtracted heat trace
    p = np.exp(-np.outer(tau, eig)).mean(axis=1) - c0 / len(eig)
    floor = max(1e-15, float(np.max(p)) * 1e-12)
    m = (p > floor) & (tau <= ir_cap)
    t = tau[m]
    p = p[m]

    if len(t) < 12 or t[-1] / t[0] < 10**min_decades:
        return {
            "status": "INSUFFICIENT_DYNAMIC_RANGE",
            "d_s": None,
            "r2": None,
            "t1": None,
            "t2": None,
            "local_ds_range": None,
        }

    lx = np.log10(t)
    ly = np.log10(p)
    best = None
    for i in range(len(t)):
        for j in range(i + 8, len(t)):
            if lx[j] - lx[i] < min_decades:
                continue
            ds, r2 = linfit(lx[i : j + 1], ly[i : j + 1])
            if not np.isfinite(r2) or r2 < min_r2 or ds <= 0:
                continue

            local = []
            for k in range(i, j - 7):
                dloc, rloc = linfit(lx[k : k + 9], ly[k : k + 9])
                if np.isfinite(rloc) and rloc >= 0.98:
                    local.append(dloc)
            if len(local) < 3:
                continue
            spread = float(max(local) - min(local))
            if spread > max_local_range:
                continue

            score = (float(lx[j] - lx[i]), r2, -spread)
            rec = {
                "status": "PLATEAU_ACCEPTED",
                "d_s": float(ds),
                "r2": float(r2),
                "t1": float(t[i]),
                "t2": float(t[j]),
                "local_ds_range": spread,
            }
            if best is None or score > best[0]:
                best = (score, rec)

    return (
        best[1]
        if best is not None
        else {
            "status": "NO_STABLE_PLATEAU",
            "d_s": None,
            "r2": None,
            "t1": None,
            "t2": None,
            "local_ds_range": None,
        }
    )


def analyze_variant(model_key, task, N, layer, filt, variant, A, weak):
    token0_present = variant == "NATIVE"
    gs = graph_stats(A, token0_present=token0_present)
    if gs is None:
        return None
    pl = plateau(gs.pop("eig"))
    out = {
        "model": model_key,
        "task": task,
        "N": int(N),
        "layer": int(layer),
        "filter": filt,
        "variant": variant,
        **weak,
        **gs,
        **pl,
    }
    out["cheeger_lower"] = (
        float(out["lambda1"] / 2.0) if np.isfinite(out["lambda1"]) else np.nan
    )
    out["cheeger_upper"] = (
        float(np.sqrt(2.0 * out["lambda1"]))
        if np.isfinite(out["lambda1"])
        else np.nan
    )
    return out


def model_load(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    kwargs = dict(
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="eager", **kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        try:
            model.config._attn_implementation = "eager"
        except Exception:
            pass
    model.eval()
    return tok, model


def mean_effective_resistance(A: np.ndarray) -> float:
    """Mean pairwise effective resistance of connected weighted graph."""
    W = (A + A.T) / 2.0
    deg = W.sum(1)
    keep = deg > 1e-14
    W = W[np.ix_(keep, keep)]
    deg = W.sum(1)
    if len(W) < 3:
        return np.nan
    Lc = np.diag(deg) - W
    mu = scipy.linalg.eigvalsh(Lc, overwrite_a=True, check_finite=False)
    pos = mu[mu > 1e-10]
    if len(pos) != len(W) - 1:
        return np.nan
    return float(2.0 / (len(W) - 1) * np.sum(1.0 / pos))


def classify(df: pd.DataFrame, model_key: str):
    d = df[df.model == model_key].copy()
    if d.empty:
        return {"model": model_key, "branch": "NO_DATA"}

    # Per task/layer/N threshold stability.
    stable_records = []
    for variant in ["NATIVE", "PUNCTURE_TOKEN0"]:
        dv = d[(d.variant == variant) & d["filter"].isin(["TOPP_085", "TOPP_090", "TOPP_095"])]
        for (task, layer, N), g in dv.groupby(["task", "layer", "N"]):
            ok = len(g) == 3 and (g.status == "PLATEAU_ACCEPTED").all()
            span = float(g.d_s.max() - g.d_s.min()) if ok else np.nan
            stable_records.append(
                {
                    "task": task,
                    "layer": int(layer),
                    "N": int(N),
                    "variant": variant,
                    "threshold_stable": bool(ok and span <= 0.15),
                    "ds_span": span,
                    "ds_mean": float(g.d_s.mean()) if ok else np.nan,
                    "t_relax_mean": float(g.t_relax.mean()),
                }
            )
    st = pd.DataFrame(stable_records)

    # Scale persistence across 256/512/1024 for the same task/layer.
    scale_candidates = []
    for variant in ["NATIVE", "PUNCTURE_TOKEN0"]:
        sv = st[st.variant == variant]
        for (task, layer), g in sv.groupby(["task", "layer"]):
            g = g.sort_values("N")
            if set(g.N.tolist()) != {256, 512, 1024}:
                continue
            if not g.threshold_stable.all():
                continue
            ds_spread = float(g.ds_mean.max() - g.ds_mean.min())
            trel = g.t_relax_mean.to_numpy()
            monotone = bool(trel[0] < trel[1] < trel[2])
            if ds_spread <= 0.15 and monotone:
                scale_candidates.append(
                    {
                        "variant": variant,
                        "task": task,
                        "layer": int(layer),
                        "ds_mean": float(g.ds_mean.mean()),
                        "ds_crossN_span": ds_spread,
                        "t_relax_256": float(trel[0]),
                        "t_relax_512": float(trel[1]),
                        "t_relax_1024": float(trel[2]),
                    }
                )

    native = [x for x in scale_candidates if x["variant"] == "NATIVE"]
    punct = [x for x in scale_candidates if x["variant"] == "PUNCTURE_TOKEN0"]

    any_plateau = bool((d.status == "PLATEAU_ACCEPTED").any())
    raw_plateaux = int(
        ((d["filter"] == "RAW") & (d.variant == "NATIVE") & (d.status == "PLATEAU_ACCEPTED")).sum()
    )

    if native:
        branch = "A_GEOMETRIC_BACKBONE_CANDIDATE"
    elif punct:
        branch = "H_HUB_SINK_CONTAMINATED"
    elif any_plateau:
        branch = "B_THRESHOLD_OR_LOCAL_ARTIFACT"
    else:
        branch = "C_A_ATTENTION_WEIGHT_NONMANIFOLD"

    # Sink scaling summary from RAW/NATIVE.
    raw = d[(d["filter"] == "RAW") & (d.variant == "NATIVE")]
    sink_by_N = {}
    for N, g in raw.groupby("N"):
        sink_by_N[str(int(N))] = {
            "hub_index_mean": float(g.hub_index.mean()),
            "token0_hub_index_mean": float(g.token0_hub_index.mean()),
            "token0_stationary_pi_mean": float(g.token0_stationary_pi.mean()),
            "t_relax_mean": float(g.t_relax.mean()),
            "lambda1_mean": float(g.lambda1.mean()),
        }

    return {
        "model": model_key,
        "branch": branch,
        "native_scale_candidates": native,
        "punctured_scale_candidates": punct,
        "raw_plateaux": raw_plateaux,
        "any_plateau": any_plateau,
        "sink_scaling": sink_by_N,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", choices=MODELS, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    model_id = MODELS[args.model_key]

    run_meta = {
        "model_key": args.model_key,
        "model_id": model_id,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "started_unix": time.time(),
        "protocol": {
            "lengths": [256, 512, 1024],
            "top_p": [0.85, 0.90, 0.95],
            "min_decades": 1.5,
            "r2": 0.995,
            "local_ds_range": 0.15,
            "ir_fraction": 0.25,
        },
    }

    print(f"Loading {model_id}", flush=True)
    tok, model = model_load(model_id)
    run_meta["config"] = {
        "hidden_size": getattr(model.config, "hidden_size", None),
        "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
        "num_attention_heads": getattr(model.config, "num_attention_heads", None),
        "num_key_value_heads": getattr(model.config, "num_key_value_heads", None),
        "max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
    }
    print(json.dumps(run_meta["config"], indent=2), flush=True)

    rows = []
    resistance_rows = []
    for task, core in TASKS.items():
        for N in [256, 512, 1024]:
            ids = exact_length_ids(tok, core, N)
            mask = torch.ones_like(ids)
            t0 = time.time()
            print(f"FORWARD model={args.model_key} task={task} N={N}", flush=True)
            with torch.inference_mode():
                outputs = model(
                    input_ids=ids,
                    attention_mask=mask,
                    output_attentions=True,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
            elapsed = time.time() - t0
            if outputs.attentions is None:
                raise RuntimeError("Model did not return attentions.")
            print(f"forward_seconds={elapsed:.2f} layers={len(outputs.attentions)}", flush=True)

            for l, att in enumerate(outputs.attentions):
                # mean over query heads; GQA query heads are the returned attention maps.
                A = att[0].detach().float().cpu().numpy().mean(axis=0).astype(np.float64)
                weak = weak_mass(A)
                filt_map = {
                    "RAW": A,
                    "TOPP_085": row_top_p(A, 0.85),
                    "TOPP_090": row_top_p(A, 0.90),
                    "TOPP_095": row_top_p(A, 0.95),
                }
                if N == 512:
                    filt_map["ABOVE_UNIFORM"] = above_uniform(A)

                for filt_name, B in filt_map.items():
                    rec = analyze_variant(args.model_key, task, N, l, filt_name, "NATIVE", B, weak)
                    if rec is not None:
                        rows.append(rec)
                    if len(B) > 3:
                        Bp = puncture(B, 0)
                        recp = analyze_variant(
                            args.model_key,
                            task,
                            N,
                            l,
                            filt_name,
                            "PUNCTURE_TOKEN0",
                            Bp,
                            weak,
                        )
                        if recp is not None:
                            rows.append(recp)

                # Expander-pivot resistance diagnostic only at N=512,
                # representative early/mid/late layers, TOPP_090.
                if N == 512 and l in {0, len(outputs.attentions)//2, len(outputs.attentions)-1}:
                    B90 = filt_map["TOPP_090"]
                    resistance_rows.append(
                        {
                            "model": args.model_key,
                            "task": task,
                            "N": N,
                            "layer": l,
                            "filter": "TOPP_090",
                            "variant": "NATIVE",
                            "mean_effective_resistance": mean_effective_resistance(B90),
                        }
                    )

            del outputs
            gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "layer_filter_results.csv", index=False)
    rdf = pd.DataFrame(resistance_rows)
    rdf.to_csv(outdir / "resistance_pivot.csv", index=False)

    summary = classify(df, args.model_key)
    summary["run_meta"] = run_meta
    summary["finished_unix"] = time.time()
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Compact layer-level stability table.
    stable = []
    for variant in ["NATIVE", "PUNCTURE_TOKEN0"]:
        dv = df[(df.variant == variant) & df["filter"].isin(["TOPP_085","TOPP_090","TOPP_095"])]
        for (task, N, layer), g in dv.groupby(["task","N","layer"]):
            ok = len(g)==3 and (g.status=="PLATEAU_ACCEPTED").all()
            span = float(g.d_s.max()-g.d_s.min()) if ok else np.nan
            stable.append({
                "model":args.model_key,"task":task,"N":int(N),"layer":int(layer),
                "variant":variant,"all_top_p_plateau":bool(ok),
                "ds_span":span,"threshold_stable":bool(ok and span<=0.15),
                "ds_mean":float(g.d_s.mean()) if ok else np.nan,
                "t_relax_mean":float(g.t_relax.mean()),
            })
    pd.DataFrame(stable).to_csv(outdir / "threshold_stability.csv", index=False)

    print("FINAL_SUMMARY", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
