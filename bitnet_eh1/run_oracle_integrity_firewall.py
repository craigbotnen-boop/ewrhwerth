#!/usr/bin/env python3
"""Fail-closed F0-F4 oracle-integrity firewall for pinned BitNet online QAT."""
from __future__ import annotations

import argparse, hashlib, json, os, platform, subprocess, traceback
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from transformers import BitNetConfig
from transformers.integrations.bitnet import ActQuant, AutoBitLinear, WeightQuant
from transformers.models.bitnet.modeling_bitnet import BitNetMLP
from bitnet_energy_head_smoke_core import convert_to_online_bitlinear

ID = "BITNET_EH1_ORACLE_INTEGRITY_FIREWALL_F0_F4_V1"
TX_REV = "096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4"


def commit() -> str | None:
    if os.environ.get("BITNET_EH1_CODE_COMMIT"):
        return os.environ["BITNET_EH1_CODE_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    d, r = a.detach().double() - b.detach().double(), b.detach().double()
    ma = float(d.abs().max()) if d.numel() else 0.0
    rm = float(d.square().mean().sqrt()) if d.numel() else 0.0
    rma = float(r.abs().max()) if r.numel() else 0.0
    rrm = float(r.square().mean().sqrt()) if r.numel() else 0.0
    return {"max_abs": ma, "rms": rm, "max_rel": ma / max(rma, 1e-30), "rms_rel": rm / max(rrm, 1e-30), "reference_max": rma, "reference_rms": rrm}


def check(name: str, a: torch.Tensor, b: torch.Tensor, max_rel: float, rms_rel: float) -> dict[str, Any]:
    m = metrics(a, b)
    row = {"test": name, "limits": {"max_rel": max_rel, "rms_rel": rms_rel}, "metrics": m}
    if m["max_rel"] > max_rel or m["rms_rel"] > rms_rel:
        raise AssertionError(json.dumps(row, sort_keys=True))
    row["passed"] = True
    return row


def profile(dtype: torch.dtype) -> dict[str, float]:
    if dtype == torch.float32:
        return {"f2m": 2e-7, "f2r": 2e-7, "f3m": 2e-6, "f3r": 2e-6, "spm": 2e-5, "spr": 2e-6, "f1m": 2e-12, "f1r": 2e-13}
    return {"f2m": 5e-3, "f2r": 2e-3, "f3m": 1e-2, "f3r": 5e-3, "spm": 5e-2, "spr": 1.5e-2, "f1m": 2e-12, "f1r": 2e-13}


def make_mlp(h: int, d: int, eps: float, dtype: torch.dtype, device: torch.device, seed: int) -> BitNetMLP:
    cfg = BitNetConfig(vocab_size=4096, hidden_size=h, intermediate_size=d, num_hidden_layers=12, num_attention_heads=8, num_key_value_heads=2, hidden_act="relu2", max_position_embeddings=2048, rms_norm_eps=eps, use_cache=False, attention_bias=False, attention_dropout=0.0)
    torch.manual_seed(seed)
    m = BitNetMLP(cfg)
    with torch.no_grad():
        g = torch.Generator(device="cpu").manual_seed(seed)
        for layer in (m.gate_proj, m.up_proj, m.down_proj):
            torch.nn.init.xavier_normal_(layer.weight, generator=g)
        m.ffn_sub_norm.weight.uniform_(0.75, 1.25, generator=g)
    convert_to_online_bitlinear(m)
    if not all(isinstance(v, AutoBitLinear) for v in (m.gate_proj, m.up_proj, m.down_proj)):
        raise AssertionError("AutoBitLinear conversion failed")
    return m.to(device=device, dtype=dtype).eval()


def capture(m: BitNetMLP, x: torch.Tensor) -> dict[str, torch.Tensor]:
    c: dict[str, torch.Tensor] = {}
    hs = []
    def out(name):
        def hook(_m, _i, o): c[name] = o.detach().clone()
        return hook
    def inp(name):
        def hook(_m, i): c[name] = i[0].detach().clone()
        return hook
    hs += [m.gate_proj.register_forward_hook(out("gate")), m.up_proj.register_forward_hook(out("up")), m.ffn_sub_norm.register_forward_pre_hook(inp("h")), m.ffn_sub_norm.register_forward_hook(out("n")), m.down_proj.register_forward_hook(out("y"))]
    try:
        with torch.no_grad(): c["mlp"] = m(x).detach().clone()
    finally:
        for h in hs: h.remove()
    return c


def masks(d: int, widths: list[int], seeds: list[int]) -> list[dict[str, Any]]:
    out = []
    for k in widths:
        if not 0 < k < d: raise ValueError(f"bad removed width {k}")
        items = [("tail", None, torch.arange(d-k, d))]
        for seed in seeds:
            g = torch.Generator().manual_seed(seed + 1000003*k)
            items.append(("random", seed, torch.randperm(d, generator=g)[:k].sort().values))
        for kind, seed, omit in items:
            flag = torch.zeros(d, dtype=torch.bool); flag[omit] = True
            keep = torch.arange(d)[~flag]
            out.append({"mask_id": f"{kind}_{'none' if seed is None else seed}_k{k}", "mask_type": kind, "seed": seed, "removed_width": k, "omit": omit, "keep": keep, "omit_sha256": sha(omit)})
    return out


def qabs(n: torch.Tensor, keep: torch.Tensor, omit: torch.Tensor) -> dict[str, Any]:
    v = n.float().abs(); keep, omit = keep.to(v.device), omit.to(v.device)
    mk, mo = v.index_select(-1, keep).amax(-1), v.index_select(-1, omit).amax(-1)
    flag = torch.zeros(v.shape[-1], dtype=torch.bool, device=v.device); flag[omit] = True
    ctl = flag[v.argmax(-1)]; rho = mo / mk.clamp_min(1e-30)
    qs = torch.quantile(rho.flatten(), torch.tensor([0., .25, .5, .75, .95, .99, 1.], device=v.device))
    ef = v.square().sum(-1); eo = v.index_select(-1, omit).square().sum(-1)
    return {"tokens": rho.numel(), "omitted_argmax_count": int(ctl.sum()), "omitted_argmax_fraction": float(ctl.float().mean()), "rho_mean": float(rho.mean()), "rho_quantiles": dict(zip(("q0","q25","q50","q75","q95","q99","q100"), map(float, qs.tolist()))), "omitted_energy_fraction_mean": float((eo/ef.clamp_min(1e-30)).mean())}


def run(dtype: torch.dtype, device: torch.device, args, mask_bank) -> dict[str, Any]:
    p = profile(dtype); m = make_mlp(args.hidden_size, args.intermediate_size, args.eps, dtype, device, args.seed)
    g = torch.Generator().manual_seed(args.seed + 1)
    x = torch.randn(args.batch_size, args.sequence_length, args.hidden_size, generator=g).to(device=device, dtype=dtype)
    c = capture(m, x)
    h0 = m.act_fn(c["gate"]) * c["up"]
    if not torch.equal(h0, c["h"]): raise AssertionError(f"F0 tensor mismatch {metrics(h0,c['h'])}")
    if not torch.equal(c["mlp"], c["y"]): raise AssertionError("F0 output hook mismatch")
    h, n, y = c["h"], c["n"], c["y"]
    gamma, w, bias = m.ffn_sub_norm.weight.detach(), m.down_proj.weight.detach(), m.down_proj.bias
    h32 = h.float(); r2 = h32.square().mean(-1, keepdim=True)
    n2 = gamma * (h32 * torch.rsqrt(r2 + m.ffn_sub_norm.variance_epsilon)).to(h.dtype)
    f2 = check("F2_NATIVE_SUBLN", n2, n, p["f2m"], p["f2r"])
    nq, wq = ActQuant.apply(n), WeightQuant.apply(w)
    y2 = F.linear(nq, wq, bias)
    f3n = check("F3_NATIVE_QAT_REBUILD", y2, y, p["f3m"], p["f3r"])
    rows = []
    for mask in mask_bank:
        keep, omit = mask["keep"].to(device), mask["omit"].to(device)
        h64, ga64, w64 = h.double(), gamma.double(), w.double(); b64 = bias.double() if bias is not None else None
        s64, z64 = h64.index_select(-1,keep), h64.index_select(-1,omit)
        nr2 = h64.square().mean(-1,keepdim=True); sr2 = (s64.square().sum(-1,keepdim=True)+z64.square().sum(-1,keepdim=True))/args.intermediate_size
        f1d = check("F1_DENOM_"+mask["mask_id"], sr2, nr2, p["f1m"], p["f1r"])
        nn64 = ga64*h64*torch.rsqrt(nr2+args.eps); yf = F.linear(nn64,w64,b64)
        yk = F.linear(nn64.index_select(-1,keep),w64.index_select(1,keep),b64); yo = F.linear(nn64.index_select(-1,omit),w64.index_select(1,omit),None)
        f1s = check("F1_SPLIT_"+mask["mask_id"], yk+yo, yf, p["f1m"], p["f1r"])
        f1t = check("F1_NULL_"+mask["mask_id"], yf-yk, yo, p["f1m"], p["f1r"])
        yks = F.linear(nq.index_select(-1,keep),wq.index_select(1,keep),bias); yos = F.linear(nq.index_select(-1,omit),wq.index_select(1,omit),None)
        f3s = check("F3_SHARED_SPLIT_"+mask["mask_id"], yks+yos, y2, p["spm"], p["spr"])
        f3t = check("F3_SHARED_NULL_"+mask["mask_id"], y2-yks, yos, p["spm"], p["spr"])
        ns = gamma.index_select(0,keep)*(h32.index_select(-1,keep)*torch.rsqrt(r2+m.ffn_sub_norm.variance_epsilon)).to(h.dtype)
        nsq = ActQuant.apply(ns); ya = F.linear(nsq,wq.index_select(1,keep),bias)
        ws = w.index_select(1,keep); wsq = WeightQuant.apply(ws); yn = F.linear(nsq,wsq,bias)
        swf = float(1/w.float().abs().mean().clamp_min(1e-5)); swn = float(1/ws.float().abs().mean().clamp_min(1e-5))
        ar = n.float().abs().amax(-1)/ns.float().abs().amax(-1).clamp_min(1e-30)
        rows.append({"mask_id":mask["mask_id"],"mask_type":mask["mask_type"],"seed":mask["seed"],"removed_width":mask["removed_width"],"omit_sha256":mask["omit_sha256"],"f1":{"denominator":f1d,"dense_split":f1s,"transport_null":f1t},"f3":{"shared_split":f3s,"transport_null":f3t},"f4":{"shared_to_activation_narrow":metrics(ya,yks),"activation_narrow_to_fully_narrow":metrics(yn,ya),"full_weight_scale":swf,"narrow_weight_scale":swn,"narrow_over_full_weight_scale":swn/max(swf,1e-30),"full_over_surviving_absmax_mean":float(ar.mean()),"full_over_surviving_absmax_max":float(ar.max())},"qabs":qabs(n,mask["keep"],mask["omit"])})
    return {"dtype":str(dtype).replace("torch.",""),"device":str(device),"shape":{"batch_size":args.batch_size,"sequence_length":args.sequence_length,"hidden_size":args.hidden_size,"intermediate_size":args.intermediate_size},"hashes":{"input":sha(x),"pre_subln":sha(h),"subln":sha(n),"down":sha(y)},"tolerances":p,"f0":{"tensor_location":True,"mlp_output_equals_down_output":True},"f2":f2,"f3_native":f3n,"masks":rows,"verdict":"ORACLE_INTEGRITY_F0_F4_PASS"}


def ints(s: str) -> list[int]: return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,default=Path("BITNET_EH1_ORACLE_INTEGRITY_FIREWALL_F0_F4_V1.json")); ap.add_argument("--device",default="cpu"); ap.add_argument("--dtypes",default="float32,bfloat16"); ap.add_argument("--hidden-size",type=int,default=1024); ap.add_argument("--intermediate-size",type=int,default=2816); ap.add_argument("--batch-size",type=int,default=2); ap.add_argument("--sequence-length",type=int,default=8); ap.add_argument("--eps",type=float,default=1e-5); ap.add_argument("--seed",type=int,default=20260716); ap.add_argument("--removed-widths",default="32,64,128"); ap.add_argument("--random-mask-seeds",default="101,211,307"); a=ap.parse_args()
    receipt={"firewall_id":ID,"scope":"PREEXECUTION_IMPLEMENTATION_INTEGRITY_NON_SCIENTIFIC","transformers_revision":TX_REV,"code_commit":commit(),"python_version":platform.python_version(),"torch_version":torch.__version__,"claim_ceiling":"Operational F0-F4 parity only; no mature-checkpoint or scientific mechanism inference."}
    try:
        torch.set_num_threads(max(1,min(4,torch.get_num_threads()))); dev=torch.device(a.device); bank=masks(a.intermediate_size,ints(a.removed_widths),ints(a.random_mask_seeds)); ds=[x.strip() for x in a.dtypes.split(",") if x.strip()]
        dm={"float32":torch.float32,"bfloat16":torch.bfloat16}; runs=[run(dm[d],dev,a,bank) for d in ds]
        receipt.update({"configuration":{"device":str(dev),"dtypes":ds,"hidden_size":a.hidden_size,"intermediate_size":a.intermediate_size,"batch_size":a.batch_size,"sequence_length":a.sequence_length,"eps":a.eps,"seed":a.seed,"removed_widths":ints(a.removed_widths),"random_mask_seeds":ints(a.random_mask_seeds),"mask_count":len(bank)},"runs":runs,"verdict":"ORACLE_INTEGRITY_FIREWALL_F0_F4_PASS","authorization_effect":"Synthetic pinned-operational F0-F4 firewall cleared; mature-checkpoint identity and atlas authorization remain separate gates."})
    except Exception as e:
        receipt.update({"verdict":"ORACLE_INTEGRITY_FIREWALL_F0_F4_FAIL","exception_type":type(e).__name__,"exception_message":str(e),"traceback":traceback.format_exc(),"authorization_effect":"No oracle atlas authorization."}); a.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)); print(json.dumps(receipt,indent=2,sort_keys=True)); raise
    a.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)); print(json.dumps(receipt,indent=2,sort_keys=True))
if __name__=="__main__": main()
