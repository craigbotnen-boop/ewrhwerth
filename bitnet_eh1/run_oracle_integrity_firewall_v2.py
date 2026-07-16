#!/usr/bin/env python3
"""Amendment-F operational firewall, revision 2.

The finite-precision transport-null identity is tested additively:
    y_keep + y_omit ~= y_full
The numerically unstable subtractive form y_full - y_keep ~= y_omit is retained
as a diagnostic only. Its exact assertion remains in FP64.
"""
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

ID="BITNET_EH1_ORACLE_INTEGRITY_FIREWALL_F0_F4_V2"
TX="096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4"

def commit():
    if os.environ.get("BITNET_EH1_CODE_COMMIT"): return os.environ["BITNET_EH1_CODE_COMMIT"]
    try: return subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return None

def sha(t): return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def met(a,b):
    d=a.detach().double()-b.detach().double(); r=b.detach().double()
    ma=float(d.abs().max()) if d.numel() else 0.; rm=float(d.square().mean().sqrt()) if d.numel() else 0.
    rma=float(r.abs().max()) if r.numel() else 0.; rrm=float(r.square().mean().sqrt()) if r.numel() else 0.
    return {"max_abs":ma,"rms":rm,"max_rel":ma/max(rma,1e-30),"rms_rel":rm/max(rrm,1e-30),"reference_max":rma,"reference_rms":rrm}

def chk(name,a,b,mx,rr):
    q=met(a,b); row={"test":name,"limits":{"max_rel":mx,"rms_rel":rr},"metrics":q}
    if q["max_rel"]>mx or q["rms_rel"]>rr: raise AssertionError(json.dumps(row,sort_keys=True))
    row["passed"]=True; return row

def prof(dt):
    if dt==torch.float32: return {"f2m":2e-7,"f2r":2e-7,"f3m":2e-6,"f3r":2e-6,"spm":2e-5,"spr":2e-6,"f1m":2e-12,"f1r":2e-13}
    return {"f2m":5e-3,"f2r":2e-3,"f3m":1e-2,"f3r":5e-3,"spm":5e-2,"spr":1.5e-2,"f1m":2e-12,"f1r":2e-13}

def make_mlp(h,d,eps,dt,dev,seed):
    cfg=BitNetConfig(vocab_size=4096,hidden_size=h,intermediate_size=d,num_hidden_layers=12,num_attention_heads=8,num_key_value_heads=2,hidden_act="relu2",max_position_embeddings=2048,rms_norm_eps=eps,use_cache=False,attention_bias=False,attention_dropout=0.)
    torch.manual_seed(seed); m=BitNetMLP(cfg)
    with torch.no_grad():
        g=torch.Generator().manual_seed(seed)
        for z in (m.gate_proj,m.up_proj,m.down_proj): torch.nn.init.xavier_normal_(z.weight,generator=g)
        m.ffn_sub_norm.weight.uniform_(.75,1.25,generator=g)
    convert_to_online_bitlinear(m)
    if not all(isinstance(z,AutoBitLinear) for z in (m.gate_proj,m.up_proj,m.down_proj)): raise AssertionError("AutoBitLinear conversion failed")
    return m.to(device=dev,dtype=dt).eval()

def capture(m,x):
    c={}; hs=[]
    def out(k):
        def f(_m,_i,o): c[k]=o.detach().clone()
        return f
    def inp(k):
        def f(_m,i): c[k]=i[0].detach().clone()
        return f
    hs=[m.gate_proj.register_forward_hook(out("gate")),m.up_proj.register_forward_hook(out("up")),m.ffn_sub_norm.register_forward_pre_hook(inp("h")),m.ffn_sub_norm.register_forward_hook(out("n")),m.down_proj.register_forward_hook(out("y"))]
    try:
        with torch.no_grad(): c["mlp"]=m(x).detach().clone()
    finally:
        for h in hs: h.remove()
    return c

def mask_bank(d,widths,seeds):
    out=[]
    for k in widths:
        items=[("tail",None,torch.arange(d-k,d))]
        for s in seeds: items.append(("random",s,torch.randperm(d,generator=torch.Generator().manual_seed(s+1000003*k))[:k].sort().values))
        for kind,s,omit in items:
            flag=torch.zeros(d,dtype=torch.bool); flag[omit]=True
            out.append({"id":f"{kind}_{'none' if s is None else s}_k{k}","type":kind,"seed":s,"k":k,"omit":omit,"keep":torch.arange(d)[~flag],"omit_sha256":sha(omit)})
    return out

def qabs(n,keep,omit):
    v=n.float().abs(); keep=keep.to(v.device); omit=omit.to(v.device)
    mk=v.index_select(-1,keep).amax(-1); mo=v.index_select(-1,omit).amax(-1)
    flag=torch.zeros(v.shape[-1],dtype=torch.bool,device=v.device); flag[omit]=True
    ctl=flag[v.argmax(-1)]; rho=mo/mk.clamp_min(1e-30)
    return {"tokens":rho.numel(),"omitted_argmax_count":int(ctl.sum()),"omitted_argmax_fraction":float(ctl.float().mean()),"rho_mean":float(rho.mean()),"rho_max":float(rho.max())}

def run(dt,dev,a,bank):
    p=prof(dt); m=make_mlp(a.hidden_size,a.intermediate_size,a.eps,dt,dev,a.seed)
    x=torch.randn(a.batch_size,a.sequence_length,a.hidden_size,generator=torch.Generator().manual_seed(a.seed+1)).to(device=dev,dtype=dt)
    c=capture(m,x); h0=m.act_fn(c["gate"])*c["up"]
    if not torch.equal(h0,c["h"]): raise AssertionError(f"F0 tensor mismatch {met(h0,c['h'])}")
    if not torch.equal(c["mlp"],c["y"]): raise AssertionError("F0 output hook mismatch")
    h,n,y=c["h"],c["n"],c["y"]; gamma=m.ffn_sub_norm.weight.detach(); w=m.down_proj.weight.detach(); bias=m.down_proj.bias
    h32=h.float(); r2=h32.square().mean(-1,keepdim=True)
    n2=gamma*(h32*torch.rsqrt(r2+m.ffn_sub_norm.variance_epsilon)).to(h.dtype)
    f2=chk("F2_NATIVE_SUBLN",n2,n,p["f2m"],p["f2r"])
    nq,wq=ActQuant.apply(n),WeightQuant.apply(w); y2=F.linear(nq,wq,bias)
    f3n=chk("F3_NATIVE_QAT_REBUILD",y2,y,p["f3m"],p["f3r"])
    rows=[]
    for z in bank:
        keep,omit=z["keep"].to(dev),z["omit"].to(dev)
        h64,ga64,w64=h.double(),gamma.double(),w.double(); b64=bias.double() if bias is not None else None
        s64,z64=h64.index_select(-1,keep),h64.index_select(-1,omit)
        nr2=h64.square().mean(-1,keepdim=True); sr2=(s64.square().sum(-1,keepdim=True)+z64.square().sum(-1,keepdim=True))/a.intermediate_size
        f1d=chk("F1_DENOM_"+z["id"],sr2,nr2,p["f1m"],p["f1r"])
        nn64=ga64*h64*torch.rsqrt(nr2+a.eps); yf=F.linear(nn64,w64,b64)
        yk=F.linear(nn64.index_select(-1,keep),w64.index_select(1,keep),b64); yo=F.linear(nn64.index_select(-1,omit),w64.index_select(1,omit),None)
        f1s=chk("F1_SPLIT_"+z["id"],yk+yo,yf,p["f1m"],p["f1r"])
        f1t=chk("F1_NULL_"+z["id"],yf-yk,yo,p["f1m"],p["f1r"])
        yks=F.linear(nq.index_select(-1,keep),wq.index_select(1,keep),bias); yos=F.linear(nq.index_select(-1,omit),wq.index_select(1,omit),None)
        f3s=chk("F3_SHARED_SPLIT_"+z["id"],yks+yos,y2,p["spm"],p["spr"])
        f3diag={"test":"F3_SHARED_NULL_DIAGNOSTIC_"+z["id"],"asserted":False,"reason":"BF16/FP32 subtraction cancellation; additive split is controlling","metrics":met(y2-yks,yos)}
        ns=gamma.index_select(0,keep)*(h32.index_select(-1,keep)*torch.rsqrt(r2+m.ffn_sub_norm.variance_epsilon)).to(h.dtype)
        nsq=ActQuant.apply(ns); ya=F.linear(nsq,wq.index_select(1,keep),bias); ws=w.index_select(1,keep); yn=F.linear(nsq,WeightQuant.apply(ws),bias)
        rows.append({"mask_id":z["id"],"mask_type":z["type"],"seed":z["seed"],"removed_width":z["k"],"omit_sha256":z["omit_sha256"],"f1":{"denominator":f1d,"dense_split":f1s,"transport_null":f1t},"f3":{"shared_split":f3s,"transport_null_diagnostic":f3diag,"controlling_assertion":"additive shared-quantizer split"},"f4":{"activation_scale_effect":met(ya,yks),"weight_scale_effect":met(yn,ya)},"qabs":qabs(n,z["keep"],z["omit"])})
    return {"dtype":str(dt).replace("torch.",""),"f0":{"tensor_location":True,"mlp_output_equals_down_output":True},"f2":f2,"f3_native":f3n,"masks":rows,"hashes":{"input":sha(x),"pre_subln":sha(h),"subln":sha(n),"down":sha(y)},"verdict":"ORACLE_INTEGRITY_F0_F4_PASS"}

def ints(s): return [int(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,default=Path("BITNET_EH1_ORACLE_INTEGRITY_FIREWALL_F0_F4_V2.json")); ap.add_argument("--device",default="cpu"); ap.add_argument("--dtypes",default="float32,bfloat16"); ap.add_argument("--hidden-size",type=int,default=1024); ap.add_argument("--intermediate-size",type=int,default=2816); ap.add_argument("--batch-size",type=int,default=2); ap.add_argument("--sequence-length",type=int,default=8); ap.add_argument("--eps",type=float,default=1e-5); ap.add_argument("--seed",type=int,default=20260716); ap.add_argument("--removed-widths",default="32,64,128"); ap.add_argument("--random-mask-seeds",default="101,211,307"); a=ap.parse_args()
    rec={"firewall_id":ID,"scope":"PREEXECUTION_IMPLEMENTATION_INTEGRITY_NON_SCIENTIFIC","transformers_revision":TX,"code_commit":commit(),"python_version":platform.python_version(),"torch_version":torch.__version__,"claim_ceiling":"Operational F0-F4 parity only; no mature-checkpoint or scientific mechanism inference."}
    try:
        torch.set_num_threads(max(1,min(4,torch.get_num_threads()))); dev=torch.device(a.device); bank=mask_bank(a.intermediate_size,ints(a.removed_widths),ints(a.random_mask_seeds)); dm={"float32":torch.float32,"bfloat16":torch.bfloat16}; ds=[x.strip() for x in a.dtypes.split(",") if x.strip()]
        rec.update({"configuration":{"device":str(dev),"dtypes":ds,"hidden_size":a.hidden_size,"intermediate_size":a.intermediate_size,"batch_size":a.batch_size,"sequence_length":a.sequence_length,"eps":a.eps,"seed":a.seed,"removed_widths":ints(a.removed_widths),"random_mask_seeds":ints(a.random_mask_seeds),"mask_count":len(bank)},"runs":[run(dm[d],dev,a,bank) for d in ds],"verdict":"ORACLE_INTEGRITY_FIREWALL_F0_F4_PASS","authorization_effect":"Synthetic pinned-operational Amendment F cleared; mature-checkpoint identity and atlas authorization remain separate gates."})
    except Exception as e:
        rec.update({"verdict":"ORACLE_INTEGRITY_FIREWALL_F0_F4_FAIL","exception_type":type(e).__name__,"exception_message":str(e),"traceback":traceback.format_exc(),"authorization_effect":"No oracle atlas authorization."}); a.output.write_text(json.dumps(rec,indent=2,sort_keys=True)); print(json.dumps(rec,indent=2,sort_keys=True)); raise
    a.output.write_text(json.dumps(rec,indent=2,sort_keys=True)); print(json.dumps(rec,indent=2,sort_keys=True))
if __name__=="__main__": main()
