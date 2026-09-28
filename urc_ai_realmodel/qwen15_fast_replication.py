#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, gc, os
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr, wilcoxon
from transformers import AutoTokenizer, AutoModelForCausalLM
import recoverability_gap as base
import natural_causal_extension as nce

SEED=20260927
MODEL="Qwen/Qwen2.5-1.5B-Instruct"

def load():
    tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
    m=AutoModelForCausalLM.from_pretrained(
        MODEL,torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation="eager"
    )
    m.eval(); return tok,m

class Capture:
    def __init__(self,m):
        self.layers=m.model.layers; self.values={}; self.handles=[]
        for i,layer in enumerate(self.layers):
            def mk(ii):
                def h(mod,inp,out): self.values[ii]=out.detach().float().cpu()
                return h
            self.handles.append(layer.self_attn.v_proj.register_forward_hook(mk(i)))
    def close(self):
        for h in self.handles:h.remove()

def single(m,tok,cap,text,idA,idB,att=True):
    ids=torch.tensor([tok.encode(text,add_special_tokens=False)],dtype=torch.long)
    cap.values.clear()
    with torch.inference_mode():
        o=m(input_ids=ids,output_attentions=att,output_hidden_states=True,use_cache=False,return_dict=True)
    lg=o.logits[0,-1,:].detach().float().cpu().numpy()
    margin=float(lg[idB]-lg[idA]); return ids,o,margin

def patch_pair(m,layer,ids,donor,positions,idA,idB):
    B=len(positions); xx=ids.repeat(B,1)
    def h(mod,inp,out):
        if isinstance(out,tuple):
            z=out[0].clone()
            for bi,p in enumerate(positions): z[bi,p,:]=donor[p,:].to(z.device,z.dtype)
            return (z,)+tuple(out[1:])
        z=out.clone()
        for bi,p in enumerate(positions): z[bi,p,:]=donor[p,:].to(z.device,z.dtype)
        return z
    hh=layer.register_forward_hook(h)
    try:
        with torch.inference_mode():
            o=m(input_ids=xx,output_hidden_states=True,use_cache=False,return_dict=True)
    finally: hh.remove()
    hf=o.hidden_states[-1][:,-1,:].detach().float().cpu().numpy()
    lg=o.logits[:,-1,:].detach().float().cpu().numpy()
    return hf,lg[:,idB]-lg[:,idA]

def jac_directional(m,layer,ids,donor_delta,idA,idB,positions):
    old=[p.requires_grad for p in m.parameters()]
    for p in m.parameters():p.requires_grad_(False)
    box={}
    def h(mod,inp,out):
        if isinstance(out,tuple):
            z=out[0].detach().requires_grad_(True); z.retain_grad(); box["z"]=z
            return (z,)+tuple(out[1:])
        z=out.detach().requires_grad_(True); z.retain_grad(); box["z"]=z; return z
    hh=layer.register_forward_hook(h)
    try:
        o=m(input_ids=ids,use_cache=False,return_dict=True)
        margin=o.logits[0,-1,idB]-o.logits[0,-1,idA]
        margin.backward()
        g=box["z"].grad[0].detach().float().cpu()
        vals=[float(torch.dot(g[p],donor_delta[p])) for p in positions]
    finally:
        hh.remove()
        for p,r in zip(m.parameters(),old):p.requires_grad_(r)
        m.zero_grad(set_to_none=True)
    return vals

def rank_desc(x,pos):
    return int(np.where(np.argsort(-np.asarray(x))==pos)[0][0])+1

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--n",type=int,default=10)
    a=ap.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    tok,m=load();idA,idB,sA,sB=nce.label_ids(tok);cap=Capture(m)
    L=len(cap.layers); layers=[0,L//2,L-1]; mid=layers[1]
    pairs=nce.make_pairs(a.n)
    rows=[]
    for i,p in enumerate(pairs):
        ids,o,bm=single(m,tok,cap,p["A"],idA,idB,True)
        base_final=o.hidden_states[-1][0,-1,:].detach().float().cpu().numpy()
        base_val=cap.values[mid].clone()
        dids,od,dm=single(m,tok,cap,p["B"],idA,idB,False)
        if ids.shape!=dids.shape: continue
        diff=np.flatnonzero(ids[0].numpy()!=dids[0].numpy())
        if len(diff)!=2: continue
        cpos,dpos=map(int,diff)
        Ah=o.attentions[mid][0].detach().float().cpu().numpy()
        raw=Ah[:,-1,:].mean(0)
        hn=base.head_residual_norms(m,"qwen",mid,base_val,0)
        contrib=nce.contribution_target(Ah,hn)
        base_mid=o.hidden_states[mid+1][0].detach().float().cpu()
        donor_mid=od.hidden_states[mid+1][0].detach().float().cpu()
        delta=donor_mid-base_mid
        source_delta_norm=np.linalg.norm(delta.numpy(),axis=1)+1e-9
        hf,mg=patch_pair(m,cap.layers[mid],ids,donor_mid,[cpos,dpos],idA,idB)
        hdelta_raw=np.linalg.norm(hf-base_final[None,:],axis=1)
        hdelta=hdelta_raw/source_delta_norm[[cpos,dpos]]
        md=mg-bm
        jac=jac_directional(m,cap.layers[mid],ids,delta,idA,idB,[cpos,dpos])
        # early/late causal-token patch
        shifts={}
        for li in [layers[0],layers[-1]]:
            donor=od.hidden_states[li+1][0].detach().float().cpu()
            _,x=patch_pair(m,cap.layers[li],ids,donor,[cpos],idA,idB)
            shifts[str(li)]=float(x[0]-bm)
        rows.append({
            "pair_id":p["pair_id"],"N":ids.shape[1],
            "causal_pos":cpos,"distractor_pos":dpos,
            "raw_rank_causal":rank_desc(raw,cpos),"raw_rank_distractor":rank_desc(raw,dpos),
            "contrib_rank_causal":rank_desc(contrib,cpos),"contrib_rank_distractor":rank_desc(contrib,dpos),
            "causal_hidden_effect":float(hdelta[0]),"distractor_hidden_effect":float(hdelta[1]),
            "causal_hidden_effect_raw":float(hdelta_raw[0]),"distractor_hidden_effect_raw":float(hdelta_raw[1]),
            "causal_source_delta_norm":float(source_delta_norm[cpos]),"distractor_source_delta_norm":float(source_delta_norm[dpos]),
            "causal_margin_shift":float(md[0]),"distractor_margin_shift":float(md[1]),
            "causal_toward_donor":int(md[0]>0),"distractor_toward_donor":int(md[1]>0),
            "jac_attr_causal":jac[0],"jac_attr_distractor":jac[1],
            "causal_shift_early":shifts[str(layers[0])],"causal_shift_mid":float(md[0]),"causal_shift_late":shifts[str(layers[-1])],
        })
        del o,od;gc.collect()
        print(i+1,len(pairs),rows[-1],flush=True)
    cap.close()
    df=pd.DataFrame(rows);df.to_csv(out/"large_fast_examples.csv",index=False)
    finite=np.concatenate([df.causal_margin_shift.values,df.distractor_margin_shift.values])
    approx=np.concatenate([df.jac_attr_causal.values,df.jac_attr_distractor.values])
    rho=float(spearmanr(approx,finite).statistic)
    rhoabs=float(spearmanr(np.abs(approx),np.abs(finite)).statistic)
    s={
      "model":"qwen15","model_id":MODEL,"layers":layers,"n":len(df),
      "median_causal_hidden_effect":float(df.causal_hidden_effect.median()),
      "median_distractor_hidden_effect":float(df.distractor_hidden_effect.median()),
      "median_causal_margin_shift":float(df.causal_margin_shift.median()),
      "median_distractor_margin_shift":float(df.distractor_margin_shift.median()),
      "causal_toward_donor_fraction":float(df.causal_toward_donor.mean()),
      "distractor_toward_donor_fraction":float(df.distractor_toward_donor.mean()),
      "median_raw_rank_causal":float(df.raw_rank_causal.median()),
      "median_contrib_rank_causal":float(df.contrib_rank_causal.median()),
      "contribution_rank_improves_fraction":float((df.contrib_rank_causal<df.raw_rank_causal).mean()),
      "rho_directional_jacobian_vs_finite_margin":rho,
      "rho_abs_directional_jacobian_vs_abs_margin":rhoabs,
      "median_layer_causal_shift":{"early":float(df.causal_shift_early.median()),"mid":float(df.causal_shift_mid.median()),"late":float(df.causal_shift_late.median())},
    }
    try:
      s["wilcoxon_causal_gt_distractor_hidden_p"]=float(wilcoxon(df.causal_hidden_effect,df.distractor_hidden_effect,alternative="greater").pvalue)
      s["wilcoxon_causal_gt_distractor_abs_margin_p"]=float(wilcoxon(np.abs(df.causal_margin_shift),np.abs(df.distractor_margin_shift),alternative="greater").pvalue)
    except Exception:
      pass
    (out/"summary.json").write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2),flush=True)
if __name__=="__main__":main()
