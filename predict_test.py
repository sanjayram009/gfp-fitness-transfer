#!/usr/bin/env python3
"""
predict_test.py (v2) — TEST inference -> matching_results.tsv + candidate_pairs.tsv.
Reads df_drop_frac/top_k/with_char/selection/unique_target from meta.json (set by run_baseline.py)
so test-time blocking + selection exactly match what was measured on held-out data.

  python3 predict_test.py --test-dir "$TEST" --out-dir "$OUT" [--fallback] [--validator PATH]
"""
import argparse, json, os, subprocess, time, numpy as np, pandas as pd
import er_normalize as N, blocking as B, features as F, train_select as T, common as C
import lightgbm as lgb

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--test-dir",required=True); ap.add_argument("--out-dir",required=True)
    ap.add_argument("--fallback",action="store_true"); ap.add_argument("--fallback-col",default="name_tset")
    ap.add_argument("--fallback-thr",type=float,default=0.90); ap.add_argument("--threads",type=int,default=2)
    ap.add_argument("--validator",default=""); a=ap.parse_args()
    meta=json.load(open(os.path.join(a.out_dir,"meta.json")))
    countries=meta["countries"]; fmt=meta["gt_format"]
    ck=os.path.join(a.out_dir,"ckpt_test"); os.makedirs(ck,exist_ok=True)

    files=C.find_files(a.test_dir); log(f"test files={files}")
    norms={}
    for k in ("s1","s2","s3"):
        p=os.path.join(a.out_dir,f"norm_test_{k}.parquet")
        if os.path.exists(p): norms[k]=pd.read_parquet(p)
        else:
            df=C.read_tsv(files[k]); norms[k]=N.normalize_frame(df,C.detect_cols(list(df.columns)))
            norms[k].to_parquet(p,index=False)
        log(f"test/{k}: {len(norms[k]):,} rows")
    n1,n2,n3=norms["s1"],norms["s2"],norms["s3"]

    log("== build FULL test candidates (same df_drop_frac/top_k as training) ==")
    cand=B.build_candidates(n1,n2,n3,os.path.join(a.out_dir,"cand_test.parquet"),countries,gt=None,
                            df_drop_frac=meta["df_drop_frac"],k=meta["top_k"],threads=a.threads,
                            ckpt_dir=ck,s1_sample=None,with_char=meta.get("with_char",False),
                            char_k=meta.get("char_k",20))
    n_s1=n1[n1.country.isin(countries)].shape[0]
    log(f"test candidates={len(cand):,}  cands/S1={len(cand)/max(n_s1,1):.1f}")
    cand=F.add_features(cand,n1,{"2":n2,"3":n3})

    unique_target=meta.get("unique_target",False)
    if a.fallback:
        keep=T.fallback_rule(cand,a.fallback_col,a.fallback_thr); log(f"FALLBACK rule {a.fallback_col}>={a.fallback_thr}")
        if unique_target:
            prob=cand[a.fallback_col].to_numpy()
            keep=T.enforce_unique_target(cand,keep,prob)
    else:
        model=lgb.Booster(model_file=os.path.join(a.out_dir,"model.txt"))
        prob=model.predict(cand[meta["features"]].to_numpy(np.float32))
        if meta.get("selection")=="stage2":
            keep=T.select_expected_f05_mask(cand,prob)
            log("selection=stage2 (per-S1 expected-F0.5)")
        else:
            keep=prob>=meta["threshold"]
            log(f"selection=stage1  thr={meta['threshold']}")
        if unique_target:
            keep=T.enforce_unique_target(cand,keep,prob)
            log("applied D15: <=1 S1 per target")
        log(f"kept={int(keep.sum()):,}")

    kept=cand[keep]
    pred2={}; pred3={}
    for sid,tid,src in zip(kept.s1_id,kept.t_id,kept.src):
        (pred2 if str(src)=="2" else pred3).setdefault(sid,[]).append(tid)
    all_s1=list(n1[n1.country.isin(countries)]["id"])
    mr=os.path.join(a.out_dir,"matching_results.tsv"); cp=os.path.join(a.out_dir,"candidate_pairs.tsv")
    C.write_matching_results(mr,all_s1,pred2,pred3,fmt); C.write_candidate_pairs(cp,cand)
    n_pred=len(set(pred2)|set(pred3))
    log(f"wrote {mr} ({len(all_s1):,} S1 rows, {n_pred:,} non-empty) and {cp} ({len(cand):,} pairs)")

    candset=set(zip(cand.s1_id,cand.t_id))
    bad=[(s,t) for d in (pred2,pred3) for s,ts in d.items() for t in ts if (s,t) not in candset]
    log(f"subset check: {'OK' if not bad else f'FAIL {len(bad)} matches not in candidates'}")
    if a.validator and os.path.exists(a.validator):
        log("== running competition validator ==")
        print(subprocess.run(["python3",a.validator,mr],capture_output=True,text=True).stdout)

if __name__=="__main__": main()
