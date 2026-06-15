import numpy as np, json
from harness import load_case, newtonpf

# Ablation configs: start from full FP32, promote one axis at a time + the accumulator fix.
CFG = {
 'FP32(all)':        dict(state='fp32',resid='fp32',     jac='fp32',solve='fp32',update='fp32'),
 'FP32+state64':     dict(state='fp64',resid='fp32',     jac='fp32',solve='fp32',update='fp64'),
 'FP32+resid64':     dict(state='fp32',resid='fp64',     jac='fp32',solve='fp32',update='fp32'),
 'FP32+resid(acc64)':dict(state='fp32',resid='fp32acc64',jac='fp32',solve='fp32',update='fp32'),
 'state64+resid(acc64)':dict(state='fp64',resid='fp32acc64',jac='fp32',solve='fp32',update='fp64'),
 'Mixed(ref)':       dict(state='fp64',resid='fp64',     jac='fp32',solve='fp32',update='fp64'),
}
CASES=['case300','case1354pegase','case3120sp','case9241pegase']
print("Per-op precision ablation — final TRUE ||F||_inf (fp64-recomputed) / iters\n")
hdr=f"{'config':22s}"+ "".join(f"{c.replace('case',''):>16s}" for c in CASES); print(hdr)
loaded={c:load_case(c) for c in CASES}
rows={}
for name,cfg in CFG.items():
    line=f"{name:22s}"; rows[name]={}
    for c in CASES:
        Yb,Sb,V0,ref,pv,pq=loaded[c]
        r=newtonpf(Yb,Sb,V0,ref,pv,pq,cfg,tol=1e-8,maxit=60)
        line+=f"  {r['final_trueF']:.1e}/{r['iters']:<2d}  "
        rows[name][c]=(r['final_trueF'],r['iters'],r['converged'])
    print(line)
json.dump({k:{c:[float(v[0]),v[1],bool(v[2])] for c,v in d.items()} for k,d in rows.items()},
          open('/tmp/pf_prec/resB.json','w'),indent=1)
print("\nInterpretation key: state64 isolates state-precision floor; resid64 isolates residual floor;")
print("resid(acc64)=fp32 storage + fp64 accumulation (the cheap GPU-feasible fix).")
