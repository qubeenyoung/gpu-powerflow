import numpy as np, time, json
from harness import load_case, newtonpf, build_J, condest1, PROFILES, true_mismatch_inf, _run_to_conv

CASES=['case9','case30','case57','case118','case300',
       'case_illinois200','case1354pegase','case2869pegase','case3120sp',
       'case6470rte','case6515rte','case9241pegase','GBnetwork']
U32=2**-24  # fp32 unit roundoff ~5.96e-8

print(f"{'case':16s} {'n':>6s} {'kappa1(J)':>10s} {'k*u32':>8s} | "
      f"{'FP64 it/trueF':>16s} | {'Mixed it/trueF':>16s} | {'FP32 it/trueF':>16s} | pred")
rows=[]
for c in CASES:
    try:
        Yb,Sb,V0,ref,pv,pq=load_case(c)
    except Exception as e:
        print(f"{c:16s} LOAD FAIL {repr(e)[:50]}"); continue
    n=Yb.shape[0]
    res={}
    for name,cfg in PROFILES.items():
        res[name]=newtonpf(Yb,Sb,V0,ref,pv,pq,cfg,tol=1e-8,maxit=50)
    # kappa at FP64-converged V
    rf=res['FP64']
    Vm=np.abs(V0).copy(); Va=np.angle(V0).copy()
    # reconstruct converged V by rerunning FP64 returning V: quick redo
    from harness import ibus_mismatch
    Vc=_run_to_conv(Yb,Sb,V0,ref,pv,pq)
    kap=condest1(build_J(Yb,Vc,pv,pq))
    ku=kap*U32
    pred = "Mixed≈FP64" if ku<0.1 else ("Mixed risk" if ku<1 else "Mixed breaks")
    def fmt(r): return f"{r['iters']:2d}/{r['final_trueF']:.1e}"
    print(f"{c:16s} {n:6d} {kap:10.2e} {ku:8.1e} | {fmt(res['FP64']):>16s} | "
          f"{fmt(res['Mixed']):>16s} | {fmt(res['FP32']):>16s} | {pred}")
    rows.append(dict(case=c,n=int(n),kappa=float(kap),ku=float(ku),
        FP64=res['FP64']['final_trueF'],Mixed=res['Mixed']['final_trueF'],FP32=res['FP32']['final_trueF'],
        it64=res['FP64']['iters'],itM=res['Mixed']['iters'],it32=res['FP32']['iters'],
        etaM=float(np.mean(res['Mixed']['hist']['eta'])) if res['Mixed']['hist']['eta'] else None))
json.dump(rows,open('/tmp/pf_prec/resA.json','w'),indent=1)
print("\n-- Mixed forcing term eta (mean over iters) per case --")
for r in rows: print(f"  {r['case']:16s} eta_mean={r['etaM']:.2e}" if r['etaM'] else f"  {r['case']}: n/a")
