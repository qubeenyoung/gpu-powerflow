import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from harness import load_case, _run_to_conv, build_J, condest1
np.set_printoptions(suppress=True)
u32=2.0**-24

def q(x,ps=(50,90,99,100)): 
    x=np.asarray(x); return {p:np.percentile(x,p) for p in ps}

def analyze(c):
    Yb,Sb,V0,ref,pv,pq=load_case(c); n=Yb.shape[0]
    V=_run_to_conv(Yb,Sb,V0,ref,pv,pq)
    pvpq=np.r_[pv,pq]; npvpq=len(pvpq); npq=len(pq)
    Ib=Yb@V; S=V*np.conj(Ib)
    # ---- (A) LINEAR SOLVE precision: backward vs forward error of fp32 LU ----
    J=build_J(Yb,V,pv,pq,'fp64')
    mis=S-Sb; F=np.r_[mis[pvpq].real,mis[pq].imag]
    # use a generic RHS (b=J@e) so F!=0 (at convergence F~0); measure solver, not residual
    rng=np.random.default_rng(0); xtrue=rng.standard_normal(J.shape[0]); b=J@xtrue
    x64=spla.splu(J.tocsc()).solve(b)
    x32=spla.splu(J.astype(np.float32).tocsc()).solve(b.astype(np.float32)).astype(np.float64)
    # normwise relative backward error (Rigal-Gaches):  ||b-Jx|| / (||J|| ||x|| + ||b||)
    nrm=lambda v:np.linalg.norm(v,np.inf); Jn=spla.onenormest(J.tocsc())
    bwd32=nrm(b-J@x32)/(Jn*nrm(x32)+nrm(b))
    fwd32=nrm(x32-x64)/nrm(x64)
    kap=condest1(J)
    # ---- (B) MISMATCH precision: summation condition number kappa_sum per bus ----
    absrow=np.array(np.abs(Yb)@np.abs(V)).ravel()      # Σ_j |Y_ij V_j|
    ksum=absrow/np.maximum(np.abs(Ib),1e-300)          # cond of the bus-current dot product
    # restrict to buses that actually carry an equation (pvpq for P, pq for Q)
    ksum_eq=ksum[pvpq]
    # ---- (C) DATA distributions ----
    dist=dict(
      absV=q(np.abs(V)), absY=q(np.abs(Yb.data)),
      term=q(np.abs(np.asarray(np.abs(Yb).multiply(np.abs(V)[None,:]).data)).ravel()),
      absIb=q(np.abs(Ib)), absS=q(np.abs(S)), absSbus=q(np.abs(Sb)+1e-30),
      ksum=q(ksum_eq))
    # predicted fp32 mismatch floor from running-error bound: ~ u32 * kappa_sum * |S|
    pred_floor = u32*np.median(ksum_eq)*np.median(np.abs(S))
    return dict(case=c,n=n,kappa_J=kap, bwd32=bwd32, fwd32=fwd32, kJu=kap*u32,
                ksum_med=np.median(ksum_eq), ksum_90=np.percentile(ksum_eq,90),
                ksum_99=np.percentile(ksum_eq,99), pred_floor=pred_floor, dist=dist)

CASES=['case118','case300','case1354pegase','case3120sp','case9241pegase']
R=[analyze(c) for c in CASES]
print("="*100)
print("(A) LINEAR SOLVE  — fp32 LU: backward error ω (≈u32 이면 backward-stable) vs forward error (≈κ·u32)")
print(f"{'case':16s}{'κ(J)':>11s}{'κ(J)·u32':>10s}{'fp32 ω(backward)':>18s}{'fp32 fwd-err':>14s}{'fwd/(κu)':>10s}")
for r in R:
    print(f"{r['case']:16s}{r['kappa_J']:11.1e}{r['kJu']:10.1e}{r['bwd32']:18.1e}{r['fwd32']:14.1e}{r['fwd32']/max(r['kJu'],1e-30):10.2f}")
print(f"\n  u32 = {u32:.2e}.  ω≈u32 ⇒ fp32 LU는 후방안정(backward stable): 계수행렬을 ~u32만큼만 바꾼 정확해.")
print("="*100)
print("(B) MISMATCH  — 합산 조건수 κ_sum = Σ_j|Y_ij V_j| / |Ibus_i| (전력식 내부 상쇄), 그리고 예측 바닥 u32·κ_sum·|S|")
print(f"{'case':16s}{'κ_sum med':>11s}{'κ_sum 90%':>11s}{'κ_sum 99%':>11s}{'예측 fp32바닥':>14s}")
for r in R:
    print(f"{r['case']:16s}{r['ksum_med']:11.0f}{r['ksum_90']:11.0f}{r['ksum_99']:11.0f}{r['pred_floor']:14.1e}")
print("="*100)
print("(C) 입출력 데이터 분포 (백분위 50/90/99/max)")
for r in R:
    d=r['dist']
    print(f"\n[{r['case']}]  n={r['n']}")
    for k in ['absV','absY','term','absIb','absS','ksum']:
        v=d[k]; print(f"   {k:7s} 50%={v[50]:.3g}  90%={v[90]:.3g}  99%={v[99]:.3g}  max={v[100]:.3g}")
