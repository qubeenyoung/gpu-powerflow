import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
import warnings; warnings.filterwarnings("ignore")

# ---------- case loading ----------
def load_case(name):
    """Return Ybus(csc c128), Sbus(c128), V0(c128 flat-ish start), ref,pv,pq."""
    try:
        import pypower.api as pa
        if hasattr(pa, name):
            from pypower.api import makeYbus, makeSbus, bustypes, ext2int
            ppc = ext2int(getattr(pa, name)())
            base,bus,gen,branch = ppc["baseMVA"],ppc["bus"],ppc["gen"],ppc["branch"]
            Ybus,_,_ = makeYbus(base,bus,branch)
            ref,pv,pq = bustypes(bus,gen)
            Sbus = makeSbus(base,bus,gen)
            Vconv = bus[:,7]*np.exp(1j*np.deg2rad(bus[:,8]))
            return _flatstart(Ybus.tocsc().astype(np.complex128), Sbus.astype(np.complex128), Vconv, ref,pv,pq)
    except Exception:
        pass
    import pandapower as pp, pandapower.networks as nw
    net = getattr(nw, name)()
    pp.runpp(net, numba=False, max_iteration=80, init="flat")
    it = net._ppc['internal']
    Ybus = it['Ybus'].tocsc().astype(np.complex128)
    Sbus = it['Sbus'].astype(np.complex128)
    Vconv = it['V'].astype(np.complex128)
    ref,pv,pq = np.atleast_1d(it['ref']).astype(int),np.atleast_1d(it['pv']).astype(int),np.atleast_1d(it['pq']).astype(int)
    return _flatstart(Ybus,Sbus,Vconv,ref,pv,pq)

def _flatstart(Ybus,Sbus,Vconv,ref,pv,pq):
    n=Ybus.shape[0]; Vm=np.ones(n); Va=np.zeros(n)
    Vm[pv]=np.abs(Vconv[pv]); Vm[ref]=np.abs(Vconv[ref]); Va[ref]=np.angle(Vconv[ref])
    V0=Vm*np.exp(1j*Va)
    return Ybus,Sbus,V0.astype(np.complex128),ref,pv,pq

# ---------- precision emulation ----------
def r32c(x): return x.astype(np.complex64).astype(np.complex128)   # fp32 storage of complex
def r32r(x): return x.astype(np.float32).astype(np.float64)        # fp32 storage of real

def ibus_mismatch(Ybus, V, Sbus, pv, pq, mode):
    """Return S(=V*conj(Ybus V)) in c128 and F (real residual) under precision `mode`."""
    pvpq=np.r_[pv,pq]
    if mode=='fp64':
        Ib=Ybus@V; S=V*np.conj(Ib)
    elif mode=='fp32':                       # complex64 storage + fp32 accumulate
        Yb=Ybus.astype(np.complex64); Vc=V.astype(np.complex64)
        Ib=(Yb@Vc); S=(Vc*np.conj(Ib)).astype(np.complex128)
    elif mode=='fp32acc64':                  # fp32 storage, fp64 accumulate (the "compensated/acc-fix")
        Yb=r32c(Ybus.toarray()) if False else Ybus.astype(np.complex64).astype(np.complex128)
        Vc=r32c(V); Ib=Yb@Vc; S=Vc*np.conj(Ib)
    else: raise ValueError(mode)
    mis=S-Sbus
    F=np.r_[mis[pvpq].real, mis[pq].imag]
    return S,F

def true_mismatch_inf(Ybus,V,Sbus,pv,pq):
    pvpq=np.r_[pv,pq]; Ib=Ybus@V; mis=V*np.conj(Ib)-Sbus
    return np.max(np.abs(np.r_[mis[pvpq].real, mis[pq].imag]))

def build_J(Ybus,V,pv,pq,jac='fp64'):
    pvpq=np.r_[pv,pq]; n=len(V)
    Ib=Ybus@V
    diagV=sp.diags(V); diagIb=sp.diags(Ib); diagVn=sp.diags(V/np.abs(V))
    dS_dVm=diagV@np.conj(Ybus@diagVn)+np.conj(diagIb)@diagVn
    dS_dVa=1j*diagV@np.conj(diagIb-Ybus@diagV)
    J11=dS_dVa[np.ix_(pvpq,pvpq)].real
    J12=dS_dVm[np.ix_(pvpq,pq)].real
    J21=dS_dVa[np.ix_(pq,pvpq)].imag
    J22=dS_dVm[np.ix_(pq,pq)].imag
    J=sp.bmat([[J11,J12],[J21,J22]],format='csc')
    if jac=='fp32': J=J.astype(np.float32).astype(np.float64)
    return J

def lin_solve(J,F,solve='fp64'):
    """Return dx and forcing term eta=||F-J dx||/||F|| (measured in fp64)."""
    if solve=='fp64':
        lu=spla.splu(J.tocsc()); dx=lu.solve(F)
    else:  # true fp32 sparse LU
        lu=spla.splu(J.astype(np.float32).tocsc()); dx=lu.solve(F.astype(np.float32)).astype(np.float64)
    eta=np.linalg.norm(F-J@dx)/max(np.linalg.norm(F),1e-300)
    return dx,eta

def newtonpf(Ybus,Sbus,V0,ref,pv,pq,cfg,tol=1e-8,maxit=50):
    Va=np.angle(V0).copy(); Vm=np.abs(V0).copy(); n=len(V0)
    pvpq=np.r_[pv,pq]; npvpq=len(pvpq); npq=len(pq)
    hist={'normF':[],'trueF':[],'eta':[]}
    V=Vm*np.exp(1j*Va)
    for it in range(maxit):
        if cfg['state']=='fp32':  # emulate fp32 state storage
            Vm=r32r(Vm); Va=r32r(Va); V=Vm*np.exp(1j*Va)
        S,F=ibus_mismatch(Ybus,V,Sbus,pv,pq,cfg['resid'])
        nF=np.max(np.abs(F))                       # solver-perceived
        tF=true_mismatch_inf(Ybus,V,Sbus,pv,pq)    # honest accuracy (fp64)
        hist['normF'].append(nF); hist['trueF'].append(tF)
        if nF<tol: break
        J=build_J(Ybus,V,pv,pq,cfg['jac'])
        dx,eta=lin_solve(J,F,cfg['solve']); hist['eta'].append(eta)
        if cfg['update']=='fp32': dx=r32r(dx)
        Va[pvpq]=Va[pvpq]-dx[:npvpq]; Vm[pq]=Vm[pq]-dx[npvpq:]
        V=Vm*np.exp(1j*Va)
    converged = hist['trueF'][-1] < 1e-6
    return {'iters':it,'converged':converged,'final_normF':hist['normF'][-1],
            'final_trueF':hist['trueF'][-1],'hist':hist}

def condest1(J):
    """1-norm condition estimate kappa_1(J) via onenormest + splu."""
    J=J.tocsc()
    try:
        lu=spla.splu(J)
        n=J.shape[0]
        Ainv=spla.LinearOperator((n,n),matvec=lambda b:lu.solve(b),rmatvec=lambda b:lu.solve(b,'T'))
        normJ=spla.onenormest(J); normInv=spla.onenormest(Ainv)
        return normJ*normInv
    except Exception as e:
        return float('nan')

PROFILES={
 'FP64':  dict(state='fp64',resid='fp64',jac='fp64',solve='fp64',update='fp64'),
 'Mixed': dict(state='fp64',resid='fp64',jac='fp32',solve='fp32',update='fp64'),
 'FP32':  dict(state='fp32',resid='fp32',jac='fp32',solve='fp32',update='fp32'),
}
if __name__=='__main__':
    Yb,Sb,V0,ref,pv,pq=load_case('case118')
    for name,cfg in PROFILES.items():
        r=newtonpf(Yb,Sb,V0,ref,pv,pq,cfg)
        print(f"{name:6s} iters={r['iters']:2d} conv={r['converged']} final_trueF={r['final_trueF']:.2e} "
              f"final_normF={r['final_normF']:.2e} eta={[f'{e:.1e}' for e in r['hist']['eta'][:4]]}")
    print("kappa_1(J@conv)=",f"{condest1(build_J(Yb,V0*0+ (np.abs(V0)*np.exp(1j*np.angle(V0))),pv,pq)):.2e}")

def _run_to_conv(Ybus,Sbus,V0,ref,pv,pq,maxit=50):
    import numpy as np
    Va=np.angle(V0).copy(); Vm=np.abs(V0).copy()
    pvpq=np.r_[pv,pq]; npvpq=len(pvpq)
    V=Vm*np.exp(1j*Va)
    for it in range(maxit):
        S,F=ibus_mismatch(Ybus,V,Sbus,pv,pq,'fp64')
        if np.max(np.abs(F))<1e-10: break
        J=build_J(Ybus,V,pv,pq,'fp64'); dx,_=lin_solve(J,F,'fp64')
        Va[pvpq]-=dx[:npvpq]; Vm[pq]-=dx[npvpq:]; V=Vm*np.exp(1j*Va)
    return V
