#!/usr/bin/env python3
# Parse ncu --csv level-profile output (graph-bypass OFF build, --no-multistream).
# no-tier-split => 1 launch per panel-etree level, leaf->root order = ID order.
import csv, io, sys

MET = {
 'sm__warps_active.avg.pct_of_peak_sustained_active':'warp',
 'sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active':'TC',
 'dram__throughput.avg.pct_of_peak_sustained_elapsed':'DRAM',
 'l1tex__throughput.avg.pct_of_peak_sustained_elapsed':'L1',
 'lts__throughput.avg.pct_of_peak_sustained_elapsed':'L2',
 'gpu__time_duration.sum':'dur',
}
def load(path):
    lines=open(path).read().splitlines()
    h=next(i for i,l in enumerate(lines) if l.startswith('"ID"'))
    rows={}
    for r in csv.DictReader(io.StringIO("\n".join(lines[h:]))):
        i=int(r['ID']); d=rows.setdefault(i,{'name':r['Kernel Name'],'grid':r['Grid Size']})
        m=MET.get(r['Metric Name'])
        if m:
            v=r['Metric Value'].replace(',','')
            d[m]=float(v) if v not in('','N/A') else 0.0
    return [rows[i] for i in sorted(rows)]

def tier(n):
    return 'small' if 'factor_small' in n else 'mid' if 'factor_mid' in n else 'big' if 'factor_big' in n else '?'
def gridx(g):
    return int(g.strip('()').split(',')[0])

def per_level(case, path):
    L=load(path)
    tot=sum(d['dur'] for d in L)
    print(f"\n################ {case}  (no-tier-split, B=1, tf32)  levels={len(L)}  factor_total={tot/1e6:.3f} ms ################")
    print(f"{'L':>3} {'tier':<5} {'grid':>7} {'dur_us':>8} {'time%':>6} {'warp%':>6} {'TC%':>5} {'DRAM%':>6} {'L1%':>6} {'L2%':>6}")
    band={'small':[0,0],'mid':[0,0],'big':[0,0]}  # [dur, count]
    for L_i,d in enumerate(L):
        t=tier(d['name']); pct=100*d['dur']/tot
        band[t][0]+=d['dur']; band[t][1]+=1
        print(f"{L_i:>3} {t:<5} {gridx(d['grid']):>7} {d['dur']/1e3:>8.1f} {pct:>5.1f}% "
              f"{d['warp']:>6.1f} {d['TC']:>5.1f} {d['DRAM']:>6.1f} {d['L1']:>6.1f} {d['L2']:>6.1f}")
    print(f"  --- band 요약 (tier=레벨 maxfsz 분류) ---")
    for t in ('small','mid','big'):
        dur,cnt=band[t]
        if cnt:
            # time-weighted occupancy across that tier's levels
            wsum=sum(d['dur']*d['warp'] for d in L if tier(d['name'])==t)/dur
            tcsum=sum(d['dur']*d['TC'] for d in L if tier(d['name'])==t)/dur
            print(f"    {t:<5}: levels={cnt:>2}  time%={100*dur/tot:>5.1f}  occ(warp%,tw)={wsum:>5.1f}  TC%(tw)={tcsum:>4.1f}")
    return tot

def tier_summary(case, path):
    L=load(path)
    tot=sum(d['dur'] for d in L)
    agg={}
    for d in L:
        t=tier(d['name']); a=agg.setdefault(t,{'dur':0,'w':0,'tc':0,'dram':0,'l1':0,'l2':0,'n':0})
        a['dur']+=d['dur']; a['n']+=1
        for k,m in (('w','warp'),('tc','TC'),('dram','DRAM'),('l1','L1'),('l2','L2')):
            a[k]+=d['dur']*d[m]
    print(f"\n==== {case} TIER 요약 (production tier-split, time-weighted) factor_total={tot/1e6:.3f} ms ====")
    print(f"{'tier':<6}{'launches':>9}{'time%':>7}{'warp%':>7}{'TC%':>6}{'DRAM%':>7}{'L1%':>6}{'L2%':>6}")
    for t in ('small','mid','big'):
        if t in agg:
            a=agg[t]; dur=a['dur']
            print(f"{t:<6}{a['n']:>9}{100*dur/tot:>6.1f}%{a['w']/dur:>7.1f}{a['tc']/dur:>6.1f}{a['dram']/dur:>7.1f}{a['l1']/dur:>6.1f}{a['l2']/dur:>6.1f}")

cases=[('case13659','13659'),('ACTIVSg25k','25k'),('ACTIVSg70k','70k')]
for name,c in cases:
    per_level(name, f'{c}_notiersplit.csv')
for name,c in cases:
    tier_summary(name, f'{c}_tiersplit.csv')
