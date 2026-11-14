"""Addendum 5, Section 8: simulation comparing interval methods for rho = dMCB / (dMCB + DSC lost)
with 5 seeds (percentile bootstrap with non-positive denominators as +inf or dropped, and the
Fieller / test-inversion interval). Per-seed (dMCB, DSC lost) are bivariate normal with SDs taken
from the main study's primary cell (0.0048, 0.020). Only cases where R1 holds are scored.
No real data. Run: python scripts/addendum5_rho_sim.py  (about 10 minutes on CPU)
"""
import numpy as np
from scipy import stats
rng=np.random.default_rng(1)
n=5; tc=stats.t.ppf(0.975,n-1); B=2000
def tci(x):
    m=x.mean(); h=tc*x.std(ddof=1)/np.sqrt(n); return m-h,m+h
def boot(m,d,mode):
    idx=rng.integers(0,n,(B,n)); M=m[idx].mean(1); D=d[idx].mean(1); den=M+D
    r=np.where(den>0,M/np.where(den>0,den,1),np.inf if mode=="inf" else np.nan)
    if mode=="drop": r=r[np.isfinite(r)]
    return np.percentile(r,2.5),np.percentile(r,97.5),(den<=0).mean()
def fieller(m,d):
    # set of c with |t(L_c)|<=tc, L_c=(1-c)m - c d = m - c(m+d)
    a,b=m,m+d  # L_c = a - c b
    ma,mb=a.mean(),b.mean(); S=np.cov(a,b)/n; saa,sbb,sab=S[0,0],S[1,1],S[0,1]
    # (ma - c mb)^2 <= tc^2 (saa -2c sab + c^2 sbb)
    A=mb**2-tc**2*sbb; Bq=-2*(ma*mb-tc**2*sab); C=ma**2-tc**2*saa
    if A<=0: return -np.inf,np.inf
    disc=Bq**2-4*A*C
    if disc<0: return -np.inf,np.inf
    r1=(-Bq-np.sqrt(disc))/(2*A); r2=(-Bq+np.sqrt(disc))/(2*A); return r1,r2
sd_m,sd_d=0.0048,0.020
print("rho_true corr  P(R1) | mean%nonpos  max%nonpos | coverage|R1: inf drop fieller | 'contradicted'|R1: inf drop fieller | 'R2 rho-part'|R1: inf drop fieller")
for rho_t in [0.0,0.25,0.5]:
  for corr in [-0.3,0,0.3]:
    for mud in [0.028,0.012]:
      mum=rho_t*mud/(1-rho_t)
      cov=[[sd_m**2,corr*sd_m*sd_d],[corr*sd_m*sd_d,sd_d**2]]
      res=[];np_=[]
      for _ in range(1500):
        x=rng.multivariate_normal([mum,mud],cov,n); m,d=x[:,0],x[:,1]
        lo,hi=tci(m+d)
        if not (lo>0 and (m+d).mean()>=0.005): continue
        row=[]
        for mode in ["inf","drop"]:
            l,h,f=boot(m,d,mode); row.append((l,h)); 
        np_.append(f)
        row.append(fieller(m,d)); res.append(row)
      k=len(res)
      cov_=[np.mean([r[j][0]<=rho_t<=r[j][1] for r in res]) for j in range(3)]
      con=[np.mean([r[j][0]>0.25 for r in res]) for j in range(3)]
      r2=[np.mean([r[j][1]<0.25 for r in res]) for j in range(3)]
      print(f"{rho_t:4.2f} {corr:+.1f} mud={mud} P(R1)={k/1500:.2f} | {100*np.mean(np_):5.1f}% {100*np.max(np_):5.1f}% | "
            +" ".join(f"{c:.3f}" for c in cov_)+" | "+" ".join(f"{c:.3f}" for c in con)+" | "+" ".join(f"{c:.3f}" for c in r2))
