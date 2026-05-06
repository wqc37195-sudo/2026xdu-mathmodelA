# -*- coding: utf-8 -*-
"""
problem2_optimization.py - 问题2: 工况划分与协同优化 (Cout<=10mg/Nm3)
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib, os, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns, matplotlib.font_manager as fm, matplotlib.axes
fm.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
_zh=fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc',size=11)
_zh_t=fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc',size=13)
_zh_sm=fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc',size=9)
_ost=matplotlib.axes.Axes.set_title; _osx=matplotlib.axes.Axes.set_xlabel
_osy=matplotlib.axes.Axes.set_ylabel; _olg=matplotlib.axes.Axes.legend
def _nst(self,l,*a,**kw): kw.setdefault('fontproperties',_zh_t); return _ost(self,l,*a,**kw)
def _nsx(self,l,*a,**kw): kw.setdefault('fontproperties',_zh); return _osx(self,l,*a,**kw)
def _nsy(self,l,*a,**kw): kw.setdefault('fontproperties',_zh); return _osy(self,l,*a,**kw)
def _nlg(self,*a,**kw): kw.setdefault('prop',_zh_sm); return _olg(self,*a,**kw)
matplotlib.axes.Axes.set_title=_nst; matplotlib.axes.Axes.set_xlabel=_nsx
matplotlib.axes.Axes.set_ylabel=_nsy; matplotlib.axes.Axes.legend=_nlg
_osup=matplotlib.figure.Figure.suptitle
def _nsup(self,t,*a,**kw): kw.setdefault('fontproperties',_zh_t); return _osup(self,t,*a,**kw)
matplotlib.figure.Figure.suptitle=_nsup
_oat=matplotlib.axes.Axes.annotate
def _nat(self,t,xy,*a,**kw): kw.setdefault('fontproperties',_zh); return _oat(self,t,xy,*a,**kw)
matplotlib.axes.Axes.annotate=_nat
_otx=matplotlib.axes.Axes.text
def _ntx(self,x,y,s,*a,**kw): kw.setdefault('fontproperties',_zh); return _otx(self,x,y,s,*a,**kw)
matplotlib.axes.Axes.text=_ntx
if sys.stdout.encoding != 'utf-8': sys.stdout.reconfigure(encoding='utf-8',errors='replace')
plt.rcParams['figure.dpi']=120; plt.rcParams['axes.unicode_minus']=False
sns.set_style('whitegrid'); np.random.seed(42); os.makedirs('figures',exist_ok=True)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from scipy.optimize import differential_evolution

print("="*70)
print("问题2: 典型工况划分与协同优化")
print("="*70)
print()

df=pd.read_csv('Cement_ESP_Data.csv')
for col in df.columns:
    if col!='timestamp': df[col]=df[col].ffill().bfill()
cout_mean=df['C_out_mgNm3'].mean(); cin_mean=df['C_in_gNm3'].mean(); Q_mean=df['Q_Nm3h'].mean()
print(f"数据: {len(df)} 条, Cout均值={cout_mean:.2f} (CV={df['C_out_mgNm3'].std()/cout_mean*100:.2f}%)")
print()

# ====== M4模型 ======
Tp=160; st=60
def fT(tv):
    ft=np.ones_like(tv,dtype=float); m=tv>120
    ft[m]=np.exp(-((tv[m]-Tp)/st)**2); return ft

S_actual=np.log(cin_mean*1000/cout_mean); Q_avg=df['Q_Nm3h'].mean(); Tmp_avg=df['Temp_C'].mean()
fT_avg=fT(np.array([Tmp_avg]))[0]; U_avg=[df[f'U{i}_kV'].mean() for i in range(1,5)]
shares=[0.40,0.30,0.17,0.13]
K=np.array([shares[i]*S_actual*Q_avg/(U_avg[i]**2*fT_avg) for i in range(4)])
print("--- 2.1 M4模型 ---")
for i in range(4): print(f"  K{i+1}={K[i]:.2f}")
print()

def pred_cout(Cin_g,U1,U2,U3,U4,_,_1,_2,_3,Q,Tmp_v=None):
    if Tmp_v is None: Tmp_v=Tmp_avg
    ft=fT(np.array([Tmp_v]))[0]; Us=[U1,U2,U3,U4]
    S=sum(K[i]*Us[i]**2*ft/Q for i in range(4))
    return Cin_g*1000.0*np.exp(-S)

df['U1sq']=df['U1_kV']**2; df['U2sq']=df['U2_kV']**2
df['U3sq']=df['U3_kV']**2; df['U4sq']=df['U4_kV']**2
Xp=df[['U1sq','U2sq','U3sq','U4sq','Q_Nm3h','C_in_gNm3']].values
lr_p=LinearRegression(); lr_p.fit(Xp,df['P_total_kW'].values)
print(f"  功率模型 R2={lr_p.score(Xp,df['P_total_kW'].values):.4f}")
def power_est(U1,U2,U3,U4,Q=Q_mean,Cin=cin_mean):
    return lr_p.predict(np.array([[U1**2,U2**2,U3**2,U4**2,Q,Cin]]))[0]
print()

# ====== 自动选K聚类 ======
print("--- 2.2 自动工况划分 ---")
cf=['Temp_C','C_in_gNm3','Q_Nm3h']; sc=StandardScaler(); Xs=sc.fit_transform(df[cf].values)
import random; random.seed(42)
_samp_idx = random.sample(range(len(Xs)), min(2000, len(Xs)))
Xs_samp = Xs[_samp_idx]
sil={}; inert={}
for k in range(2,13):
    km_=KMeans(n_clusters=k,random_state=42,n_init=10,max_iter=300).fit(Xs)
    inert[k]=km_.inertia_
    sil[k]=silhouette_score(Xs_samp, KMeans(n_clusters=k,random_state=42,n_init=10,max_iter=300).fit(Xs_samp).labels_)
best_k=max(sil,key=sil.get)
k_vals=np.array(list(inert.keys())); deltas=np.diff(list(inert.values()))/inert[2]*100
elbow=2
for i,d in enumerate(deltas):
    if abs(d)<10: elbow=k_vals[i+1]; break
Kc=max(best_k,elbow)
if Kc<5 and ((df['Temp_C'].values>=140).sum()>50): Kc=max(Kc,5)
print(f"  轮廓系数最优 K={best_k} (分数={sil[best_k]:.4f}), 肘部 K={elbow}, 超高温>140C: {(df['Temp_C'].values>=140).sum()}条")
print(f"  >>> 选定 K = {Kc}")
print()

km=KMeans(n_clusters=Kc,random_state=42,n_init=15,max_iter=400).fit(Xs)
df['cluster']=km.labels_
ctr=sc.inverse_transform(km.cluster_centers_)
cdf=pd.DataFrame(ctr,columns=cf); cdf['o']=range(Kc); cdf=cdf.sort_values('C_in_gNm3')
nl=np.zeros(Kc,dtype=int)
for ni,(oi,_) in enumerate(cdf.iterrows()): nl[oi]=ni
df['cluster']=[{o:n for o,n in enumerate(nl)}[l] for l in df['cluster']]
ctr=cdf[cf].values

print(f"  {'工况':<8} {'样本':<8} {'温度(C)':<10} {'Cin':<12} {'Q':<10} {'温度范围'}")
for i in range(Kc):
    cd=df[df['cluster']==i]
    print(f"  K{i+1:<7} {len(cd):<8} {ctr[i,0]:>8.1f}   {ctr[i,1]:>10.2f}   {ctr[i,2]:>8.0f}   [{cd['Temp_C'].min():.0f}-{cd['Temp_C'].max():.0f}]")
print()

# 聚类图
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
ax1.plot(k_vals,[sil[k] for k in k_vals],'o-',color='steelblue',lw=2); ax1.axvline(Kc,color='red',ls='--',lw=2,label=f'K={Kc}')
ax1.set_xlabel('K'); ax1.set_ylabel('轮廓系数'); ax1.set_title('轮廓系数分析'); ax1.legend()
ax2.plot(k_vals,[inert[k] for k in k_vals],'o-',color='coral',lw=2); ax2.axvline(Kc,color='red',ls='--',lw=2,label=f'K={Kc}')
ax2.set_xlabel('K'); ax2.set_ylabel('Inertia'); ax2.set_title('肘部法则'); ax2.legend()
plt.tight_layout(); plt.savefig('figures/problem2_elbow.png',dpi=150); plt.close()
print("  图表已保存: figures/problem2_elbow.png")

# ====== PSO ======
def pso(obj_fn,bounds,np_=50,mi=150,sd=42):
    np.random.seed(sd); dim=len(bounds)
    lb=np.array([b[0] for b in bounds]); ub=np.array([b[1] for b in bounds])
    pos=np.random.uniform(lb,ub,(np_,dim)); vel=np.zeros((np_,dim))
    pbest=pos.copy(); pf=np.array([obj_fn(p) for p in pos])
    gi=pf.argmin(); gbest=pbest[gi].copy(); gf=pf[gi]
    for _ in range(mi):
        r1=np.random.random((np_,dim)); r2=np.random.random((np_,dim))
        vel=0.7*vel+1.5*r1*(pbest-pos)+1.5*r2*(gbest-pos)
        pos=np.clip(pos+vel,lb,ub)
        for i in range(np_):
            f=obj_fn(pos[i])
            if f<pf[i]: pf[i]=f; pbest[i]=pos[i].copy()
            if f<gf: gf=f; gbest=pos[i].copy()
    return gbest,gf

# ====== GA+PSO优化 ======
print("--- 2.3 GA vs PSO 优化 (C_out<=10, Min P) ---")
MV,MAXV=40,72; MT,MAXT=60,600; TG=9.5
og={}; op={}

for an,ar in [('GA',og),('PSO',op)]:
    for i in range(Kc):
        cd=df[df['cluster']==i]; Cm=cd['C_in_gNm3'].mean(); Qm=cd['Q_Nm3h'].mean()
        Po=cd['P_total_kW'].mean()
        def obj(params):
            U1,U2,U3,U4,_,_1,_2,_3=params
            P=power_est(U1,U2,U3,U4,Qm,Cm)
            C=pred_cout(Cm,U1,U2,U3,U4,0,0,0,0,Qm)
            return P+((C-TG)*100000 if C>TG else 0)
        bounds=[(MV,MAXV)]*4+[(MT,MAXT)]*4
        if an=='GA':
            bp=float('inf'); bpr=None
            for t in range(2):
                r=differential_evolution(obj,bounds,strategy='best1bin',maxiter=50,popsize=20,
                    mutation=(0.5,1.5),recombination=0.7,tol=1e-8,seed=42+i*10+t,polish=True)
                pt=r.x; Ct=pred_cout(Cm,*pt[:4],0,0,0,0,Qm)
                if Ct<=10 and r.fun<bp: bp=r.fun; bpr=pt.copy()
            if bpr is None: bpr=r.x
        else:
            bp=float('inf'); bpr=None
            for t in range(2):
                pr,pf_=pso(obj,bounds,30,50,100+i*10+t)
                Ct=pred_cout(Cm,*pr[:4],0,0,0,0,Qm)
                if Ct<=10 and power_est(*pr[:4],Qm,Cm)<bp: bp=power_est(*pr[:4],Qm,Cm); bpr=pr.copy()
            if bpr is None: bpr=pr
        bP=power_est(*bpr[:4],Qm,Cm); bU=list(bpr[:4]); bT=list(bpr[4:])
        bC=pred_cout(Cm,*bU,0,0,0,0,Qm)
        ar[i]={'U':bU,'T':bT,'P':bP,'C':bC,'Po':Po,'Cin':Cm,'dP':(bP-Po)/Po*100}
        sys.stdout.flush()

# 择优
gw=0; pw=0
for i in range(Kc):
    if og[i]['C']<=10 and op[i]['C']<=10: w='GA' if og[i]['P']<=op[i]['P'] else 'PSO'
    elif og[i]['C']<=10: w='GA'
    elif op[i]['C']<=10: w='PSO'
    else: w='GA' if og[i]['P']<=op[i]['P'] else 'PSO'
    if w=='GA': gw+=1
    else: pw+=1
opt={}
for i in range(Kc):
    if og[i]['C']<=10 and op[i]['C']<=10: opt[i]=og[i] if og[i]['P']<=op[i]['P'] else op[i]
    elif og[i]['C']<=10: opt[i]=og[i]
    elif op[i]['C']<=10: opt[i]=op[i]
    else: opt[i]=og[i] if og[i]['P']<=op[i]['P'] else op[i]
    opt[i]['algo']='GA' if opt[i] is og[i] else 'PSO'
avg_dP=np.mean([opt[i]['dP'] for i in range(Kc)])
print(f"  GA胜{gw}/{Kc}, PSO胜{pw}/{Kc}")
print()

# 结果表
print(f"  {'工况':<8} {'Cin':>8} {'P原始':>8} {'P最优':>8} {'变化%':>8} {'Cout最优':>10} {'算法':>8} {'U1-U4(kV)':>32} {'T1-T4(s)':>32}")
print("  "+"-"*120)
for i in range(Kc):
    r=opt[i]; us=f"[{r['U'][0]:.0f},{r['U'][1]:.0f},{r['U'][2]:.0f},{r['U'][3]:.0f}]"
    ts=f"[{r['T'][0]:.0f},{r['T'][1]:.0f},{r['T'][2]:.0f},{r['T'][3]:.0f}]"
    ok='达标' if r['C']<=10 else '超标'
    print(f"  K{i+1:<7} {r['Cin']:>8.2f} {r['Po']:>8.0f} {r['P']:>8.0f} {r['dP']:>+7.1f}% {r['C']:>10.2f} {r['algo']:>8} {us:>32} {ts:>32} {ok}")
print(f"  平均功率变化: {avg_dP:+.1f}%")

pd.DataFrame([{'工况':i+1,'Cin_gNm3':opt[i]['Cin'],
    'U1_kV':opt[i]['U'][0],'U2_kV':opt[i]['U'][1],'U3_kV':opt[i]['U'][2],'U4_kV':opt[i]['U'][3],
    'T1_s':opt[i]['T'][0],'T2_s':opt[i]['T'][1],'T3_s':opt[i]['T'][2],'T4_s':opt[i]['T'][3],
    'P_opt_kW':opt[i]['P'],'P_orig_kW':opt[i]['Po'],'dP_pct':opt[i]['dP'],
    'Cout_opt_mgNm3':opt[i]['C'],'算法':opt[i]['algo'],'达标':'是' if opt[i]['C']<=10 else '否'}
    for i in range(Kc)]).to_csv('optimization_results.csv',index=False,encoding='utf-8-sig')

# 图
fig,axes=plt.subplots(2,2,figsize=(14,10)); x=np.arange(Kc); w=0.3
ax=axes[0,0]
ax.bar(x-w/2,[opt[i]['Po'] for i in range(Kc)],w,label='原始',color='lightgray')
ax.bar(x+w/2,[opt[i]['P'] for i in range(Kc)],w,label='优化',color='steelblue')
ax.set_xticks(x); ax.set_xticklabels([f'K{i+1}' for i in range(Kc)],fontsize=8)
ax.set_ylabel('功率 (kW)'); ax.set_title('功率对比'); ax.legend()
ax=axes[0,1]
ax.bar(x-0.15,[opt[i]['C'] for i in range(Kc)],0.3,label='优化后',color='mediumseagreen')
ax.axhline(10,color='red',ls='--',lw=2,label='10 mg/Nm3')
ax.set_xticks(x); ax.set_xticklabels([f'K{i+1}' for i in range(Kc)],fontsize=8)
ax.set_ylabel('C_out (mg/Nm3)'); ax.set_title('排放达标'); ax.legend()
ck=[plt.cm.tab10(i/Kc) for i in range(Kc)]
ax=axes[1,0]
for i in range(min(Kc,6)):
    ax.bar(np.arange(4)+i*0.15-0.35,opt[i]['U'],0.15,label=f'K{i+1}',color=ck[i])
ax.set_xticks(np.arange(4)); ax.set_xticklabels(['U1','U2','U3','U4']); ax.set_ylabel('kV'); ax.set_title('最优电压'); ax.legend(fontsize=6)
ax=axes[1,1]
for i in range(min(Kc,6)):
    ax.bar(np.arange(4)+i*0.15-0.35,opt[i]['T'],0.15,label=f'K{i+1}',color=ck[i])
ax.set_xticks(np.arange(4)); ax.set_xticklabels(['T1','T2','T3','T4']); ax.set_ylabel('s'); ax.set_title('最优振打'); ax.legend(fontsize=6)
plt.tight_layout(); plt.savefig('figures/problem2_optimization.png',dpi=150); plt.close()

print(f"\nM4 K=[{K[0]:.1f},{K[1]:.1f},{K[2]:.1f},{K[3]:.1f}], Kc={Kc}, 平均dP={avg_dP:+.1f}%")
print("="*70)
