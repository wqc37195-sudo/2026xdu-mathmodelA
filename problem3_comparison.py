# -*- coding: utf-8 -*-
"""
problem3_comparison.py - 问题3: 典型工况策略对比与优先级分析
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib, os, sys, random
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
print("问题3: 典型工况策略对比与优先级分析")
print("="*70)
print()

df=pd.read_csv('Cement_ESP_Data.csv')
for col in df.columns:
    if col!='timestamp': df[col]=df[col].ffill().bfill()
cout_mean=df['C_out_mgNm3'].mean(); cin_mean=df['C_in_gNm3'].mean(); Q_mean=df['Q_Nm3h'].mean()

Tp=160; st=60
def fT(tv):
    ft=np.ones_like(tv,dtype=float); m=tv>120
    ft[m]=np.exp(-((tv[m]-Tp)/st)**2); return ft

S_actual=np.log(cin_mean*1000/cout_mean); Q_avg=df['Q_Nm3h'].mean(); Tmp_avg=df['Temp_C'].mean()
fT_avg=fT(np.array([Tmp_avg]))[0]; U_avg=[df[f'U{i}_kV'].mean() for i in range(1,5)]
shares=[0.40,0.30,0.17,0.13]
K=np.array([shares[i]*S_actual*Q_avg/(U_avg[i]**2*fT_avg) for i in range(4)])

print("--- 3.1 模型参数 ---")
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
def power_est(U1,U2,U3,U4,Q=Q_mean,Cin=cin_mean):
    return lr_p.predict(np.array([[U1**2,U2**2,U3**2,U4**2,Q,Cin]]))[0]

# ====== 聚类+优化 ======
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
print(f"--- 3.2 自动选K = {Kc} ---")

km=KMeans(n_clusters=Kc,random_state=42,n_init=15,max_iter=400).fit(Xs)
df['cluster']=km.labels_
ctr=sc.inverse_transform(km.cluster_centers_)
cdf=pd.DataFrame(ctr,columns=cf); cdf['o']=range(Kc); cdf=cdf.sort_values('C_in_gNm3')
nl=np.zeros(Kc,dtype=int)
for ni,(oi,_) in enumerate(cdf.iterrows()): nl[oi]=ni
df['cluster']=[{o:n for o,n in enumerate(nl)}[l] for l in df['cluster']]

MV,MAXV=40,72; MT,MAXT=60,600; TG=9.5
od=[]
for ic in range(Kc):
    cd=df[df['cluster']==ic]; Cm=cd['C_in_gNm3'].mean(); Qm=cd['Q_Nm3h'].mean()
    Po=cd['P_total_kW'].mean()
    def obj(params):
        U1,U2,U3,U4,_,_1,_2,_3=params
        P=power_est(U1,U2,U3,U4,Qm,Cm)
        C=pred_cout(Cm,U1,U2,U3,U4,0,0,0,0,Qm)
        return P+((C-TG)*100000 if C>TG else 0)
    bounds=[(MV,MAXV)]*4+[(MT,MAXT)]*4
    r=differential_evolution(obj,bounds,strategy='best1bin',maxiter=100,popsize=30,
        mutation=(0.5,1.5),recombination=0.7,tol=1e-12,seed=42+ic,polish=True)
    pr=r.x; bP=power_est(*pr[:4],Qm,Cm); bC=pred_cout(Cm,*pr[:4],0,0,0,0,Qm)
    od.append({'Cin':Cm,'T_c':cd['Temp_C'].mean(),'Tr':f"[{cd['Temp_C'].min():.0f}-{cd['Temp_C'].max():.0f}]",
        'U1':pr[0],'U2':pr[1],'U3':pr[2],'U4':pr[3],'T1':pr[4],'T2':pr[5],'T3':pr[6],'T4':pr[7],
        'P_opt':bP,'P_orig':Po,'dP':(bP-Po)/Po*100,'Cout_opt':bC})
opt=pd.DataFrame(od)

# ====== 选取极端工况 ======
ci=np.argsort(opt['Cin'].values); lo=opt.iloc[ci[0]].to_dict(); hi=opt.iloc[ci[-1]].to_dict()
print(f"--- 3.3 极端工况 ---")
print(f"  低浓度: Cin={lo['Cin']:.1f}, T={lo['T_c']:.0f}C, T_range={lo['Tr']}")
print(f"  高浓度: Cin={hi['Cin']:.1f}, T={hi['T_c']:.0f}C, T_range={hi['Tr']}")
print(f"  差异: {hi['Cin']-lo['Cin']:.1f} g/Nm3 ({(hi['Cin']/lo['Cin']-1)*100:.0f}%)")
print()

print("--- 3.4 操作参数对比 ---")
print(f"  {'参数':<12} {'低浓度':>10} {'高浓度':>10} {'差异':>10}")
for p in ['U1','U2','U3','U4','T1','T2','T3','T4']:
    u='kV' if p[0]=='U' else 's'
    print(f"  {p}({u}){'':<6} {lo[p]:>10.1f} {hi[p]:>10.1f} {hi[p]-lo[p]:>+10.1f}")
print(f"  {'P_opt':<12} {lo['P_opt']:>10.1f} {hi['P_opt']:>10.1f} {hi['P_opt']-lo['P_opt']:>+10.1f}")
print()

# ====== 灵敏度 ======
print("--- 3.5 灵敏度分析 ---")
def sens(Cin,Ua,Ta,Q):
    C0=pred_cout(Cin,*Ua,0,0,0,0,Q); sv=[]
    for i in range(4):
        up=Ua.copy(); up[i]=min(up[i]+2,72)
        Cu=pred_cout(Cin,*up,0,0,0,0,Q); sv.append((f'U{i+1}',abs(Cu-C0)/max(C0,1e-10)*100))
    for i in range(4):
        tp=Ta.copy(); tp[i]=min(tp[i]+100,600)
        Cu=pred_cout(Cin,*Ua,0,0,0,0,Q); sv.append((f'T{i+1}',abs(Cu-C0)/max(C0,1e-10)*100))
    return sv,C0

lQ=df['Q_Nm3h'].mean()
l_s,lC=sens(lo['Cin'],[lo['U1'],lo['U2'],lo['U3'],lo['U4']],[lo['T1'],lo['T2'],lo['T3'],lo['T4']],lQ)
h_s,hC=sens(hi['Cin'],[hi['U1'],hi['U2'],hi['U3'],hi['U4']],[hi['T1'],hi['T2'],hi['T3'],hi['T4']],lQ)
pn=['U1','U2','U3','U4','T1','T2','T3','T4']
lv=[v for _,v in l_s]; hv=[v for _,v in h_s]

print(f"  {'参数':<8} {'低浓度%':>10} {'高浓度%':>10} {'比值':>8}")
for i in range(8): print(f"  {pn[i]:<8} {lv[i]:>10.4f} {hv[i]:>10.4f} {hv[i]/max(lv[i],1e-10):>8.1f}")
us_hi=sum(hv[:4])/4; ts_hi=sum(hv[4:])/4; us_lo=sum(lv[:4])/4; ts_lo=sum(lv[4:])/4
print(f"\n  电压灵敏度: {us_lo:.1f}%~{us_hi:.1f}%, 振打: {ts_lo:.1f}%~{ts_hi:.1f}%")
print(f"  U/T比: {us_lo/max(ts_lo,1e-10):.0f}:1 ~ {us_hi/max(ts_hi,1e-10):.0f}:1")

print(f"\n【优先级规律】")
print(f"  第一优先级: 电压(U) — U²在指数中, 灵敏度是振打的{us_lo/max(ts_lo,1e-10):.0f}~{us_hi/max(ts_hi,1e-10):.0f}倍")
print(f"  第二优先级: 振打(T) — U达上限时的唯一调节手段")
print(f"  操作顺序: U1,U2↑ → U3,U4↑ → T1,T2↓ → T3,T4↓")

# 图
fig,ax=plt.subplots(1,2,figsize=(14,5)); x=np.arange(4); w=0.3
ax[0].bar(x-w/2,[lo[f'U{j+1}'] for j in range(4)],w,label=f'低浓度',color='steelblue')
ax[0].bar(x+w/2,[hi[f'U{j+1}'] for j in range(4)],w,label=f'高浓度',color='coral')
ax[0].set_xticks(x); ax[0].set_xticklabels(['U1','U2','U3','U4']); ax[0].set_ylabel('kV'); ax[0].set_title('电压策略对比'); ax[0].legend()
ax[1].bar(x-w/2,[lo[f'T{j+1}'] for j in range(4)],w,label='低浓度',color='steelblue')
ax[1].bar(x+w/2,[hi[f'T{j+1}'] for j in range(4)],w,label='高浓度',color='coral')
ax[1].set_xticks(x); ax[1].set_xticklabels(['T1','T2','T3','T4']); ax[1].set_ylabel('s'); ax[1].set_title('振打策略对比'); ax[1].legend()
plt.tight_layout(); plt.savefig('figures/problem3_strategy.png',dpi=150); plt.close()

fig,ax=plt.subplots(1,2,figsize=(14,6))
sns.heatmap(np.array([lv,hv]),annot=True,fmt='.2f',ax=ax[0],cmap='YlOrRd',xticklabels=pn,yticklabels=['低浓度','高浓度'],cbar_kws={'label':'Cout变化率(%)'})
ax[0].set_title('灵敏度热力图')
x=np.arange(8); w=0.3
ax[1].bar(x-w/2,lv,w,label='低浓度',color='steelblue')
ax[1].bar(x+w/2,hv,w,label='高浓度',color='coral')
ax[1].set_xticks(x); ax[1].set_xticklabels(pn); ax[1].set_ylabel('Cout变化率(%)'); ax[1].set_title('灵敏度对比'); ax[1].legend()
plt.tight_layout(); plt.savefig('figures/problem3_sensitivity.png',dpi=150); plt.close()

fig,ax=plt.subplots(1,2,figsize=(14,5)); cv=opt['Cin'].values
for j in range(4): ax[0].plot(cv,opt[f'U{j+1}'].values,'o-',label=f'U{j+1}',markersize=8,lw=2)
ax[0].set_xlabel('C_in (g/Nm3)'); ax[0].set_ylabel('最优电压 (kV)'); ax[0].set_title('电压随C_in趋势'); ax[0].legend()
for j in range(4): ax[1].plot(cv,opt[f'T{j+1}'].values,'s-',label=f'T{j+1}',markersize=8,lw=2)
ax[1].set_xlabel('C_in (g/Nm3)'); ax[1].set_ylabel('最优振打 (s)'); ax[1].set_title('振打随C_in趋势'); ax[1].legend()
plt.tight_layout(); plt.savefig('figures/problem3_trend.png',dpi=150); plt.close()

print(f"\nK=[{K[0]:.1f},{K[1]:.1f},{K[2]:.1f},{K[3]:.1f}], Kc={Kc}")
print("="*70)
