# -*- coding: utf-8 -*-
"""
problem4_stricter.py - 问题4: 排放标准收紧(10->5 mg/Nm3)定量分析
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
print("问题4: 排放标准收紧 (10 -> 5 mg/Nm3) 定量分析")
print("="*70)
print()

df=pd.read_csv('Cement_ESP_Data.csv')
for col in df.columns:
    if col!='timestamp': df[col]=df[col].ffill().bfill()
cout_mean=df['C_out_mgNm3'].mean(); cin_mean=df['C_in_gNm3'].mean(); Q_mean=df['Q_Nm3h'].mean()
print(f"数据: {len(df)} 条, 原始C_out={cout_mean:.1f} >> 10 >> 5 mg/Nm3")
print()

Tp=160; st=60
def fT(tv):
    ft=np.ones_like(tv,dtype=float); m=tv>120
    ft[m]=np.exp(-((tv[m]-Tp)/st)**2); return ft

S_actual=np.log(cin_mean*1000/cout_mean); Q_avg=df['Q_Nm3h'].mean(); Tmp_avg=df['Temp_C'].mean()
fT_avg=fT(np.array([Tmp_avg]))[0]; U_avg=[df[f'U{i}_kV'].mean() for i in range(1,5)]
shares=[0.40,0.30,0.17,0.13]
K=np.array([shares[i]*S_actual*Q_avg/(U_avg[i]**2*fT_avg) for i in range(4)])
print("--- 4.1 M4模型 ---")
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
print(f"  功率模型 R2={lr_p.score(Xp,df['P_total_kW'].values):.4f}")
print()

# ====== 聚类 ======
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
print(f"--- 4.2 工况划分: K={Kc} ---")
km=KMeans(n_clusters=Kc,random_state=42,n_init=15,max_iter=400).fit(Xs)
df['cluster']=km.labels_
ctr=sc.inverse_transform(km.cluster_centers_)
cdf=pd.DataFrame(ctr,columns=cf); cdf['o']=range(Kc); cdf=cdf.sort_values('C_in_gNm3')
nl=np.zeros(Kc,dtype=int)
for ni,(oi,_) in enumerate(cdf.iterrows()): nl[oi]=ni
df['cluster']=[{o:n for o,n in enumerate(nl)}[l] for l in df['cluster']]
for i in range(Kc):
    cd=df[df['cluster']==i]
    print(f"  K{i+1}: n={len(cd)}, Cin={cd['C_in_gNm3'].mean():.1f}, T={cd['Temp_C'].mean():.0f}C")
print()

# ====== 10mg vs 5mg 优化 ======
print("--- 4.3 10mg vs 5mg 优化搜索 ---")
MV,MAXV=40,72; MT,MAXT=60,600
opt10={}; opt5={}

for tg,store in [(9.5,opt10),(4.8,opt5)]:
    for i in range(Kc):
        cd=df[df['cluster']==i]; Cm=cd['C_in_gNm3'].mean(); Qm=cd['Q_Nm3h'].mean()
        def obj(params):
            U1,U2,U3,U4,_,_1,_2,_3=params
            P=power_est(U1,U2,U3,U4,Qm,Cm)
            C=pred_cout(Cm,U1,U2,U3,U4,0,0,0,0,Qm)
            return P+((C-tg)*500000 if C>tg else 0)
        bounds=[(MV,MAXV)]*4+[(MT,MAXT)]*4
        r=differential_evolution(obj,bounds,strategy='best1bin',maxiter=120,popsize=35,
            mutation=(0.6,1.8),recombination=0.7,tol=1e-12,seed=200+i,polish=True)
        pr=r.x; bP=power_est(*pr[:4],Qm,Cm); bC=pred_cout(Cm,*pr[:4],0,0,0,0,Qm)
        store[i]={'U':list(pr[:4]),'T':list(pr[4:]),'P':bP,'C':bC,'Cin':Cm,'Q':Qm}
        sys.stdout.flush()

# ====== 对比 ======
print(f"\n--- 4.4 定量对比 ---")
print(f"  {'工况':<8} {'Cin':>8} {'P_10mg':>10} {'P_5mg':>10} {'增幅%':>10} {'Cout_5mg':>10}")
total_P10=0; total_P5=0; dps=[]
for i in range(Kc):
    dp_pct=(opt5[i]['P']-opt10[i]['P'])/opt10[i]['P']*100
    total_P10+=opt10[i]['P']; total_P5+=opt5[i]['P']; dps.append(dp_pct)
    print(f"  K{i+1:<7} {opt5[i]['Cin']:>8.2f} {opt10[i]['P']:>10.0f} {opt5[i]['P']:>10.0f} {dp_pct:>+9.1f}% {opt5[i]['C']:>10.2f}")
avg_dp=np.mean(dps)
print(f"\n  P_10mg总计={total_P10:.0f} kW, P_5mg总计={total_P5:.0f} kW")
print(f"  总增幅: {(total_P5/total_P10-1)*100:.1f}%")
print(f"  *** 各工况平均电耗增幅 = {avg_dp:+.1f}% ***")
print()

pd.DataFrame([{'工况':i+1,'Cin':opt5[i]['Cin'],'P_10mg':opt10[i]['P'],'P_5mg':opt5[i]['P'],
    '增幅_pct':(opt5[i]['P']-opt10[i]['P'])/opt10[i]['P']*100,'Cout_5mg':opt5[i]['C']}
    for i in range(Kc)]).to_csv('strict_comparison.csv',index=False,encoding='utf-8-sig')

# ====== Pareto ======
print("--- 4.5 高浓度Pareto前沿 ---")
hi_i=np.argmax([opt5[i]['Cin'] for i in range(Kc)]); hi=opt5[hi_i]
print(f"  Cin = {hi['Cin']:.2f} g/Nm3")
cts=[3,4,5,6,7,8,9,10,12,15]; pwr=[]
for ct in cts:
    bp=1e10
    for _ in range(8000):
        u=np.random.uniform(45,72,4)
        c=pred_cout(hi['Cin'],u[0],u[1],u[2],u[3],0,0,0,0,hi['Q'])
        pt=power_est(u[0],u[1],u[2],u[3],hi['Q'],hi['Cin'])
        if c<=ct and pt<bp: bp=pt
    pwr.append(bp)
    print(f"  Cout<={ct:2d}: P_min={bp:.0f} kW")
p5v=pwr[cts.index(5)]; p10v=pwr[cts.index(10)]
print(f"  Pareto上 10->5mg: +{(p5v/p10v-1)*100:.1f}%")
print()

# ====== 建议 ======
print("--- 4.6 高浓度应对建议 ---")
print(f"""
  5mg最优参数: U=[{hi['U'][0]:.1f},{hi['U'][1]:.1f},{hi['U'][2]:.1f},{hi['U'][3]:.1f}]kV
               T=[{hi['T'][0]:.0f},{hi['T'][1]:.0f},{hi['T'][2]:.0f},{hi['T'][3]:.0f}]s
               P={hi['P']:.0f}kW

  [1] 电压升级: 所有电场需65-72kV, 可能需要高频电源改造
  [2] 振打优化: 缩短前级振打至120-200s保持极板清洁
  [3] 工艺协同: 源头降尘+烟气调质使温度接近160C
  [4] 设备改造: 加长电场/增加收尘面积/湿式电除尘
  [5] 经济评估: 平均电耗增幅{avg_dp:.0f}%, 需LCC分析
""")

# ====== 图 ======
fig,axes=plt.subplots(2,2,figsize=(14,10))
dv=[(opt5[i]['P']-opt10[i]['P'])/opt10[i]['P']*100 for i in range(Kc)]
ax=axes[0,0]
bars=ax.bar(range(1,Kc+1),dv,color='coral'); ax.axhline(avg_dp,color='red',ls='--',label=f'均值 {avg_dp:.1f}%')
ax.set_xlabel('工况'); ax.set_ylabel('电耗增幅 (%)'); ax.set_title('10mg->5mg 电耗增幅'); ax.legend()
for bar,val in zip(bars,dv): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,f'{val:.1f}%',ha='center',fontsize=8)
ax=axes[0,1]
x=np.arange(4); w=0.15
for i in range(min(Kc,6)):
    ax.bar(x+i*w-0.35,opt5[i]['U'],w,label=f'K{i+1}',color=[plt.cm.tab10(j/Kc) for j in range(Kc)][i],alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(['U1','U2','U3','U4']); ax.set_ylabel('kV'); ax.set_title('5mg最优电压'); ax.legend(fontsize=6)
ax=axes[1,0]
ax.plot(cts,pwr,'o-',color='steelblue',lw=2,markersize=8)
ax.axvline(10,color='orange',ls='--',lw=2,label='10mg'); ax.axvline(5,color='red',ls='--',lw=2,label='5mg')
ax.annotate(f'P={p10v:.0f}',xy=(10,p10v),xytext=(12,p10v+20),arrowprops=dict(arrowstyle='->',color='orange'),fontsize=8)
ax.annotate(f'P={p5v:.0f}\n+{(p5v/p10v-1)*100:.0f}%',xy=(5,p5v),xytext=(3,p5v+50),arrowprops=dict(arrowstyle='->',color='red'),fontsize=8,color='red')
ax.set_xlabel('C_out限值'); ax.set_ylabel('最小电耗 (kW)'); ax.set_title('Pareto前沿'); ax.legend(); ax.invert_xaxis()
ax=axes[1,1]
for i in range(min(Kc,6)):
    ax.bar(x+i*w-0.35,opt5[i]['T'],w,label=f'K{i+1}',color=[plt.cm.tab10(j/Kc) for j in range(Kc)][i],alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(['T1','T2','T3','T4']); ax.set_ylabel('s'); ax.set_title('5mg最优振打'); ax.legend(fontsize=6)
plt.suptitle('排放标准收紧 (10->5 mg/Nm3) 影响',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig('figures/problem4_stricter.png',dpi=150); plt.close()

print("图表已保存: figures/problem4_stricter.png")
print(f"\nK=[{K[0]:.1f},{K[1]:.1f},{K[2]:.1f},{K[3]:.1f}], Kc={Kc}, 平均dP={avg_dp:+.1f}%")
print("="*70)
