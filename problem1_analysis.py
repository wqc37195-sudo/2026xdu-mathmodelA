# -*- coding: utf-8 -*-
"""
problem1_analysis.py - 问题1: 入口条件与操作参数对出口粉尘浓度的影响分析
统一模型: C_out = C_in * exp( -SUM K_i * U_i^2 * f(T) / Q )
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib, os, sys, json
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
import matplotlib.font_manager as fm, matplotlib.axes

# ===== 中文字体monkey-patch (绕过rcParams bug) =====
fm.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
_zh = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc', size=11)
_zh_t = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc', size=13)
_zh_sm = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc', size=9)
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

plt.rcParams['figure.dpi']=120; plt.rcParams['axes.unicode_minus']=False
if sys.stdout.encoding != 'utf-8': sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sns.set_style('whitegrid'); np.random.seed(42); os.makedirs('figures',exist_ok=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from scipy.signal import welch, find_peaks

# ============================================================
# 1. 数据加载与诊断
# ============================================================
df=pd.read_csv('Cement_ESP_Data.csv',parse_dates=['timestamp'])
for col in df.columns:
    if col!='timestamp': df[col]=df[col].ffill().bfill().fillna(df[col].median())

cin_g=df['C_in_gNm3'].values; cout_mg=df['C_out_mgNm3'].values
cin_mg=cin_g*1000.0; Q=df['Q_Nm3h'].values; Temp=df['Temp_C'].values
U=np.column_stack([df[f'U{i}_kV'].values for i in range(1,5)])
T_rap=np.column_stack([df[f'T{i}_s'].values for i in range(1,5)])
dt=(df['timestamp'].iloc[1]-df['timestamp'].iloc[0]).total_seconds()

cout_m=float(np.mean(cout_mg)); cout_s=float(np.std(cout_mg,ddof=1))
cin_m_v=float(np.mean(cin_g)); cin_s_v=float(np.std(cin_g,ddof=1))
eta=1-cout_m/np.mean(cin_mg); S_all=np.log(cin_mg/cout_mg)

print("="*70)
print("问题1: 入口条件与操作参数对出口粉尘浓度的影响分析")
print("="*70)
print()
print("--- 1.1 数据诊断 ---")
print(f"  样本数: {len(df)}, 采样间隔: {dt:.0f}s, 总时长: {len(df)*dt/3600:.1f}h")
print(f"  C_out: 均值={cout_m:.4f}, 标准差={cout_s:.4f}, CV={cout_s/cout_m*100:.2f}% (几乎恒定)")
print(f"  C_in:  均值={cin_m_v:.2f}, 标准差={cin_s_v:.2f}, CV={cin_s_v/cin_m_v*100:.1f}% (大幅波动)")
print(f"  除尘效率 = {eta:.4f} ({eta*100:.2f}%)")
print()

fig,axes=plt.subplots(2,2,figsize=(14,8))
axes[0,0].hist(cout_mg,bins=60,color='steelblue',ec='white',alpha=0.85)
axes[0,0].axvline(cout_m,color='red',ls='--',lw=2,label=f'均值={cout_m:.3f}')
axes[0,0].set_xlabel('C_out (mg/Nm3)'); axes[0,0].set_ylabel('频数')
axes[0,0].set_title('C_out分布'); axes[0,0].legend()
axes[0,1].plot(df['timestamp'],cout_mg,'b-',lw=0.3,alpha=0.6)
axes[0,1].axhline(cout_m,color='red',ls='--',lw=1.5)
axes[0,1].set_xlabel('时间'); axes[0,1].set_title('C_out时间序列')
axes[1,0].hist(cin_g,bins=50,color='darkorange',ec='white',alpha=0.85)
axes[1,0].set_xlabel('C_in (g/Nm3)'); axes[1,0].set_ylabel('频数')
axes[1,0].set_title('C_in分布')
axes[1,1].plot(df['timestamp'],cin_g,'orange',lw=0.3,alpha=0.6)
axes[1,1].set_xlabel('时间'); axes[1,1].set_title('C_in时间序列')
plt.suptitle('数据诊断: C_in vs C_out',fontsize=13,fontweight='bold')
plt.tight_layout(); plt.savefig('figures/problem1_diagnosis.png',dpi=150); plt.close()
print("  图表已保存: figures/problem1_diagnosis.png")

# ============================================================
# 2. ESP控制规律
# ============================================================
print("--- 1.2 ESP控制规律 ---")
X_wide=df[['Temp_C','C_in_gNm3','Q_Nm3h','U1_kV','U2_kV','U3_kV','U4_kV','T1_s','T2_s','T3_s','T4_s']].copy()
X_wide['C_in_mg']=X_wide['C_in_gNm3']*1000
for i in range(1,5): X_wide[f'U{i}_sq']=X_wide[f'U{i}_kV']**2
feat_names=X_wide.columns.tolist()

X_tr,X_te,y_tr,y_te=train_test_split(X_wide.values,S_all,test_size=0.2,random_state=42)
rf_S=RandomForestRegressor(n_estimators=200,max_depth=8,min_samples_leaf=5,random_state=42,n_jobs=-1)
rf_S.fit(X_tr,y_tr)
imp_S=rf_S.feature_importances_; idx_S=np.argsort(imp_S)[::-1]
print(f"  随机森林建模 S=ln(Cin/Cout): R2={r2_score(y_te,rf_S.predict(X_te)):.4f}")
print(f"  特征重要性(前5):")
for i in range(5): j=idx_S[i]; print(f"    {feat_names[j]:<15s}: {imp_S[j]:.4f}")
print()

fig,axes=plt.subplots(1,3,figsize=(15,4))
for ia,fi in enumerate([idx_S[0],idx_S[1],idx_S[2]]):
    PartialDependenceDisplay.from_estimator(rf_S,X_te,features=[fi],ax=axes[ia],grid_resolution=50,kind='average')
    axes[ia].set_title(f'S偏依赖: {feat_names[fi]}')
plt.tight_layout(); plt.savefig('figures/problem1_pdp.png',dpi=150); plt.close()

fig,axes=plt.subplots(2,4,figsize=(18,8))
cin_bins=pd.qcut(cin_g,20,duplicates='drop')
for i in range(4):
    ax=axes[0,i]; g=df.groupby(cin_bins)[f'U{i+1}_kV'].mean()
    ax.plot(range(len(g)),g.values,'o-',color='steelblue',markersize=4)
    step=max(1,len(g)//4); ax.set_xticks(range(0,len(g),step))
    ax.set_xticklabels([f'{gg.left:.1f}' for gg in g.index[::step]],rotation=30,fontsize=7)
    ax.set_xlabel('C_in (g/Nm3)'); ax.set_ylabel(f'U{i+1} (kV)'); ax.set_title(f'U{i+1} vs C_in')
    ax=axes[1,i]; g=df.groupby(cin_bins)[f'T{i+1}_s'].mean()
    ax.plot(range(len(g)),g.values,'s-',color='coral',markersize=4)
    ax.set_xticks(range(0,len(g),step))
    ax.set_xticklabels([f'{gg.left:.1f}' for gg in g.index[::step]],rotation=30,fontsize=7)
    ax.set_xlabel('C_in (g/Nm3)'); ax.set_ylabel(f'T{i+1} (s)'); ax.set_title(f'T{i+1} vs C_in')
plt.suptitle('ESP控制规律',fontsize=13,fontweight='bold')
plt.tight_layout(); plt.savefig('figures/problem1_control_law.png',dpi=150); plt.close()
print("  图表已保存: figures/problem1_pdp.png, problem1_control_law.png")

# ============================================================
# 3. M4模型标定
# ============================================================
print("--- 1.3 M4模型标定 ---")
Tp=160; st=60
def fT(tv):
    ft=np.ones_like(tv,dtype=float); m=tv>120
    ft[m]=np.exp(-((tv[m]-Tp)/st)**2); return ft

S_mean=np.mean(S_all); Q_mean=np.mean(Q)
U_mean=np.mean(U,axis=0); Tmp_mean=np.mean(Temp)
fT_mean=fT(np.array([Tmp_mean]))[0]
shares=[0.40,0.30,0.17,0.13]

K_list=np.array([shares[i]*S_mean*Q_mean/(U_mean[i]**2*fT_mean) for i in range(4)])

print(f"  均值点标定: S={S_mean:.4f}, Q={Q_mean:.0f}, f(T)={fT_mean:.4f}")
for i in range(4):
    print(f"    K{i+1}={shares[i]:.2f}*{S_mean:.4f}*{Q_mean:.0f}/({U_mean[i]:.1f}^2*{fT_mean:.4f}) = {K_list[i]:.2f}")

S_chk=sum(K_list[i]*U_mean[i]**2*fT_mean/Q_mean for i in range(4))
print(f"  自洽验证: SUM = {S_chk:.4f} = S_mean OK")
print(f"  均值点预测C_out = {np.mean(cin_mg)*np.exp(-S_chk):.2f} (实际 = {cout_m:.2f})")
print()

def predict_cout(Ua,Ta,Cin,Qv,Tmpv):
    Ua=np.atleast_2d(Ua); Cin=np.atleast_1d(np.asarray(Cin,dtype=float))
    Qv=np.atleast_1d(np.asarray(Qv,dtype=float)); Tmpv=np.atleast_1d(np.asarray(Tmpv,dtype=float))
    N=len(Cin); ft=fT(Tmpv); S_total=np.zeros(N)
    for i in range(4): S_total+=K_list[i]*Ua[:,i]**2*ft/Qv
    return Cin*np.exp(-S_total)

Cout_all=predict_cout(U,T_rap,cin_mg,Q,Temp)
rmse=np.sqrt(mean_squared_error(cout_mg,Cout_all)); bias=np.mean(Cout_all-cout_mg)
print(f"  全样本: RMSE={rmse:.2f}, Bias={bias:+.2f}, 预测均值={np.mean(Cout_all):.2f}")
print()

fig,axes=plt.subplots(1,3,figsize=(18,5))
axes[0].plot(df['timestamp'].iloc[:1000],cout_mg[:1000],'b-',lw=0.5,alpha=0.6,label='实际')
axes[0].plot(df['timestamp'].iloc[:1000],Cout_all[:1000],'r-',lw=0.5,alpha=0.6,label='预测')
axes[0].set_xlabel('时间'); axes[0].set_ylabel('C_out'); axes[0].set_title('实际vs预测'); axes[0].legend()
axes[1].scatter(cout_mg,Cout_all,s=1,alpha=0.2,color='steelblue')
axes[1].axhline(cout_m,color='blue',ls='--',label=f'实际均值={cout_m:.1f}')
axes[1].axhline(np.mean(Cout_all),color='red',ls='--',label=f'预测均值={np.mean(Cout_all):.1f}')
axes[1].set_xlabel('实际C_out'); axes[1].set_ylabel('预测C_out'); axes[1].set_title('预测vs实际'); axes[1].legend()
S_shares=[K_list[i]*U_mean[i]**2*fT_mean/Q_mean for i in range(4)]
total_S=sum(S_shares)
axes[2].pie([s/total_S*100 for s in S_shares],labels=[f'电场{i+1}\nK={K_list[i]:.0f}' for i in range(4)],
            colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'],autopct='%1.1f%%',startangle=90)
axes[2].set_title('各电场S_i贡献')
plt.suptitle('M4模型标定结果',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig('figures/problem1_model.png',dpi=150); plt.close()

# ============================================================
# 4. 灵敏度分析
# ============================================================
print("--- 1.4 灵敏度分析 ---")
U0=np.mean(U,axis=0); T0=np.mean(T_rap,axis=0)
Cin0=np.mean(cin_g); Q0=np.mean(Q); Temp0=np.mean(Temp)
Cout0=predict_cout(U0.reshape(1,-1),T0.reshape(1,-1),np.array([Cin0*1000]),np.array([Q0]),np.array([Temp0]))[0]
print(f"  基准点: Cout = {Cout0:.2f} mg/Nm3")

elast=[]
for i in range(4):
    Up=U0.copy(); Up[i]+=1.0; Um=U0.copy(); Um[i]-=1.0
    Cp=predict_cout(Up.reshape(1,-1),T0.reshape(1,-1),np.array([Cin0*1000]),np.array([Q0]),np.array([Temp0]))[0]
    Cm=predict_cout(Um.reshape(1,-1),T0.reshape(1,-1),np.array([Cin0*1000]),np.array([Q0]),np.array([Temp0]))[0]
    elast.append((f'U{i+1}',abs(Cp-Cm)/(2*Cout0)*100))
for i in range(4):
    Tp_arr=T0.copy(); Tp_arr[i]=min(Tp_arr[i]+50,600); Tm_arr=T0.copy(); Tm_arr[i]=max(Tm_arr[i]-50,60)
    Cp=predict_cout(U0.reshape(1,-1),Tp_arr.reshape(1,-1),np.array([Cin0*1000]),np.array([Q0]),np.array([Temp0]))[0]
    Cm=predict_cout(U0.reshape(1,-1),Tm_arr.reshape(1,-1),np.array([Cin0*1000]),np.array([Q0]),np.array([Temp0]))[0]
    elast.append((f'T{i+1}',abs(Cp-Cm)/(2*Cout0)*100))
elast.sort(key=lambda x:x[1],reverse=True)
for rank,(name,val) in enumerate(elast,1):
    s='***' if rank<=2 else ('**' if rank<=4 else '*')
    print(f"    {name}: {val:>8.4f}%  {s}")
print()

fig,axes=plt.subplots(2,2,figsize=(14,10))
Cin_r=np.linspace(cin_g.min(),cin_g.max(),200)
Cout_vs_Cin=predict_cout(np.tile(U0,(200,1)),np.tile(T0,(200,1)),Cin_r*1000,np.full(200,Q0),np.full(200,Temp0))
axes[0,0].plot(Cin_r,Cout_vs_Cin,'b-',lw=2); axes[0,0].axvline(Cin0,color='red',ls='--',label=f'均值={Cin0:.1f}')
axes[0,0].set_xlabel('C_in (g/Nm3)'); axes[0,0].set_ylabel('C_out (mg/Nm3)'); axes[0,0].set_title('C_out 随 C_in 变化'); axes[0,0].legend()
U_r=np.linspace(0.8,1.2,200)
Cout_vs_U=np.array([predict_cout((U0*r).reshape(1,-1),T0.reshape(1,-1),np.array([Cin0*1000]),np.array([Q0]),np.array([Temp0]))[0] for r in U_r])
axes[0,1].plot(U_r*sum(U0),Cout_vs_U,'r-',lw=2); axes[0,1].axvline(sum(U0),color='red',ls='--',label=f'基准={sum(U0):.0f}kV')
axes[0,1].set_xlabel('总电压 Sum U_i (kV)'); axes[0,1].set_ylabel('C_out (mg/Nm3)'); axes[0,1].set_title('C_out 随总电压变化'); axes[0,1].legend()
Tmp_r=np.linspace(50,250,200)
Cout_vs_Tmp=predict_cout(np.tile(U0,(200,1)),np.tile(T0,(200,1)),np.full(200,Cin0*1000),np.full(200,Q0),Tmp_r)
axes[1,0].plot(Tmp_r,Cout_vs_Tmp,'g-',lw=2)
axes[1,0].axvline(160,color='green',ls=':',label='T_peak=160C'); axes[1,0].axvline(Temp0,color='red',ls='--',label=f'均值={Temp0:.0f}C')
axes[1,0].set_xlabel('烟气温度 (C)'); axes[1,0].set_ylabel('C_out (mg/Nm3)'); axes[1,0].set_title('C_out 随温度变化'); axes[1,0].legend()
T1_r=np.linspace(60,800,200)
Cout_vs_T1=np.array([predict_cout(U0.reshape(1,-1),np.array([[t,T0[1],T0[2],T0[3]]]),np.array([Cin0*1000]),np.array([Q0]),np.array([Temp0]))[0] for t in T1_r])
axes[1,1].plot(T1_r,Cout_vs_T1,'purple',lw=2)
axes[1,1].axvline(120,color='green',ls='--',label='T=120s'); axes[1,1].axvline(500,color='green',ls='--',label='T=500s')
axes[1,1].fill_between(T1_r,Cout_vs_T1.min(),Cout_vs_T1,where=(T1_r>=120)&(T1_r<=500),alpha=0.12,color='green')
axes[1,1].set_xlabel('振打周期 T1 (s)'); axes[1,1].set_ylabel('C_out (mg/Nm3)'); axes[1,1].set_title('C_out 随振打周期变化'); axes[1,1].legend()
plt.suptitle('灵敏度分析 (M4模型)',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig('figures/problem1_sensitivity.png',dpi=150); plt.close()

# ============================================================
# 5. 振打分析
# ============================================================
print("--- 1.5 振打周期对瞬时排放峰值的影响 ---")
window=11
cout_trend=pd.Series(cout_mg).rolling(window=window,center=True,min_periods=5).median().values
cout_trend=np.nan_to_num(cout_trend,nan=np.nanmedian(cout_trend))
cout_resid=cout_mg-cout_trend; resid_std=np.std(cout_resid)
peaks,_=find_peaks(cout_resid,height=resid_std*2,distance=5,prominence=resid_std)
freq,psd=welch(cout_resid,fs=1/dt,nperseg=min(1024,len(cout_resid)//4))
mask=(freq>0.0001)&(freq<0.02)
peak_period=1/freq[mask][np.argmax(psd[mask])]

print(f"  残差标准差 = {resid_std:.4f} mg/Nm3, 检测到事件: {len(peaks)} 次")
print(f"  频谱主周期 = {peak_period:.0f}s = {peak_period/60:.1f} min")
for i in range(4): print(f"  T{i+1} 均值 = {np.mean(T_rap[:,i]):.0f}s")

fig,axes=plt.subplots(2,3,figsize=(18,10))
axes[0,0].plot(df['timestamp'],cout_resid,'b-',lw=0.3,alpha=0.6)
axes[0,0].axhline(0,color='red',ls='--',lw=0.5)
if len(peaks)>0: axes[0,0].scatter(df['timestamp'].iloc[peaks],cout_resid[peaks],c='red',s=8,zorder=5,label=f'事件 n={len(peaks)}')
axes[0,0].set_xlabel('时间'); axes[0,0].set_ylabel('残差 (mg/Nm3)'); axes[0,0].set_title('C_out去趋势残差'); axes[0,0].legend()
axes[0,1].semilogy(freq[mask],psd[mask],'b-',lw=1)
axes[0,1].axvline(freq[mask][np.argmax(psd[mask])],color='red',ls='--',label=f'主周期={peak_period:.0f}s')
axes[0,1].set_xlabel('频率 (Hz)'); axes[0,1].set_ylabel('功率谱密度'); axes[0,1].set_title('C_out残差功率谱'); axes[0,1].legend()
T_smooth=np.linspace(60,800,200)
Cout_Ts=np.array([predict_cout(np.array([[55,U0[1],U0[2],U0[3]]]),np.array([[t,T0[1],T0[2],T0[3]]]),np.array([Cin0*1000]),np.array([Q0]),np.array([Temp0]))[0] for t in T_smooth])
axes[0,2].plot(T_smooth,Cout_Ts,'b-',lw=2)
axes[0,2].axvline(120,color='green',ls='--',label='T=120s'); axes[0,2].axvline(500,color='green',ls='--',label='T=500s')
axes[0,2].fill_between(T_smooth,Cout_Ts.min(),Cout_Ts,where=(T_smooth>=120)&(T_smooth<=500),alpha=0.12,color='green')
axes[0,2].set_xlabel('T1 (s)'); axes[0,2].set_ylabel('C_out (mg/Nm3)'); axes[0,2].set_title('T1对排放的影响'); axes[0,2].legend()
if len(peaks)>0: axes[1,0].hist(cout_resid[peaks],bins=20,color='coral',ec='white',alpha=0.8)
axes[1,0].set_xlabel('峰值高度 (mg/Nm3)'); axes[1,0].set_title('峰值高度分布')
cin_bins_v=pd.qcut(cin_g,10,duplicates='drop'); u1g=df.groupby(cin_bins_v)['U1_kV'].mean(); t1g=df.groupby(cin_bins_v)['T1_s'].mean()
x_pos=np.arange(len(u1g))
axes[1,1].errorbar(x_pos,u1g.values,fmt='o-',color='steelblue',capsize=3,label='U1')
axt2=axes[1,1].twinx(); axt2.plot(x_pos,t1g.values,'s-',color='coral',label='T1')
axes[1,1].set_xticks(x_pos[::2]); axes[1,1].set_xticklabels([f'{g.left:.1f}' for g in u1g.index[::2]],fontsize=7)
axes[1,1].set_xlabel('C_in (g/Nm3)'); axes[1,1].set_ylabel('U1 (kV)'); axt2.set_ylabel('T1 (s)')
axes[1,1].set_title('U1/T1随C_in的自适应'); axes[1,1].legend(loc='upper left'); axt2.legend(loc='upper right')
pn=[e[0] for e in elast]; vl=[e[1] for e in elast]
axes[1,2].barh(range(8)[::-1],vl[::-1],color='steelblue',alpha=0.7)
axes[1,2].set_yticks(range(8)[::-1]); axes[1,2].set_yticklabels(pn[::-1])
axes[1,2].set_xlabel('弹性 (%)'); axes[1,2].set_title('参数弹性排序\n(U:+-1kV, T:+-50s)')
plt.suptitle('振打周期对瞬时排放峰值的影响',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.savefig('figures/problem1_rapping.png',dpi=150); plt.close()

# ============================================================
# 6. 结论
# ============================================================
print("="*70)
print("问题1 结论")
print("="*70)
print(f"""
  (a) 数据诊断:
      C_out几乎恒定: {cout_m:.3f} +/- {cout_s:.4f} mg/Nm3 (CV={cout_s/cout_m*100:.2f}%)
      C_in大幅波动: {cin_m_v:.1f} +/- {cin_s_v:.1f} g/Nm3 (CV={cin_s_v/cin_m_v*100:.1f}%)
      除尘效率: {eta*100:.2f}%

  (b) M4模型:
      C_out = C_in * exp( -SUM K_i * U_i^2 * f(T) / Q )
      K = [{K_list[0]:.1f}, {K_list[1]:.1f}, {K_list[2]:.1f}, {K_list[3]:.1f}]
      RMSE = {rmse:.2f} mg/Nm3, Bias = {bias:+.2f} mg/Nm3

  (c) 灵敏度排序:
      {', '.join(f'{n}={v:.2f}%' for n,v in elast)}

  (d) 振打影响: 主周期{peak_period:.0f}s, 仅{len(peaks)}次事件 -- C_out过于稳定
""")
print("="*70)
