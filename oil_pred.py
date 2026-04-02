import streamlit as st
import akshare as ak  
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from streamlit.web import cli as stcli 

# --- 0. 自引导启动 ---
def bootstrap():
    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())

# --- 附加模块：静默获取美元指数 ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dxy_trend():
    try:
        df = ak.futures_foreign_hist(symbol="DX") 
        df.columns = [str(c).lower() for c in df.columns]
        if 'close' in df.columns:
            series = pd.to_numeric(df['close'], errors='coerce').dropna()
            if len(series) >= 5:
                return (series.iloc[-1] - series.iloc[-5]) / series.iloc[-5] 
    except: pass
    return 0.0

# --- 1. 核心数据引擎 ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data_v23(asset_type):
    symbol = "GC" if "金" in asset_type else "CL"
    try:
        df = ak.futures_foreign_hist(symbol=symbol)
        if df is not None and not df.empty:
            df.columns = [str(c).lower() for c in df.columns]
            if 'date' in df.columns and 'close' in df.columns:
                vol_col = [c for c in df.columns if 'vol' in c or 'volume' in c]
                v_col = vol_col[0] if vol_col else 'close' 
                
                df_clean = df[['date', 'close', v_col]].copy()
                df_clean.columns = ['Date', 'TARGET', 'Volume']
                df_clean['Date'] = pd.to_datetime(df_clean['Date'])
                df_clean['TARGET'] = pd.to_numeric(df_clean['TARGET'], errors='coerce')
                
                df_clean['Volume'] = pd.to_numeric(df_clean['Volume'], errors='coerce').fillna(0.0)
                if v_col == 'close': 
                    df_clean['Volume'] = 1.0 
                
                df_clean = df_clean.dropna().sort_values('Date').reset_index(drop=True)
                return df_clean
        return pd.DataFrame()
    except: return pd.DataFrame()

# --- 2. 深度量化引擎 ---
def execute_prediction(df, days, manual_score, panic_premium, backtest_days=0):
    if backtest_days > 0:
        df_train = df.iloc[:-backtest_days].tail(150).copy()
        df_truth = df.iloc[-backtest_days:].copy()
    else:
        df_train = df.tail(150).copy()
        df_truth = pd.DataFrame()

    df_train['Ordinal'] = df_train['Date'].map(datetime.toordinal)
    X = df_train[['Ordinal']].values
    y = df_train['TARGET'].values
    
    weights = df_train['Volume'].values
    if np.nansum(weights) <= 0:
        weights = np.ones_like(weights)
    else:
        weights = weights / (np.nanmean(weights) + 1e-9)
        weights = np.clip(weights, 0.01, None)
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y, sample_weight=weights)
    
    train_pred = model.predict(X_poly)
    rmse = np.sqrt(np.mean((y - train_pred)**2))
    mae = np.mean(np.abs(y - train_pred))
    
    dxy_change = fetch_dxy_trend()
    
    last_price = df_train['TARGET'].iloc[-1]
    ma60 = df_train['TARGET'].rolling(window=60).mean().iloc[-1]
    value_anchor = df_train['TARGET'].median() 
    recent_vol = df_train['TARGET'].tail(15).pct_change().std()
    is_gold = last_price > 500 

    last_date = df_train['Date'].max()
    pred_window = backtest_days if backtest_days > 0 else days
    f_dates = [last_date + timedelta(days=i) for i in range(1, pred_window + 1)]
    f_ordinals = np.array([d.toordinal() for d in f_dates]).reshape(-1, 1)
    
    base_path = model.predict(poly.transform(f_ordinals))
    t_path = []
    
    current_sim_price = last_price
    np.random.seed(42) 
    
    for i, p in enumerate(base_path):
        step_ratio = (i + 1) / pred_window
        decay = 1 / (1 + np.exp(0.5 * (i - pred_window/2))) 
        tactical_impact = 1 + (manual_score / 10.0 * 0.15 * (1 - decay))
        
        panic_decay = np.log1p(i + 1) / np.log1p(pred_window)
        panic_impact = 1 + (panic_premium / 100.0 * panic_decay)
        gravity_pull = (value_anchor - p) / p * 0.3 * step_ratio
        
        dxy_impact = -1 * dxy_change * 0.5 * step_ratio
        
        m = f_dates[i].month
        season_impact = 0.02 * step_ratio if is_gold and m in [1,2,8,9,12] else (0.03 * step_ratio if not is_gold and m in [5,6,7,8] else 0)
            
        noise = 1 + np.random.normal(0, recent_vol * 0.5)
        final_p = p * tactical_impact * panic_impact * (1 + gravity_pull + season_impact + dxy_impact) * noise
        
        hard_floor = df_train['TARGET'].min() * 0.95
        if final_p < hard_floor:
            final_p = hard_floor + (np.random.random() * 20) 
            
        t_path.append(final_p)
        current_sim_price = final_p
        
    return df_train, df_truth, f_dates, t_path, rmse, mae

# --- 3. UI 渲染 ---
def run_app():
    st.set_page_config(page_title="Oil-Pred v3.2 | 未来云图版", layout="wide")
    
    with st.sidebar:
        st.header("🎯 目标控制")
        mode = st.radio("监控品种:", ["📀 国际黄金 (COMEX)", "🛢️ WTI 原油 (NYMEX)"])
        st.divider()
        st.header("🕰️ 时光机 (测试模式)")
        enable_backtest = st.toggle("开启样本外回测")
        backtest_days = st.slider("倒退天数", 5, 60, 20) if enable_backtest else 0
        p_window = 0 if enable_backtest else st.slider("前瞻预测天数", 1, 30, 7)
        st.divider()
        st.header("🕹️ 指挥官决策")
        m_score = st.slider("常规战术修正", -10.0, 10.0, 0.0, step=0.5)
        panic_val = st.slider("🌋 黑天鹅恐慌溢价 (%)", -100, 100, 0, step=5)

    df_main = fetch_data_v23(mode)
    if not df_main.empty:
        df_main['MA5'] = df_main['TARGET'].rolling(window=5).mean()
        df_main['MA20'] = df_main['TARGET'].rolling(window=20).mean()
        df_main['STD20'] = df_main['TARGET'].rolling(window=20).std()
        df_main['BB_UP'] = df_main['MA20'] + 2 * df_main['STD20']
        df_main['BB_DN'] = df_main['MA20'] - 2 * df_main['STD20']

        df_train, df_truth, f_dates, t_path, rmse, mae = execute_prediction(df_main, p_window, m_score, panic_val, backtest_days)
        
        # 🟢 计算未来的动态布林带云团
        hist_tail = df_train['TARGET'].tail(19).tolist()
        sim_prices = hist_tail + t_path
        sim_series = pd.Series(sim_prices)
        sim_ma20 = sim_series.rolling(20).mean().dropna().tolist()
        sim_std20 = sim_series.rolling(20).std().dropna().tolist()
        sim_bb_up = [m + 2*s for m, s in zip(sim_ma20, sim_std20)]
        sim_bb_dn = [m - 2*s for m, s in zip(sim_ma20, sim_std20)]

        c1, c2, c3, c4 = st.columns(4)
        if enable_backtest:
            actual_end_p = df_truth['TARGET'].iloc[-1]
            pred_end_p = t_path[-1]
            err = (pred_end_p - actual_end_p) / actual_end_p * 100
            c1.metric("现实真相价", f"${actual_end_p:.2f}")
            c2.metric("预测终点价", f"${pred_end_p:.2f}", f"误差 {err:+.2f}%")
        else:
            cur_p = df_train['TARGET'].iloc[-1]
            c1.metric("今日现价", f"${cur_p:.2f}")
            c2.metric("前瞻预测价", f"${t_path[-1]:.2f}", f"{(t_path[-1]-cur_p):+.2f}")
        
        c3.metric("拟合方差 (RMSE)", f"{rmse:.2f}")
        c4.metric("绝对误差 (MAE)", f"{mae:.2f}")

        fig = go.Figure()
        df_show = df_train.tail(100)
        
        # 1. 绘制历史紫色云团
        fig.add_trace(go.Scatter(x=df_show['Date'], y=df_show['BB_UP'], line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=df_show['Date'], y=df_show['BB_DN'], fill='tonexty', fillcolor='rgba(155, 89, 182, 0.15)', line=dict(width=0), name="历史置信带(2σ)"))

        # 2. 🟢 绘制未来红色云团
        f_px = [df_show['Date'].iloc[-1]] + f_dates
        f_bb_up = [df_show['BB_UP'].iloc[-1]] + sim_bb_up
        f_bb_dn = [df_show['BB_DN'].iloc[-1]] + sim_bb_dn
        
        fig.add_trace(go.Scatter(x=f_px, y=f_bb_up, line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=f_px, y=f_bb_dn, fill='tonexty', fillcolor='rgba(255, 75, 75, 0.15)', line=dict(width=0), name="预测推演云团(2σ)"))

        # 3. 绘制实体线
        fig.add_trace(go.Scatter(x=df_show['Date'], y=df_show['TARGET'], name="收盘价", line=dict(color='#4A90E2', width=2)))
        fig.add_trace(go.Scatter(x=df_show['Date'], y=df_show['MA5'], name="MA5", line=dict(color='#E2A14A', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=df_show['Date'], y=df_show['MA20'], name="MA20", line=dict(color='#9B59B6', width=1.5)))

        if enable_backtest and not df_truth.empty:
            fig.add_trace(go.Scatter(x=[df_show['Date'].iloc[-1]] + list(df_truth['Date']), 
                                     y=[df_show['TARGET'].iloc[-1]] + list(df_truth['TARGET']), 
                                     name="现实真相", line=dict(color='#00FA9A', width=3)))

        fig.add_trace(go.Scatter(x=f_px, y=[df_show['TARGET'].iloc[-1]] + list(t_path), 
                                 name="量化推演路径", line=dict(color='#ff4b4b', width=3, dash='dash')))

        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,t=20,b=0), dragmode='pan', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

if __name__ == "__main__":
    bootstrap()
    run_app()