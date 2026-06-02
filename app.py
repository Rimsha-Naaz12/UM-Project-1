"""
STREAMLIT DASHBOARD — UAC Predictive Forecasting
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, os, warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="UAC Predictive Forecasting",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLES ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-box {
    background: #f0f4ff; border-left: 4px solid #3b6fd4;
    padding: 12px 16px; border-radius: 6px; margin-bottom: 8px;
}
.metric-box h4 { margin: 0; color: #1a2d6b; font-size: 13px; }
.metric-box h2 { margin: 4px 0 0 0; color: #3b6fd4; font-size: 24px; }
.warning-box { background: #fff3cd; border-left: 4px solid #e67e22;
    padding: 10px 16px; border-radius: 6px; color: #7d4e00; }
.danger-box  { background: #fdecea; border-left: 4px solid #e74c3c;
    padding: 10px 16px; border-radius: 6px; color: #7b0000; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists("data/uac_cleaned.csv"):
        st.error("Run step1_eda.py first to generate the cleaned dataset.")
        st.stop()
    return pd.read_csv("data/uac_cleaned.csv", index_col='Date', parse_dates=True)

@st.cache_resource
def load_models():
    models = {}
    for tag, name in [("rf", "Random Forest"), ("gb", "Gradient Boosting")]:
        path = f"models/{tag}_model.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models

df = load_data()
ml_models = load_models()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.image("https://www.hhs.gov/sites/default/files/hhs-logo-color.png", width=160)
st.sidebar.title("⚙️ Controls")
horizon     = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)
model_type  = st.sidebar.selectbox("Statistical Model", ["Exponential Smoothing", "ARIMA"])
show_ci     = st.sidebar.checkbox("Show Confidence Interval", value=True)
show_ml     = st.sidebar.checkbox("Show ML Overlay", value=False)
date_filter = st.sidebar.select_slider(
    "Historical view (months back)", options=[3, 6, 12, 24, 36, 60], value=12)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.title("🏥 UAC Program — Predictive Care Forecasting")
st.caption("U.S. Department of Health and Human Services | Unaccompanied Alien Children Program")
st.markdown("---")

# ─── KPI ROW ─────────────────────────────────────────────────────────────────
target = df['hhs_care'].dropna()
p90 = target.quantile(0.90)
p75 = target.quantile(0.75)
recent = target.iloc[-1]
breach_prob = (target.iloc[-30:] > p90).mean() * 100
net_flow    = (df['cbp_transfers'] - df['hhs_discharged']).iloc[-7:].mean()
los_proxy   = (df['hhs_care'] / df['hhs_discharged'].replace(0, np.nan)).iloc[-30:].mean()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Children in HHS Care", f"{int(recent):,}", f"{int(recent - target.iloc[-7])}")
col2.metric("30-Day Breach Prob", f"{breach_prob:.0f}%", delta_color="inverse")
col3.metric("7-Day Net Pressure", f"{net_flow:+.0f}/day", delta_color="inverse")
col4.metric("LOS Proxy (days)", f"{los_proxy:.1f}")
col5.metric("Max Ever Recorded", f"{int(target.max()):,}")

# Alert banner
if breach_prob > 60:
    st.markdown(f'<div class="danger-box">🚨 <b>HIGH RISK:</b> Capacity breach probability is {breach_prob:.0f}% — surge preparation recommended.</div>', unsafe_allow_html=True)
elif breach_prob > 30:
    st.markdown(f'<div class="warning-box">⚠️ <b>ELEVATED:</b> Breach probability at {breach_prob:.0f}% — monitor closely.</div>', unsafe_allow_html=True)

st.markdown("---")

# ─── TAB LAYOUT ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📊 Historical EDA", "⚠️ Early Warning", "📋 Model Comparison"])

# ── TAB 1: FORECAST ──────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"Children in HHS Care — {horizon}-Day Forecast")

    # Run selected model
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA

    @st.cache_data
    def run_forecast(model_type, horizon):
        if model_type == "Exponential Smoothing":
            m = ExponentialSmoothing(target, trend='add', seasonal='add', seasonal_periods=7).fit()
            fc = m.forecast(horizon)
            resid_std = (target - m.fittedvalues).std()
        else:
            m = ARIMA(target, order=(5, 1, 2)).fit()
            fc = m.forecast(horizon)
            resid_std = m.resid.std()
        future_idx = pd.date_range(target.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
        fc.index = future_idx
        return fc, resid_std

    fc, resid_std = run_forecast(model_type, horizon)
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=date_filter)
    hist = target[target.index >= cutoff]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist.index, hist.values, color='steelblue', linewidth=1.5, label='Historical')
    ax.plot(fc.index, fc.values, color='tomato', linewidth=2.5, linestyle='--', label=f'{horizon}-Day Forecast')
    if show_ci:
        ax.fill_between(fc.index,
                        fc.values - 1.96 * resid_std,
                        fc.values + 1.96 * resid_std,
                        alpha=0.15, color='tomato', label='95% CI')
    ax.axvline(target.index[-1], color='gray', linestyle=':', linewidth=1.2, label='Today')
    ax.set_xlabel("Date"); ax.set_ylabel("Children in HHS Care")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Forecast table
    fc_df = pd.DataFrame({'Date': fc.index.date, 'Forecast': fc.values.round(0).astype(int),
                          'Lower_95': (fc.values - 1.96*resid_std).round(0).astype(int),
                          'Upper_95': (fc.values + 1.96*resid_std).round(0).astype(int)})
    with st.expander("📋 View Forecast Table"):
        st.dataframe(fc_df, use_container_width=True)
        csv = fc_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Forecast CSV", csv, "hhs_forecast.csv", "text/csv")

# ── TAB 2: HISTORICAL EDA ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Historical Trends")
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=date_filter)
    hist_df = df[df.index >= cutoff]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0,0].plot(hist_df.index, hist_df['hhs_care'], color='steelblue')
    axes[0,0].set_title('HHS Care Load'); axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(hist_df.index, hist_df['cbp_apprehended'].rolling(7).mean(), color='tomato')
    axes[0,1].set_title('CBP Apprehensions (7-day MA)'); axes[0,1].grid(alpha=0.3)

    axes[1,0].plot(hist_df.index, hist_df['cbp_transfers'].rolling(7).mean(), label='Transfers', color='orange')
    axes[1,0].plot(hist_df.index, hist_df['hhs_discharged'].rolling(7).mean(), label='Discharges', color='green')
    axes[1,0].legend(); axes[1,0].set_title('Flow: Transfers vs Discharges'); axes[1,0].grid(alpha=0.3)

    net = hist_df['net_pressure'].rolling(7).mean()
    axes[1,1].bar(hist_df.index, net, color=['tomato' if v > 0 else 'seagreen' for v in net], width=1)
    axes[1,1].axhline(0, color='black', linewidth=0.8)
    axes[1,1].set_title('Net Pressure (7-day MA)'); axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 3: EARLY WARNING ──────────────────────────────────────────────────────
with tab3:
    st.subheader("⚠️ Early Warning System")

    net_roll = df['net_pressure'].rolling(7).mean()
    z = (net_roll - net_roll.rolling(90).mean()) / net_roll.rolling(90).std()
    df['surge_z'] = z
    breach_roll = (df['hhs_care'] > p90).rolling(30).mean() * 100
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=date_filter)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    d = df[df.index >= cutoff]
    axes[0].plot(d.index, d['hhs_care'], color='steelblue', linewidth=1.2)
    axes[0].axhline(p90, color='red', linestyle='--', linewidth=1.5, label=f'90th pct = {p90:.0f}')
    axes[0].axhline(p75, color='orange', linestyle='--', linewidth=1, label=f'75th pct = {p75:.0f}')
    axes[0].fill_between(d.index, d['hhs_care'], p90, where=(d['hhs_care']>p90), color='red', alpha=0.2)
    axes[0].set_title('HHS Care with Capacity Thresholds'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(d.index, d['surge_z'], color='purple', linewidth=1.2)
    axes[1].axhline(2, color='red', linestyle='--', linewidth=1.2, label='Surge (z>2)')
    axes[1].axhline(-2, color='green', linestyle='--', linewidth=1.2, label='Relief (z<-2)')
    axes[1].fill_between(d.index, d['surge_z'], 2, where=(d['surge_z']>2), color='red', alpha=0.3)
    axes[1].set_title('Surge Pressure Z-Score'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Count recent surges
    recent_surges = int((df['surge_z'].iloc[-90:] > 2.0).sum())
    if recent_surges > 0:
        st.markdown(f'<div class="warning-box">📍 {recent_surges} surge-level days detected in last 90 days</div>', unsafe_allow_html=True)

# ── TAB 4: MODEL COMPARISON ───────────────────────────────────────────────────
with tab4:
    st.subheader("📋 Model Performance Comparison")
    if os.path.exists("outputs/model_comparison.csv"):
        comp = pd.read_csv("outputs/model_comparison.csv")
        best_mae = comp.loc[comp['MAE'].idxmin(), 'model']
        st.dataframe(comp.style.highlight_min(subset=['MAE','RMSE','MAPE'], color='#c8f7c5'),
                     use_container_width=True)
        st.success(f"✅ Best model by MAE: **{best_mae}**")

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax, metric in zip(axes, ['MAE', 'RMSE', 'MAPE']):
            ax.barh(comp['model'], comp[metric], color='steelblue')
            ax.set_title(metric); ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Run step2_models.py first to generate model comparison results.")

st.markdown("---")
st.caption("HHS UAC Predictive Forecasting Dashboard | Built with Streamlit & Python")