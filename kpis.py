"""
STEP 3 - KPI Calculation & Early Warning Signals
UAC Predictive Forecasting Project
Run: python step3_kpis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings('ignore')

print("=" * 60)
print("STEP 3: KPI Calculation & Early Warning System")
print("=" * 60)

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/uac_cleaned.csv", index_col='Date', parse_dates=True)

# ─── KPI 1: Capacity Breach Probability ──────────────────────────────────────
# Flag days where HHS care > 90th percentile as "capacity stress"
p90 = df['hhs_care'].quantile(0.90)
p75 = df['hhs_care'].quantile(0.75)
df['capacity_stress'] = (df['hhs_care'] > p90).astype(int)
df['elevated_load']   = (df['hhs_care'] > p75).astype(int)

breach_prob = df['capacity_stress'].rolling(30).mean() * 100
print(f"90th percentile threshold: {p90:.0f} children")
print(f"Current 30-day breach probability: {breach_prob.iloc[-1]:.1f}%")

# ─── KPI 2: Net Flow Imbalance ────────────────────────────────────────────────
df['intake_exit_ratio'] = df['cbp_transfers'] / df['hhs_discharged'].replace(0, np.nan)
recent_ratio = df['intake_exit_ratio'].rolling(30).mean().iloc[-1]
print(f"Recent 30-day intake/exit ratio: {recent_ratio:.2f} (>1 = growing backlog)")

# ─── KPI 3: Average Length of Stay Proxy ─────────────────────────────────────
# LOS proxy = HHS Care / Daily Discharges (Little's Law)
df['los_proxy'] = df['hhs_care'] / df['hhs_discharged'].replace(0, np.nan)
avg_los = df['los_proxy'].rolling(30).mean().iloc[-1]
print(f"Estimated avg length of stay (recent 30d): {avg_los:.1f} days")

# ─── KPI 4: Surge Detection ───────────────────────────────────────────────────
# Z-score of 7-day rolling mean of net pressure
net_press_roll = df['net_pressure'].rolling(7).mean()
z_scores = (net_press_roll - net_press_roll.rolling(90).mean()) / net_press_roll.rolling(90).std()
df['surge_z'] = z_scores
surge_events = (z_scores > 2.0)
print(f"Surge events detected (z > 2): {surge_events.sum()} days")

# ─── SAVE KPI TABLE ───────────────────────────────────────────────────────────
kpi_summary = {
    'Metric': [
        'Peak Care Load (max)',
        'Current Care Load',
        '90th Pct Threshold',
        '30-Day Breach Prob (%)',
        'Intake/Exit Ratio (30d)',
        'Avg LOS Proxy (days)',
        'Surge Events Detected',
    ],
    'Value': [
        f"{df['hhs_care'].max():.0f}",
        f"{df['hhs_care'].iloc[-1]:.0f}",
        f"{p90:.0f}",
        f"{breach_prob.iloc[-1]:.1f}%",
        f"{recent_ratio:.2f}",
        f"{avg_los:.1f}",
        str(int(surge_events.sum())),
    ]
}
kpi_df = pd.DataFrame(kpi_summary)
kpi_df.to_csv("outputs/kpi_summary.csv", index=False)
print(f"\n[+] KPI summary saved to outputs/kpi_summary.csv")
print(kpi_df.to_string(index=False))

# ─── EARLY WARNING DASHBOARD PLOT ────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle('Early Warning & KPI Dashboard', fontsize=14, fontweight='bold')

# Panel 1: HHS Care with threshold lines
ax = axes[0]
ax.plot(df.index, df['hhs_care'], color='steelblue', linewidth=1, label='HHS Care')
ax.axhline(p75, color='orange', linestyle='--', linewidth=1.2, label=f'75th pct ({p75:.0f})')
ax.axhline(p90, color='red',    linestyle='--', linewidth=1.2, label=f'90th pct ({p90:.0f})')
ax.fill_between(df.index, df['hhs_care'], p90,
                where=(df['hhs_care'] > p90), color='red', alpha=0.2, label='Breach Zone')
ax.set_ylabel('Children in HHS Care')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_title('HHS Care Load with Capacity Thresholds')

# Panel 2: 30-day rolling breach probability
ax = axes[1]
ax.plot(df.index, breach_prob, color='tomato', linewidth=1.5)
ax.axhline(50, color='orange', linestyle=':', linewidth=1)
ax.fill_between(df.index, breach_prob, 0, alpha=0.2, color='tomato')
ax.set_ylabel('Breach Probability (%)')
ax.set_ylim(0, 110)
ax.grid(alpha=0.3)
ax.set_title('30-Day Rolling Capacity Breach Probability')

# Panel 3: Surge Z-score
ax = axes[2]
ax.plot(df.index, df['surge_z'], color='purple', linewidth=1)
ax.axhline(2.0, color='red', linestyle='--', linewidth=1.2, label='Surge Threshold (z=2)')
ax.axhline(-2.0, color='green', linestyle='--', linewidth=1.2, label='Relief Threshold (z=-2)')
ax.fill_between(df.index, df['surge_z'], 2,
                where=(df['surge_z'] > 2), color='red', alpha=0.3)
ax.set_ylabel('Surge Z-score')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_title('Net Pressure Surge Detection (Z-Score)')

plt.tight_layout()
plt.savefig('outputs/early_warning.png', dpi=150, bbox_inches='tight')
plt.close()
print("[+] Saved outputs/early_warning.png")
print("\nSTEP 3 COMPLETE. Run step4_app.py or launch the Streamlit app next.")