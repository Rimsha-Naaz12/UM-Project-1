"""
STEP 1 - Exploratory Data Analysis & Preprocessing
UAC Predictive Forecasting Project

Run:
    python eda.py
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# CREATE REQUIRED FOLDERS
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Data Loading & Cleaning")
print("=" * 60)

# Load dataset
df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")

# Remove completely empty rows
df.dropna(how='all', inplace=True)

# Convert date column
df['Date'] = pd.to_datetime(
    df['Date'],
    format='%B %d, %Y',
    errors='coerce'
)

# Remove invalid dates
df.dropna(subset=['Date'], inplace=True)

# Sort by date
df.sort_values('Date', inplace=True)

# Set date as index
df.set_index('Date', inplace=True)

# Rename columns
df.columns = [
    'cbp_apprehended',
    'cbp_custody',
    'cbp_transfers',
    'hhs_care',
    'hhs_discharged'
]

# Clean numeric columns
for col in df.columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.strip()
    )

    df[col] = pd.to_numeric(df[col], errors='coerce')

# Display basic information
print(f"\nDate range: {df.index.min().date()} -> {df.index.max().date()}")
print(f"Total records: {len(df)}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. REINDEX & INTERPOLATE
# ──────────────────────────────────────────────────────────────────────────────
print("\nReindexing daily time series...")

# Create full daily date range
full_idx = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq='D'
)

# Reindex dataset
df = df.reindex(full_idx)

# Time interpolation
df.interpolate(method='time', inplace=True)

# Rename index
df.index.name = 'Date'

print(f"After reindex: {len(df)} days")

# Summary statistics
print("\nSummary Statistics:")
print(df.describe().round(1))

# ──────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────
print("\nGenerating engineered features...")

# Net pressure
df['net_pressure'] = (
    df['cbp_transfers'] - df['hhs_discharged']
)

# Lag features
for lag in [1, 7, 14]:
    df[f'hhs_care_lag{lag}'] = df['hhs_care'].shift(lag)

# Rolling statistics
df['hhs_care_roll7_mean'] = (
    df['hhs_care'].rolling(7).mean()
)

df['hhs_care_roll14_mean'] = (
    df['hhs_care'].rolling(14).mean()
)

df['hhs_care_roll7_std'] = (
    df['hhs_care'].rolling(7).std()
)

# Calendar features
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = (
    df['day_of_week'].isin([5, 6]).astype(int)
)

# Save cleaned dataset
df.to_csv("data/uac_cleaned.csv")

print("[+] Saved: data/uac_cleaned.csv")

# ──────────────────────────────────────────────────────────────────────────────
# 4. EXPLORATORY DATA ANALYSIS PLOTS
# ──────────────────────────────────────────────────────────────────────────────
print("\nCreating EDA visualizations...")

fig, axes = plt.subplots(3, 2, figsize=(16, 14))

fig.suptitle(
    'UAC Program - Exploratory Data Analysis',
    fontsize=16,
    fontweight='bold'
)

# ─── Plot 1: HHS Care Daily ──────────────────────────────────────────────────
axes[0, 0].plot(
    df.index,
    df['hhs_care'],
    color='steelblue',
    linewidth=1.2
)

axes[0, 0].set_title('Children in HHS Care (Daily)')
axes[0, 0].grid(alpha=0.3)

# ─── Plot 2: CBP Apprehensions ───────────────────────────────────────────────
axes[0, 1].plot(
    df.index,
    df['cbp_apprehended'],
    color='tomato',
    linewidth=0.8,
    alpha=0.7
)

axes[0, 1].plot(
    df.index,
    df['cbp_apprehended'].rolling(30).mean(),
    color='darkred',
    linewidth=2,
    label='30-Day Moving Avg'
)

axes[0, 1].set_title('Daily CBP Apprehensions')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# ─── Plot 3: Transfers vs Discharges ─────────────────────────────────────────
axes[1, 0].plot(
    df.index,
    df['cbp_transfers'].rolling(7).mean(),
    label='Transfers',
    color='orange'
)

axes[1, 0].plot(
    df.index,
    df['hhs_discharged'].rolling(7).mean(),
    label='Discharges',
    color='green'
)

axes[1, 0].set_title('7-Day Rolling: Transfers vs Discharges')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# ─── Plot 4: Net Pressure ────────────────────────────────────────────────────
net = df['net_pressure'].rolling(7).mean()

colors = [
    'tomato' if value > 0 else 'seagreen'
    for value in net
]

axes[1, 1].bar(
    df.index,
    net,
    color=colors,
    width=1.0,
    alpha=0.7
)

axes[1, 1].axhline(
    0,
    color='black',
    linewidth=0.8
)

axes[1, 1].set_title('Net Pressure (Transfers - Discharges)')
axes[1, 1].grid(alpha=0.3)

# ─── Plot 5: Monthly Average HHS Care ────────────────────────────────────────
monthly = df['hhs_care'].resample('ME').mean()

axes[2, 0].bar(
    monthly.index,
    monthly.values,
    color='mediumpurple',
    width=20
)

axes[2, 0].set_title('Monthly Average HHS Care')
axes[2, 0].grid(alpha=0.3)

# ─── Plot 6: Day-of-Week Analysis ────────────────────────────────────────────
dow_avg = df.groupby('day_of_week')['hhs_care'].mean()

axes[2, 1].bar(
    ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    dow_avg.values,
    color='cadetblue'
)

axes[2, 1].set_title('Average HHS Care by Day of Week')
axes[2, 1].grid(alpha=0.3)

# Improve layout
plt.tight_layout()

# Save figure
plt.savefig(
    'outputs/eda_overview.png',
    dpi=150,
    bbox_inches='tight'
)

plt.close()

print("[+] Saved: outputs/eda_overview.png")

# ──────────────────────────────────────────────────────────────────────────────
# 5. SEASONAL DECOMPOSITION
# ──────────────────────────────────────────────────────────────────────────────
print("\nPerforming seasonal decomposition...")

result = seasonal_decompose(
    df['hhs_care'].dropna(),
    model='additive',
    period=30
)

fig2 = result.plot()
fig2.set_size_inches(14, 10)

plt.tight_layout()

plt.savefig(
    'outputs/decomposition.png',
    dpi=150,
    bbox_inches='tight'
)

plt.close()

print("[+] Saved: outputs/decomposition.png")

# ──────────────────────────────────────────────────────────────────────────────
# COMPLETION MESSAGE
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 COMPLETE")
print("=" * 60)

print("\nGenerated Files:")
print("  • data/uac_cleaned.csv")
print("  • outputs/eda_overview.png")
print("  • outputs/decomposition.png")

print("\nNext Step:")
print("  Run -> python step2_models.py")