import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# -----------------------
# Load configuration
# -----------------------
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# -----------------------
# LOAD DATA (policy-cycle CSV, no header)
# -----------------------
primary_path = os.path.join(INPUT_DATA_DIR, '086', 'data_policycle.csv')
fallback_path = '/mnt/data/data_policycle.csv'
filepath = primary_path if os.path.exists(primary_path) else fallback_path

raw = pd.read_csv(filepath, header=None)

# Column mapping (MATLAB indices 1-based → pandas 0-based)
col_map = {'date': 0, 'dem_dum4': 5, 'vwretd': 7, 'rf': 10}
df = raw.iloc[:, [col_map['date'], col_map['dem_dum4'], col_map['vwretd'], col_map['rf']]].copy()
df.columns = ['date', 'dem_dum4', 'vwretd', 'rf']

# -----------------------
# CLEAN + SAMPLE WINDOW
# -----------------------
df.replace(-99, np.nan, inplace=True)
df['date'] = pd.to_numeric(df['date'], errors='coerce')
df['vwretd'] = pd.to_numeric(df['vwretd'], errors='coerce')
df['rf'] = pd.to_numeric(df['rf'], errors='coerce')
df['dem_dum4'] = pd.to_numeric(df['dem_dum4'], errors='coerce')
df.dropna(subset=['date', 'dem_dum4', 'vwretd', 'rf'], inplace=True)

# Restrict to sample
df = df[(df['date'] >= 192701) & (df['date'] <= 201512)].copy()
df = df.sort_values('date').reset_index(drop=True)

# Ensure Dem is binary
df['Dem'] = (df['dem_dum4'] > 0).astype(int)

# -----------------------
# Construct DV as ratio (replicate logs internally)
# -----------------------
one_plus_mkt = 1.0 + df['vwretd'].to_numpy()
one_plus_rf  = 1.0 + df['rf'].to_numpy()
valid = (one_plus_mkt > 0) & (one_plus_rf > 0)
df = df.loc[valid].copy()
df['y_ratio'] = (1.0 + df['vwretd']) / (1.0 + df['rf'])

# -----------------------
# Build Year 1 dummies
# -----------------------
n = len(df)
dem_series = df['Dem'].to_numpy()
dem_1 = np.zeros(n, dtype=int)
rep_1 = np.zeros(n, dtype=int)

# transition to Dem
trans_to_dem = np.where((dem_series[1:] == 1) & (dem_series[:-1] == 0))[0]
for idx in trans_to_dem:
    start = idx + 1
    end = min(start + 12, n)
    dem_1[start:end] = 1

# transition to Rep
trans_to_rep = np.where((dem_series[1:] == 0) & (dem_series[:-1] == 1))[0]
for idx in trans_to_rep:
    start = idx + 1
    end = min(start + 12, n)
    rep_1[start:end] = 1

df['dem_1'] = dem_1
df['rep_1'] = rep_1

# Keep only Year 1 months (either Dem or Rep)
subset = (df['dem_1'] == 1) | (df['rep_1'] == 1)
df_sub = df.loc[subset].copy()

# Treatment dummy = dem_1
df_sub['Dem1'] = df_sub['dem_1'].astype(int)

# -----------------------
# DESIGN: Reduced form — constant + Dem1
# -----------------------
y = df_sub['y_ratio']
X = sm.add_constant(df_sub[['Dem1']], has_constant='add')

# -----------------------
# METADATA
# -----------------------
metadata = {
    'paper_id': '086',
    'table_id': '2',
    'panel_identifier': '3',
    'model_type': 'log-linear',
    'comments': 'Table 2, Column 1 (Year 1 only)--as their preferred col. DV ratio (1+R_mkt)/(1+R_rf) on Dem1 dummy; monthly sample 1927:01–2015:12.'
}

# -----------------------
# RUN
# -----------------------
res = replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='Dem1',
    elasticity=False,
    fe=None,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)

# -----------------------
# Notes
# -----------------------
print("\nNote: The Dem1 coefficient is monthly. Multiply by 1200 to get the annualized % difference reported in Table 2, Column 1 (Year 1).")
print("Other columns in Table 2 (Years 1–2; Years 1–3; Full term) follow the same pattern with different dummies.")
