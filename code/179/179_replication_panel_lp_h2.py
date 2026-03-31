import numpy as np
import os
import sys
import json
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from replication import replicate
from utils.json_utils import ReplicationJSONBuilder, load_config

# Load configuration
config = load_config()
raw_data_folder = config.get('rawdata', './rawdata')
intermediate_data_folder = config.get('intermediatedata', './intermediate_data')
final_data_folder = config.get('finaldata', './final_data')
output_folder = config.get('output', './output')

# =====================================
# Bilal & Kanzig (2026) - Panel Local Projections, horizon 2
# s03S_lp_panel_globalshock.do, first block
# DV: GDP_{t+2} / GDP_{t-1} (level ratio from raw GDP per capita)
# Shock: global temperature shock (Berkeley Earth, Hamilton filtered h=2, standardized)
# xtscc d{2}lnrgdppc_pwt L(0/2).shock L(1/2).dlnrgdppc_pwt
#        L(0/2).dummyrecession L(1/2).(dlnrgdppc_world_pwt dlnpoil_wti treasury1y)
#        c.lintrend#i.subregion_en, fe lag(4)
# =====================================

# Load data
df = pd.read_stata('/Users/ellenmunroe/Downloads/dataverse_files/1_empirics/data/micc_pwt_panel.dta')

# Convert categorical columns to numeric codes
for col in df.columns:
    if df[col].dtype.name == 'category':
        df[col] = df[col].cat.codes

# Sample restriction: 1960-2019
df = df[(df['year'] >= 1960) & (df['year'] <= 2019)].copy()

# Sort for panel operations
df = df.sort_values(['country_code_en', 'year']).reset_index(drop=True)

# Construct additional variables
df['dummyrecession'] = df['recessiondates']
df['lintrend'] = df['year'] - 1960 + 1

# First differences of log variables (within country)
df['dlnrgdppc_pwt'] = df.groupby('country_code_en')['lnrgdppc_pwt'].diff()
df['dlnrgdppc_world_pwt'] = df.groupby('country_code_en')['lnrgdppc_world_pwt'].diff()
df['dlnpoil_wti'] = df.groupby('country_code_en')['lnpoil_wti'].diff()

# Shock variable
shock_var = 'gtmp_bkly_aw_dtfe2s'

# LP parameters
p = 2       # lags of DV
ps = 2      # lags of shock
horizon = 2

# Create lags/leads within country
def create_lag(df, var, lag, groupvar='country_code_en'):
    return df.groupby(groupvar)[var].shift(lag)

def create_lead(df, var, lead, groupvar='country_code_en'):
    return df.groupby(groupvar)[var].shift(-lead)

# Create lagged shock: L0, L1, L2
for l in range(0, ps + 1):
    df[f'{shock_var}_L{l}'] = create_lag(df, shock_var, l)

# Create lagged DV (first difference): L1, L2
for l in range(1, p + 1):
    df[f'dlnrgdppc_pwt_L{l}'] = create_lag(df, 'dlnrgdppc_pwt', l)

# Create lagged controls
for l in range(0, 3):
    df[f'dummyrecession_L{l}'] = create_lag(df, 'dummyrecession', l)

for var in ['dlnrgdppc_world_pwt', 'dlnpoil_wti', 'treasury1y']:
    for l in range(1, 3):
        df[f'{var}_L{l}'] = create_lag(df, var, l)

# Create subregion-specific linear time trends
subregion_dummies = pd.get_dummies(df['subregion_en'], prefix='subreg', drop_first=True).astype(float)
subregion_trend_cols = []
for col in subregion_dummies.columns:
    trend_col = f'{col}_trend'
    df[trend_col] = subregion_dummies[col].values * df['lintrend'].values
    subregion_trend_cols.append(trend_col)

# DV in levels: GDP_{t+h} / GDP_{t-1}
L1_rgdppc = create_lag(df, 'rgdppc_pwt', 1)
Fi_rgdppc = create_lead(df, 'rgdppc_pwt', horizon)
df['y_level'] = Fi_rgdppc / L1_rgdppc

# Define RHS variable groups
shock_lags = [f'{shock_var}_L{l}' for l in range(0, ps + 1)]
dv_lags = [f'dlnrgdppc_pwt_L{l}' for l in range(1, p + 1)]
recession_lags = [f'dummyrecession_L{l}' for l in range(0, 3)]
control_lags = []
for var in ['dlnrgdppc_world_pwt', 'dlnpoil_wti', 'treasury1y']:
    for l in range(1, 3):
        control_lags.append(f'{var}_L{l}')

# X columns: interest (contemporaneous shock) first, then shock lags, controls, trends, country FE
x_cols = shock_lags + dv_lags + recession_lags + control_lags + subregion_trend_cols + ['country_code_en']

# Drop missing
all_vars = ['y_level'] + x_cols
reg_df = df[all_vars].dropna().reset_index(drop=True)

# =====================================
# Horizon 2
# =====================================

metadata_h2 = {
    'paper_id': '179',
    'table_id': 'PanelLP_GDP',
    'panel_identifier': 'h2',
    'model_type': 'log-linear'
}

# DV in levels (GDP ratio)
y_h2 = pd.DataFrame(reg_df['y_level'].values, columns=['rgdppc_ratio'])

# X matrix: interest variable first, then other regressors, then FE
X_h2 = reg_df[x_cols].reset_index(drop=True)

# Save to parquet
out_dir = os.path.join(output_folder, '179', 'PanelLP_GDP_h2')
os.makedirs(out_dir, exist_ok=True)

y_h2.to_parquet(os.path.join(out_dir, 'y.parquet'), index=False)
X_h2.to_parquet(os.path.join(out_dir, 'X.parquet'), index=False)

# Save metadata
with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata_h2, f, indent=2)

# Separate X into regressors and FE for replicate()
x_regressor_cols = shock_lags + dv_lags + recession_lags + control_lags + subregion_trend_cols

replicate(
    metadata=metadata_h2,
    y=y_h2.values,
    X=X_h2[x_regressor_cols].values,
    interest=f'{shock_var}_L0',
    endog_x=None,
    z=None,
    fe=X_h2[['country_code_en']].values,
    elasticity=True,
    replicated=True,
    kwargs_estimator={'estimator_type': 'ols'},
    kwargs_ols=None,
    kwargs_ppml=None,
    fit_full_model=False,
    output=True,
    output_dir=output_folder
)
