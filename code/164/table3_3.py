import os, yaml
import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# -------------------------------
# Load config and dataset (folder 164)
# -------------------------------
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# Main reduced-form dataset
data_path = os.path.join(INPUT_DATA_DIR, '164/Districts_1914_1895_1869_final.dta')
df = pd.read_stata(data_path)

# -------------------------------
# Reproduce Stata transforms used for Table 2 (Section 4.3)
# -------------------------------

# 1) Scale variables to match Stata preprocessing
# rlength1914 = rlength1914 / 100, expl_post = expl_post / 100
# (Make copies to avoid mutating the raw columns if you re-use df later.)
df['rlength1914_scaled'] = df['rlength1914'] / 100.0
df['expl_post_scaled'] = df['expl_post'] / 100.0

# 2) routeAVE = ((routeBA + routeRS + routeBH + routeLP)/4) * 100
df['routeAVE'] = ((df['routeBA'] + df['routeRS'] + df['routeBH'] + df['routeLP']) / 4.0) * 100.0

# 3) lextension_km2 = ln(extension_km2)  (the Stata code builds this; some datasets already have 'lextens')
if 'lextension_km2' not in df.columns:
    # prefer exact name; fall back to 'lextens' if already present
    if 'lextens' in df.columns and df['lextens'].notna().any():
        df['lextension_km2'] = df['lextens']
    else:
        df['lextension_km2'] = np.log(df['extension_km2'].astype(float))

# 4) Build lpop1869 and glpop6914 = lpop1914 - lpop1869 (as in the Stata block above)
#    The file has a per-row 'lpop' that varies by 'year'; we need cross-year max-by-district as in egen ... , by(dcode1895)
def max_by_year(col, year_value):
    tmp = df.loc[df['year'] == year_value, ['dcode1895', col]].rename(columns={col: f'{col}{year_value}'})
    return tmp

lpop_1869 = max_by_year('lpop', 1869)
lpop_1914 = max_by_year('lpop', 1914)

df = df.merge(lpop_1869, on='dcode1895', how='left')
df = df.merge(lpop_1914, on='dcode1895', how='left')

df['glpop6914'] = df['lpop1914'] - df['lpop1869']  # growth in logs, as in Stata

# 5) Controls: need lpop1869 explicitly for the 1914 cross-section regression
df['lpop1869_ctrl'] = df['lpop1869']

# -------------------------------
# Define the 1914 cross-section sample for Table 2
# -------------------------------
d1914 = df[df['year'] == 1914].copy()

# Drop rows with any missing needed vars
needed = [
    'glpop6914', 'rlength1914_scaled', 'routeAVE', 'expl_post_scaled',
    'latitude', 'longitude', 'lextension_km2', 'lpop1869_ctrl'
]
d1914 = d1914.dropna(subset=needed).copy()

# -------------------------------
# Set up variables for replicate()
# -------------------------------
# IMPORTANT: replicate() logs internally, so pass y IN LEVELS.
# glpop6914 is a difference in logs, so exp(glpop6914) returns the growth factor (P1914/P1869),
# and log(y) inside replicate() will recover glpop6914.
y = np.exp(d1914['glpop6914'].astype(float).values)

# Endogenous regressor and instruments
endog_var = 'rlength1914_scaled'
instruments = ['routeAVE', 'expl_post_scaled']

# Controls (no FEs in this table/column)
controls = ['latitude', 'longitude', 'lextension_km2', 'lpop1869_ctrl']

# Design matrices for 2SLS via replicate()
X = d1914[[endog_var] + controls].copy()
X = sm.add_constant(X, prepend=False)

Z = d1914[instruments + controls].copy()
Z = sm.add_constant(Z, prepend=False)

# -------------------------------
# Metadata for logging/output
# -------------------------------
metadata = {
    'paper_id': '164',
    'table_id': '2',
    'panel_identifier': '3',
    'model_type': 'log-linear',
    'comments': (
        'Table 2, Column 3 (full sample, year=1914). '
        'DV: glpop6914 (log pop growth 1869–1914); passed as level y = exp(glpop6914). '
        'Endogenous regressor: rlength1914/100 (rail length). '
        'Instruments: routeAVE, expl_post/100. Controls: latitude, longitude, ln(area), lpop1869. '
        'Robust (HC1) standard errors. No fixed effects.'
    )
}

# -------------------------------
# Run replication (IV with robust SEs)
# -------------------------------
res = replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=endog_var,
    endog_x=[endog_var],
    z=Z,
    elasticity=False,                 # DV is log after internal logging; not an elasticity spec
    fe=None,                          # no fixed effects in Table 2 col 3
    output=True,
    output_dir=OUTPUT_DIR,
    replicated=True
)

