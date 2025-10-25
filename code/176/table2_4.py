import yaml
import os
import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm

# Make replicate() importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# -------------------------------
# Load config paths
# -------------------------------
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# -------------------------------
# Load time-series data (folder 176)
# -------------------------------
data_path = os.path.join(INPUT_DATA_DIR, '176/ts_union_ineq.dta')
df = pd.read_stata(data_path)

# -------------------------------
# Match Stata sample & construct time polynomials
# Table 2 uses years up to 2014 and a cubic in time
# -------------------------------
df = df[df['year'] <= 2014].copy()
df['t']   = df['year'] - 1937
df['tsq'] = (df['t'] ** 2) / 100.0
df['tcb'] = (df['t'] ** 3) / 1000.0

# -------------------------------
# Keep required vars; drop missing
# Column (4): L9010M on funionAverage + ln_chsr_eu + t, tsq, tcb + macro controls
# -------------------------------
req = [
    'L9010M',           # DV in logs in the data; pass in levels to replicate()
    'funionAverage',    # union density (avg Gallup/BLS)
    'ln_chsr_eu',       # skill share (CPS/Census efficiency units)
    't','tsq','tcb',    # time cubic
    'max_min_wage',     # macro controls
    'new_unemp',
    'TopMarginalTax'
]
df_clean = df.dropna(subset=req).copy()

print(f"N for Table 2 Col 4 after dropna: {len(df_clean)}")
print(f"Years included: {df_clean['year'].min()}–{df_clean['year'].max()}")

# -------------------------------
# Build y (levels) and X (regressors)
# -------------------------------
# replicate() logs internally, so exponentiate the logged DV
y = np.exp(df_clean['L9010M'].astype(float))

X = df_clean[[
    'funionAverage',
    'ln_chsr_eu',
    't','tsq','tcb',
    'max_min_wage',
    'new_unemp',
    'TopMarginalTax'
]].copy()

# add constant
X = sm.add_constant(X, prepend=True)

# -------------------------------
# Metadata
# -------------------------------
metadata = {
    'paper_id': '176',
    'table_id': '2',
    'panel_identifier': '4',
    'model_type': 'log-linear',
    'comments': 'Table 2, Column 4: log(90/10 male ratio) on union density + skill share + cubic trend + min wage + unemployment + top tax. Sample ≤ 2014.'
}

# -------------------------------
# Run replication
# -------------------------------
replicate(
    metadata=metadata,
    y=y,                        # levels; replicate() logs internally
    X=X,
    interest='funionAverage',   # coefficient of interest
    fe=None,                    # no FE (time polynomial instead)
    elasticity=False,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
