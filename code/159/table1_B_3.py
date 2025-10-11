import yaml
import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm  # needed by replicate internals in some setups

# ---------- Import replicate helper ----------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# ---------- Config & paths ----------
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# ---------- Load required files (folder 159) ----------
td    = pd.read_stata(os.path.join(INPUT_DATA_DIR, '159/task_displacement_demographics_1980.dta'))
out80 = pd.read_stata(os.path.join(INPUT_DATA_DIR, '159/outcomes_demographics_1980.dta'))
out16 = pd.read_stata(os.path.join(INPUT_DATA_DIR, '159/outcomes_demographics_2016.dta'))

# ---------- Merge on 'group' ----------
df = td.merge(out80, on='group', how='inner', suffixes=('', '_80')) \
       .merge(out16, on='group', how='inner', suffixes=('', '_16'))

# ---------- Dependent variable: Δlog wage 2016–1980 ----------
w80 = 'ipums_hrwage_1980'
w16 = 'ipums_hrwage_2016'
df['change_wage_16'] = np.log(df[w16]) - np.log(df[w80])

# ---------- Columns for Column (3), Panel B ----------
# Task displacement (Panel B — automation-led)
tdvar = 'tdg_ro33_bea21_t_cd_87_16'
# Industry shifters
icg = 'icg_bea21_vadded_87_16'
# Manufacturing share 1980
manu80 = 'ipums_sector_manufacturing_1980'
if manu80 not in df.columns:
    raise ValueError("Missing 'ipums_sector_manufacturing_1980' in outcomes_demographics_1980.dta")

# ---------- Gender & Education fixed effects from 'group' ----------
tokens = df['group'].astype(str).str.split('_', expand=True)
gender_series = tokens[0] if tokens.shape[1] >= 1 else pd.Series(index=df.index, dtype='object')
educ_series   = tokens[4] if tokens.shape[1] >= 5 else (
    tokens[1] if tokens.shape[1] >= 2 else pd.Series(index=df.index, dtype='object')
)
df['gender_fe'] = pd.Categorical(gender_series).codes
df['educ_fe']   = pd.Categorical(educ_series).codes

# ---------- Build X and y ----------
X = df[[tdvar, icg, manu80, 'gender_fe', 'educ_fe']].copy()
X = sm.add_constant(X, prepend=False)
y = np.exp(df['change_wage_16'].astype(float))

# Indices of FE columns inside X
fe_indices = [X.columns.get_loc('gender_fe'), X.columns.get_loc('educ_fe')]

# ---------- Clean rows ----------
needed = [tdvar, icg, manu80, 'gender_fe', 'educ_fe', 'const']
mask = y.notna() & X[needed].notna().all(axis=1) & np.isfinite(y)
Xc = X.loc[mask]
yc = y.loc[mask]

# ---------- Weights (apply mask) ----------
w = df['ipums_whours_1980'].astype(float)
wc = w.loc[mask]

# ---------- Metadata ----------
metadata = {
    'paper_id': '159',
    'table_id': '1',
    'panel_identifier': 'B_3',
    'model_type': 'log-linear',
    'comments': 'Econometrica 2022 — Table 1, Col (3), Panel B: Δln wage (2016–1980) on automation-led task displacement (Panel B), industry shifters, 1980 manufacturing share; FE: gender & education.'
}

# ---------- Run replicate ----------
replicate(
    metadata=metadata,
    y=yc,
    X=Xc,
    interest=tdvar,
    fe=fe_indices,                 # absorb gender & education FE
    elasticity=False,              # not an elasticity
    weights=wc,                    # population weights aligned to sample
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
