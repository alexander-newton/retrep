import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# =============================================================
# Hau, Huang & Wang (RESTUD 2020) — Table 5, Column (5)
# Stata target:
#   reghdfe dlnA1 dlnMW_tfp? lnExWk_tfp? lnExWk_dlnMW_tfp? ///
#           if all & inrange(year,2002,2008),
#           absorb(fnid ind2#year) vce(cluster countyyr)
# =============================================================

# ---------- Config & paths ----------
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# ---------- Load data ----------
filepath = os.path.join(INPUT_DATA_DIR, '151/firmdata.dta')
df = pd.read_stata(filepath)

# ---------- Minimal cleaning (consistent with earlier scripts) ----------
# Keep main window and full-sample flag
df = df[df['year'].between(2002, 2008) & (df['all'] == 1)].copy()

# Coerce only variables that may contain blanks/"." in the sample
num_cols = [
    'dlnA1',
    'dlnMW_tfp1','dlnMW_tfp2',
    'lnExWk_tfp1','lnExWk_tfp2',
    'lnExWk_dlnMW_tfp1','lnExWk_dlnMW_tfp2'
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# FE identifiers: use fnid as-is; compact numeric code for ind2#year
df['fe_fnid'] = df['fnid']
df['fe_ind2_year'] = df['ind2'].astype('int64') * 10000 + df['year'].astype('int64')

# Cluster groups: countyyr is already numeric in your sample
cluster_groups = df['countyyr'].values

# Drop rows missing only in variables we use
df = df.dropna(subset=num_cols).copy()

# ---------- Prepare inputs for replicate() ----------
# y in LEVELS (replicate() logs internally)
y = np.exp(df['dlnA1'].values)

# Regressors + FE code columns in X (FE columns are for indexing; estimator absorbs them)
X = pd.DataFrame({
    'const': 1.0,
    'dlnMW_tfp1': df['dlnMW_tfp1'].values,
    'dlnMW_tfp2': df['dlnMW_tfp2'].values,
    'lnExWk_tfp1': df['lnExWk_tfp1'].values,
    'lnExWk_tfp2': df['lnExWk_tfp2'].values,
    'lnExWk_dlnMW_tfp1': df['lnExWk_dlnMW_tfp1'].values,
    'lnExWk_dlnMW_tfp2': df['lnExWk_dlnMW_tfp2'].values,
    'fe_fnid': df['fe_fnid'].astype('int64').values,
    'fe_ind2_year': df['fe_ind2_year'].astype('int64').values,
})

# Pass FE column names and clustered SEs
fe = ['fe_fnid', 'fe_ind2_year']

# ---------- Metadata ----------
metadata = {
    'paper_id': '151',
    'table_id': '5',
    'panel_identifier': '5',  # Column (5)
    'model_type': 'log-log',
    'comments': 'Table 5 Col (5): TFP heterogeneity — dlnA1 on tfp-specific dlnMW, lnExWk, and interactions; firm & ind2×year FE; clustered by county-year.'
}

# ---------- Run replication ----------
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=['lnExWk_dlnMW_tfp1'],  # main coefficients of interest
    fe=fe,
    elasticity=True,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_groups}},
    output=True,
    output_dir=OUTPUT_DIR,
    replicated=True
)
