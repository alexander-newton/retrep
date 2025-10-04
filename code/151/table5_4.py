import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# =============================================================
# Hau, Huang & Wang (RESTUD 2020) — Table 5, Column (4)
# Stata target:
#   reghdfe dlnA1 dlnMW_own? lnExWk_own? lnExWk_dlnMW_own? soe prv ///
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

# ---------- Minimal cleaning based on your sample ----------
# Keep main window and full-sample flag
df = df[df['year'].between(2002, 2008) & (df['all'] == 1)].copy()

# Coerce only variables that may contain blanks/"." in the sample
num_cols = [
    'dlnA1',
    'dlnMW_own1','dlnMW_own2','dlnMW_own3',
    'lnExWk_own1','lnExWk_own2','lnExWk_own3',
    'lnExWk_dlnMW_own1','lnExWk_dlnMW_own2','lnExWk_dlnMW_own3',
    'soe','prv'
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# FE identifiers: use fnid as-is; compact numeric code for ind2#year
df['fe_fnid'] = df['fnid']
df['fe_ind2_year'] = df['ind2'].astype('int64') * 10000 + df['year'].astype('int64')

# Cluster groups: countyyr is already numeric in your sample
cluster_groups = df['countyyr'].values

# Drop rows missing only in variables we use
df = df.dropna(subset=[
    'dlnA1',
    'dlnMW_own1','dlnMW_own2','dlnMW_own3',
    'lnExWk_own1','lnExWk_own2','lnExWk_own3',
    'lnExWk_dlnMW_own1','lnExWk_dlnMW_own2','lnExWk_dlnMW_own3',
    'soe','prv'
]).copy()

# ---------- Prepare inputs for replicate() ----------
# y in LEVELS (replicate() logs internally)
y = np.exp(df['dlnA1'].values)

# Regressors + FE code columns in X (FE columns are for indexing; estimator absorbs them)
X = pd.DataFrame({
    'const': 1.0,
    'dlnMW_own1': df['dlnMW_own1'].values,
    'dlnMW_own2': df['dlnMW_own2'].values,
    'dlnMW_own3': df['dlnMW_own3'].values,
    'lnExWk_own1': df['lnExWk_own1'].values,
    'lnExWk_own2': df['lnExWk_own2'].values,
    'lnExWk_own3': df['lnExWk_own3'].values,
    'lnExWk_dlnMW_own1': df['lnExWk_dlnMW_own1'].values,
    'lnExWk_dlnMW_own2': df['lnExWk_dlnMW_own2'].values,
    'lnExWk_dlnMW_own3': df['lnExWk_dlnMW_own3'].values,
    'soe': df['soe'].values,
    'prv': df['prv'].values,
    'fe_fnid': df['fe_fnid'].astype('int64').values,
    'fe_ind2_year': df['fe_ind2_year'].astype('int64').values,
})

# Pass FE column names and clustered SEs
fe = ['fe_fnid', 'fe_ind2_year']

# ---------- Metadata ----------
metadata = {
    'paper_id': '151',
    'table_id': '5',
    'panel_identifier': '4',  # Column (4)
    'model_type': 'log-log',
    'comments': 'Table 5 Col (4): ownership heterogeneity — dlnA1 on own-specific dlnMW, lnExWk, and interaction; firm & ind2×year FE; clustered by county-year.'
}

# ---------- Run replication ----------
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=['lnExWk_dlnMW_own3'],
    fe=fe,
    elasticity=True,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_groups}},
    output=True,
    output_dir=OUTPUT_DIR,
    replicated=True
)
