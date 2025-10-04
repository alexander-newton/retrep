import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np


import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# =============================================================
# Hau, Huang & Wang (RESTUD 2020) — Table 5, Column (2)
# Stata target:
#   reghdfe dlnA1 dlnMW lnExWk lnExWk_dlnMW if all & inrange(year,2002,2008),
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

# Only coerce the four variables that show blanks/"." in the sample
for c in ['dlnA1', 'dlnMW', 'lnExWk', 'lnExWk_dlnMW']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# FE identifiers: use fnid as-is; build a compact numeric code for ind2#year
df['fe_fnid'] = df['fnid']
df['fe_ind2_year'] = df['ind2'].astype('int64') * 10000 + df['year'].astype('int64')  

# Cluster groups: countyyr is already numeric in your sample
cluster_groups = df['countyyr'].values

# Drop rows missing only in variables we use
df = df.dropna(subset=['dlnA1','dlnMW','lnExWk','lnExWk_dlnMW']).copy()

# ---------- Prepare inputs for replicate() ----------
# y in LEVELS (replicate() logs internally)
y = np.exp(df['dlnA1'].values)

# Regressors + FE code columns in X (not used as regressors; estimator absorbs them)
X = pd.DataFrame({
    'const': 1.0,
    'dlnMW': df['dlnMW'].values,
    'lnExWk': df['lnExWk'].values,
    'lnExWk_dlnMW': df['lnExWk_dlnMW'].values,
    'fe_fnid': df['fe_fnid'].astype('int64').values,
    'fe_ind2_year': df['fe_ind2_year'].astype('int64').values,
})

# Pass FE column names and clustered SEs
fe = ['fe_fnid', 'fe_ind2_year']


# ---------- Metadata ----------
metadata = {
    'paper_id': '151',
    'table_id': '5',
    'panel_identifier': '2',  # Column (2)
    'model_type': 'log-log',
    'comments': 'Table 5 Col (2): dlnA1 on dlnMW, lnExWk, lnExWk_dlnMW with firm FE and industry×year FE; clustered by county-year.'
}

# ---------- Run replication ----------
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='lnExWk_dlnMW',
    fe=fe,
    elasticity=True,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_groups}},
    output=True,
    output_dir=OUTPUT_DIR,
    replicated=True
)


