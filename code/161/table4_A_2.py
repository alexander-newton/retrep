import yaml, os
import pandas as pd
import numpy as np
import statsmodels.api as sm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# -------------------------------
# Load config and dataset
# -------------------------------
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

data_path = os.path.join(INPUT_DATA_DIR, '161/countrylevel_data.dta')
df = pd.read_stata(data_path)

# -------------------------------
# Sample restrictions (match Stata script)
# -------------------------------
df = df[(df['basesample'] == 1) & (df['year'] > 1960) & (df['predyield1'].notna())].copy()

# -------------------------------
# Variables (ensure existence, minimal)
# DV (logged in Stata): lnrgdppc
# Regressor: predyield1aes  ("ln GR x initial AES")
# Interaction block: year##c.aes_1961  -> aes_1961 main effect + year×aes_1961
# FE: isocode + year
# Cluster: isocode
# -------------------------------
needed = ['lnrgdppc', 'predyield1aes', 'aes_1961', 'isocode', 'year']
df = df.dropna(subset=needed).copy()

# -------------------------------
# Fixed effects and cluster groups (exactly like your example style)
# -------------------------------
df['country_fe'] = pd.Categorical(df['isocode']).codes
df['year_fe']    = pd.Categorical(df['year'].astype(int)).codes
fe_vars = ['country_fe', 'year_fe']
cluster = pd.Categorical(df['isocode']).codes

# -------------------------------
# Build year × aes_1961 interactions (minimal)
# -------------------------------
year_d = pd.get_dummies(df['year'].astype(int), prefix='y', drop_first=True)
inter_df = year_d.mul(df['aes_1961'].values, axis=0)  # each dummy * aes_1961

# -------------------------------
# Design matrices (y in LEVELS since replicate() logs internally)
# -------------------------------
y = np.exp(df['lnrgdppc'].astype(float).values)

X = pd.concat(
    [df[['predyield1aes'] + fe_vars + ['aes_1961']], inter_df],
    axis=1
)
X = sm.add_constant(X, prepend=False)

# -------------------------------
# Metadata
# -------------------------------
metadata = {
    'paper_id': '161',
    'table_id': '4',
    'panel_identifier': 'A_2',
    'model_type': 'log-linear',
    'comments': (
        'Table 4, Panel A, Column (2): lnrgdppc on predyield1aes with '
        'isocode FE, year FE, and year×aes_1961 interactions. '
        'Clustered by isocode. Sample: basesample==1, year>1960, predyield1!=.'
    )
}

# -------------------------------
# Run replication (OLS, clustered SEs)
# -------------------------------
replicate(
    metadata=metadata,
    y=y,                         # IMPORTANT: pass levels; replicate() logs internally
    X=X,
    interest='predyield1aes',
    elasticity=False,            # semi-log DV; do not compute elasticity
    fe=fe_vars,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
