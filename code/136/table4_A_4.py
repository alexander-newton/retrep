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

# Folder 136, file: Data/data with conflict.dta
data_path = os.path.join(INPUT_DATA_DIR, '136/data with conflict.dta')
df = pd.read_stata(data_path)

# -------------------------------
# Sample: long differences (1940 vs 1980), base sample
# -------------------------------
df = df[(df['newsample40'] == 1) & (df['year'].isin([1940, 1980]))].copy()

# -------------------------------
# Variables (exact names from your dataset)
# DV (logged in Stata): logdeathpop40U  = log(1 + battle_deaths / pop_1940)
# Endogenous regressor: logmaddpop
# Instrument: compsjmhatit
# FE: country + year
# Cluster: ctrycluster
# -------------------------------
dv_log = 'logdeathpop40U'
endog_var = 'logmaddpop'
instrument = 'compsjmhatit'

# y IN LEVELS, because replicate() logs internally:
y = np.exp(df[dv_log].astype(float).values)

# Fixed effects: country + year
# (You have both 'country' (string) and 'countrynum' (numeric); use 'country' for clarity)
df['country_fe'] = pd.Categorical(df['country']).codes
df['year_fe'] = pd.Categorical(df['year'].astype(int)).codes
fe_vars = ['country_fe', 'year_fe']

# Design matrices
X = df[[endog_var] + fe_vars].copy()
X = sm.add_constant(X, prepend=False)

Z = df[[instrument] + fe_vars].copy()
Z = sm.add_constant(Z, prepend=False)

# Cluster groups (as in Stata)
cluster = pd.Categorical(df['ctrycluster']).codes

# -------------------------------
# Metadata (for your logs)
# -------------------------------
metadata = {
    'paper_id': '136',
    'table_id': '4',
    'panel_identifier': 'A_4',
    'model_type': 'log-linear',
    'comments': (
        'Table 4, Column 4 (long differences). DV = log(1 + battle deaths / pop_1940), '
        'endog = logmaddpop, IV = compsjmhatit. Country FE + year FE, clustered by ctrycluster. '
        'Sample: newsample40==1 & year in {1940,1980}.'
    )
}

# -------------------------------
# Run replication
# -------------------------------
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=endog_var,
    endog_x=[endog_var],
    z=Z,
    elasticity=False,  # semi-log DV; do not compute elasticity
    fe=fe_vars,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True,
    output_dir=OUTPUT_DIR,
    replicated=True

)
