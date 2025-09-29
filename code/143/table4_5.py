import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# Load BSV network dataset (folder 143)
filepath = os.path.join(INPUT_DATA_DIR, '143/BSV Working-Instrumented Dataset.dta')
df = pd.read_stata(filepath)

# -------------------------------
# Table 4, Column 5 (2SLS):
# distance-3 IV only, connected firms,
# firm FE + year FE + community×year FE (20-comm partition via module06),
# clustered by module06.
# -------------------------------

# Sample restriction: connected firms
df_clean = df.copy()
df_clean = df_clean[df_clean['connected'] == 1]

# Variables
dep_log = 'lsales'                # DV in logs in Stata; we pass levels (exp(lsales))
endog_var = 'sr_symm_spilloginv1_1'         # connection-weighted log external R&D (endogenous)
instrument = 'binary_IV2spilloginv1_1'     # distance-3 instrument (preferred)

controls = [
    'lppent','lemp',  'lgrd1', 'lgrd1_dum',
    'lgspilltec1', 'lgspillsic1', 'spilloggeo1',
    'lsales_ind', 'lsales_ind1', 'lpind_ind'
]

# Required columns
required_vars = [dep_log, endog_var, instrument, 'num', 'year', 'module06'] + controls
df_clean = df_clean.dropna(subset=required_vars)

# y in levels (replicate() logs internally)
y = np.exp(df_clean[dep_log].astype(float).values)

# Fixed effects:
# - firm FE: num
# - year FE: year
# - community×year FE: module06 × year
df_clean['firm_code'] = pd.Categorical(df_clean['num']).codes
df_clean['year'] = df_clean['year'].astype(int)
df_clean['commyear_code'] = pd.Categorical(
    df_clean['module06'].astype(int).astype(str) + '_' + df_clean['year'].astype(str)
).codes

fe_vars = ['firm_code', 'year', 'commyear_code']

# X matrix: endogenous + controls + FE codes + constant
X_vars = [endog_var] + controls + fe_vars
X = df_clean[X_vars].copy()
X = sm.add_constant(X, prepend=False)

# z matrix: instrument + SAME controls + FE codes + constant
z_vars = [instrument] + controls + fe_vars
z = df_clean[z_vars].copy()
z = sm.add_constant(z, prepend=False)

# Cluster variable: community (module06)
cluster = pd.Categorical(df_clean['module06'].astype(int)).codes

# Metadata
metadata = {
    'paper_id': '143',
    'table_id': '4',
    'panel_identifier': '5',
    'model_type': 'log-log',
    'comments': (
        'Table 4, Column 5: 2SLS with distance-3 instrument only (IV2spilloginv1); '
        'sample = connected firms; firm FE, year FE, community×year FE (module06×year); '
        'clustered by module06. DV passed in levels (exp(lsales)); replicate() logs internally.'
    )
}

# Run replication
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=endog_var,
    endog_x=[endog_var],
    z=z,
    elasticity=True,
    fe=fe_vars,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True,
    output_dir=OUTPUT_DIR,
    replicated=True
)
