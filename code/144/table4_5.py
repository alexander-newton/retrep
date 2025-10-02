# code/144/table4_5.py

import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# -------------------------------
# Config
# -------------------------------
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# -------------------------------
# Load dataset (paper 144)
# -------------------------------
filepath = os.path.join(INPUT_DATA_DIR, '144/pollution_price_reg_elec_utilities_all.dta')
df = pd.read_stata(filepath)

# -------------------------------
# Table 4, Column 5 (IV-2SLS):
# - DV: lheatrate (log heat rate) -> pass levels exp(lheatrate)
# - Endogenous: lq (output)
# - IV: lstatedemand -> lq
# - Exogenous of interest: isnoeffort (rate-case year)
# - Other exogenous: isneither, lN, FGD
# - FE: firmcase FE + year FE (as codes)
# - Cluster: firmcase
# - Sample: phase1 == 1
# -------------------------------

# Sample restriction: Phase I
df = df[df['phase1'] == 1].copy()

# Required columns
dep_log = 'lheatrate'
endog_vars = ['lq']
instruments = ['lstatedemand']
exo_interest = 'isnoeffort'
exo_others = ['isneither', 'lN', 'FGD']
fe_keys = ['firmcase', 'year']

required = [dep_log, exo_interest] + exo_others + endog_vars + instruments + fe_keys
df = df.dropna(subset=required).copy()

# y in levels (replicate() takes logs internally)
y = np.exp(df[dep_log].astype(float).values)

# -------------------------------
# Fixed effects as numeric codes
# -------------------------------
df['firmcase_fe'] = pd.Categorical(df['firmcase'].astype(str)).codes
df['year_fe'] = df['year'].astype(int)

fe_vars = ['firmcase_fe', 'year_fe']

# -------------------------------
# Build X and Z
# X must include endogenous + all exogenous (incl. FE codes)
# Z = instruments + same exogenous (incl. FE codes)
# -------------------------------
X_vars = endog_vars + [exo_interest] + exo_others + fe_vars
Z_vars = instruments + [exo_interest] + exo_others + fe_vars

X = df[X_vars].copy().astype(float)
Z = df[Z_vars].copy().astype(float)

# Add constants (prepend=False to mirror your example style)
X = sm.add_constant(X, prepend=False)
Z = sm.add_constant(Z, prepend=False)

# -------------------------------
# FE list passed separately
# -------------------------------
fe = fe_vars

# -------------------------------
# Clustering: by firmcase
# -------------------------------
cluster = pd.Categorical(df['firmcase'].astype(str)).codes


# -------------------------------
# Metadata
# -------------------------------
metadata = {
    'paper_id': '144',
    'table_id': '4',
    'panel_identifier': '4',
    'model_type': 'log-linear',
    'comments': (
        'Table 4, Column 5: IV with FE (i.year & i.firmcase). '
        'Endog: lq. IV: lstatedemand (for lq). '
        'Exogenous includes isnoeffort (coef of interest), isneither, lN, FGD. '
        'Clustered by firmcase. y=exp(lheatrate); replicate() logs internally.'
    ),
}

# -------------------------------
# Run replication
# -------------------------------
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=exo_interest,   # 'isnoeffort'
    endog_x=endog_vars,       # ['lq'] (names)
    z=Z,
    fe=fe,                    # ['firmcase_fe','year_fe']
    elasticity=False,
    kwargs_ols = {'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True,
    output_dir=OUTPUT_DIR,
    replicated=True
)
