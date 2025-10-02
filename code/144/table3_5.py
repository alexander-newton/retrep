# code/144/table3_5.py

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
# Table 3, Column 5 (IV-2SLS):
# - DV: lCtilde (log O&M var cost) -> pass levels exp(lCtilde)
# - Endogenous: lq (output), lsstd (SO2 standard)
# - IVs: lstatedemand -> lq; lplow & lphigh -> lsstd
# - Exogenous of interest: isnoeffort (rate-case year)
# - Other exogenous: isneither, lN, FGD, lpl, lpc, lpo, lpg
# - FE: firmcase FE + year FE (as codes)
# - Cluster: firmcase
# - Sample: phase1 == 1
# -------------------------------

# Sample restriction: Phase I
df = df[df['phase1'] == 1].copy()

# Required columns
dep_log = 'lCtilde'
endog_vars = ['lq', 'lsstd']
instruments = ['lstatedemand', 'lplow', 'lphigh']
exo_interest = 'isnoeffort'
exo_others = ['isneither', 'lN', 'FGD', 'lpl', 'lpc', 'lpo', 'lpg', 'lsstdFGD']
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
# X must contain endogenous + all exogenous (incl. FE codes)
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
# FE list passed separately (so replicate can absorb/track them if needed)
# (we pass the *names* of the FE columns)
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
    'table_id': '3',
    'panel_identifier': '5',
    'model_type': 'log-linear',
    'comments': (
        'Table 3, Column 5: IV with FE (i.year & i.firmcase). '
        'Endog: lq, lsstd. IVs: lstatedemand (for lq), lplow & lphigh (for lsstd). '
        'Exogenous includes isnoeffort (coef of interest) and isneither, lN, FGD, lpl, lpc, lpo, lpg. '
        'Clustered by firmcase. y=exp(lCtilde); replicate() logs internally.'
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
    endog_x=endog_vars,       # ['lq','lsstd'] (names)
    z=Z,
    fe=fe,                    # ['firmcase_fe','year_fe']
    elasticity=False,
    kwargs_ols = {'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True,
    output_dir=OUTPUT_DIR,
    replicated=True
)
