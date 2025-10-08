"""
Industrial Espionage and Productivity - Table 2 Column 3 Replication
Authors: Albrecht Glitz and Erik Meyersson (2019)
Replicating Table 2, Column 3
"""

import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '009/regdata_3_yes_.33_.06.dta')
df = pd.read_stata(filepath)

cols_needed = [
    'c3difflnTFP',        # DV (log-form in data; we'll exponentiate)
    'difflnTFP',          # lagged ΔlnTFP control (as in their code block)
    'inf_gva',            # espionage inflow per GVA (main regressor)
    'diff_patents_gva',   # patent gap control
    'year', 'branch',     # fixed effects
    'weight_workers'      # analytic weights
]
df = df.dropna(subset=cols_needed).copy()

# Dependent variable in LEVELS (replicate() takes logs internally)
y = np.exp(df['c3difflnTFP'])

# Regressors: main vars first; FE passed separately
X_vars = ['inf_gva', 'diff_patents_gva', 'difflnTFP', 'year', 'branch']
X = sm.add_constant(df[X_vars], prepend=False)

# Clustering, weights, FE
clusters = df['branch']
weights = df['weight_workers']
fes = ['year', 'branch']   # corresponds to yd_* and br_* in Stata

# ---------------- Metadata ----------------
metadata = {
    'paper_id': '009',
    'table_id': '2',
    'panel_identifier': '3',        # 3-year differences (c3)
    'model_type': 'log-linear',     # log DV handled internally by replicate()
    'comments': 'Table 2, Col (3): ΔlnTFP gap on espionage(inf_gva) + patents(diff_patents_gva) + lagged ΔlnTFP with year & branch FE, clustered by branch, [aw=weight_workers]. Replicates: reg c3difflnTFP espionage patents difflnTFP yd_* br_* [aw=weight_workers], cluster(branch)'
}

# ---------------- Run replication ----------------
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='inf_gva',
    elasticity=False,
    fe=fes,
    weights=weights,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': clusters}},
    # output=True, output_dir=OUTPUT_DIR, replicated=True
)
