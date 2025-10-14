# === Fajgelbaum & Redding (2022) — Table 1, Col (1) ===
# Target Stata: reg lpopdens ldistTOP4 if year==1869 , robust

import os, sys, yaml
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Make replication package importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# ---- Config & paths ----
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)
OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# ---- Load the reduced-form dataset (folder 164) ----
filepath = os.path.join(INPUT_DATA_DIR, '164/Districts_1914_1895_1869_final.dta')
df = pd.read_stata(filepath)

# ---- Restrict to 1869 cross-section (full sample) ----
df = df[df['year'] == 1869].copy()

# ---- Build ldistTOP4 from available distance variables ----
# Uses only variables you listed: distBA, distRS, distLP, distBH
needed = ['distBA', 'distRS', 'distLP', 'distBH', 'lpopdens']


distTOP4 = pd.concat(
    [df['distBA'], df['distRS'], df['distLP'], df['distBH']],
    axis=1
).min(axis=1)

# Guard against zeros before log
distTOP4 = distTOP4.replace(0, np.nan)
df['ldistTOP4'] = np.log(distTOP4)

# ---- Drop rows with missing DV or regressor ----
df = df.dropna(subset=['lpopdens', 'ldistTOP4']).copy()

# ---- Prepare y and X ----
# DV must be in LEVELS because replicate() logs internally
y = np.exp(df['lpopdens'])

# Only regressor is ldistTOP4 + constant
X = df[['ldistTOP4']].copy()
X = sm.add_constant(X, prepend=False)

# ---- Metadata ----
metadata = {
    'paper_id': '164',
    'table_id': '1',
    'panel_identifier': '1',
    'model_type': 'log-linear',
    'comments': 'Table 1, Column (1): lpopdens on ldistTOP4, year=1869, full sample. DV passed in levels because replicate() logs internally.'
}

# ---- Run replication ----
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='ldistTOP4',
    fe=None,                # No FE in this column
    elasticity=False,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
