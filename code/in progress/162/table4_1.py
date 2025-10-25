import os
import sys
import yaml
import numpy as np
import pandas as pd
import statsmodels.api as sm

# -- import your replicate() helper
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# ==============================
# CONFIG & PATHS
# ==============================
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

CSV_PATH = os.path.join(INPUT_DATA_DIR, '162', 'asipanel2.csv')
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

# ==============================
# LOAD
# ==============================
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"[load] asipanel2.csv -> {df.shape[0]:,} rows, {df.shape[1]} cols")

need = ['id','year','logoutput','T2','T3','T4','T5','rainfall']
missing = [c for c in need if c not in df.columns]
if missing:
    raise KeyError(f"Missing expected columns: {missing}")

df = df[need].dropna().copy()
# Enforce numeric types (CSV can introduce strings)
num_cols = ['logoutput','T2','T3','T4','T5','rainfall']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=num_cols)

# ==============================
# DESIGN MATRIX (X) with FE inside (all float64)
# ==============================
# Core regressors
X = df[['T2','T3','T4','T5','rainfall']].astype('float64').rename(columns={
    'T2': 'T20_25',
    'T3': 'T25_30',
    'T4': 'T30_35',
    'T5': 'T35_50',
    'rainfall': 'rain_avg'
})
X = sm.add_constant(X).astype('float64')

# FE codes appended INSIDE X (as float64)
fe_plant = pd.Categorical(df['id']).codes.astype('float64')
fe_year  = pd.Categorical(df['year']).codes.astype('float64')
X['fe_plant'] = fe_plant
X['fe_year']  = fe_year

# FE positional indices inside X (IMPORTANT)
fe_indices = [X.columns.get_loc('fe_plant'), X.columns.get_loc('fe_year')]

# DV in levels (replicate() logs internally), as float64
y = np.exp(df['logoutput'].astype('float64').values).astype('float64')

# ==============================
# METADATA
# ==============================
metadata = {
    'paper_id': '162',
    'table_id': '4',
    'panel_identifier': '1',
    'model_type': 'log-linear',
    'comments': ('Table 4, Col 1 — coef-only. DV=log(output) [levels passed]. '
                 'X = T2..T5 + rainfall; FE passed as indices inside X; '
                 'all columns forced to float64; auto var-type detection disabled.')
}

# ==============================
# DISABLE AUTO TYPE DETECTION IN ESTIMATOR (avoids dtype issues)
# ==============================
kwargs_estimator = {
    'binary_vars': [],
    'categorical_vars': [],
    'ordinal_vars': []
}

# ==============================
# RUN replicate() — no SEs (coef-only)
# ==============================
_ = replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='T35_50',                 # highlight top bin; not essential for coef-only
    fe=fe_indices,                     # FE as column indices inside X
    elasticity=False,
    #kwargs_estimator=kwargs_estimator, # disable auto detection
    # no kwargs_ols → no SEs
    #output=True,
    #output_dir=OUTPUT_DIR,
    #replicated=True
)