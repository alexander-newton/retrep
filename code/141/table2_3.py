import yaml
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# ==============================
# CONFIG & PATHS
# ==============================
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# Authors’ control extract (used by GetResults.ado for FE spec)
filepath = os.path.join(INPUT_DATA_DIR, '141/FSctrl.dta')
df = pd.read_stata(filepath)

# ==============================
# SAMPLE RESTRICTIONS
# ==============================
df = df.loc[df['ctrl'] == 1].copy()   # like [pw=ctrl_wgt] if ctrl

# Drop IDs with fewer than 2 obs
df['__n'] = df.groupby('ID')['ID'].transform('size')
df = df.loc[df['__n'] >= 2].drop(columns='__n')

# Zero out new flags if no job change
g = df.groupby('ID')['jid']
df['tjob'] = (g.transform('max') != g.transform('min')).astype(int)
for c in ['new','new_ene','new_ee']:
    df.loc[df['tjob'] == 0, c] = 0

# ==============================
# VARIABLES
# ==============================
need = [
    'lhwage4','ur','hs','sc','cg','mc','union','nevermarried',
    'tenure','tensq','tt','new_ee','new_ene','ID','ctrl_wgt'
]
df = df[need].dropna()

# ==============================
# FIXED EFFECTS (within transform)
# ==============================
# Demean ur, then build interactions
df['ur_md'] = df['ur'] - df.groupby('ID')['ur'].transform('mean')
df['ur_md_new_ee']  = df['ur_md'] * df['new_ee']
df['ur_md_new_ene'] = df['ur_md'] * df['new_ene']

# Demean DV and controls
demean_vars = ['lhwage4','hs','sc','cg','mc','union','nevermarried','tenure','tensq','tt']
for v in demean_vars:
    df[v + '_md'] = df[v] - df.groupby('ID')[v].transform('mean')

# ==============================
# DESIGN MATRIX
# ==============================
# replicate() logs y internally → pass levels
y = np.exp(df['lhwage4_md'].astype(float).values)

X = df[[
    'ur_md','ur_md_new_ee','ur_md_new_ene',
    'hs_md','sc_md','cg_md','mc_md',
    'union_md','nevermarried_md','tenure_md','tensq_md','tt_md',
    'new_ee','new_ene'
]].astype(float)
X = sm.add_constant(X)

weights = df['ctrl_wgt'].astype(float).values
cluster = df['ID'].astype(str).values

# ==============================
# METADATA
# ==============================
metadata = {
    'paper_id': '141',
    'table_id': '2',
    'panel_identifier': '3',
    'model_type': 'log-linear',
    'comments': 'GHT Table 2, Column 3. FSctrl.dta with ctrl==1, FE by ID, '
                'zero new flags if no job change, ur interactions built after demeaning. '
                'DV = lhwage4 (levels passed). Interest = ur_md_new_ene.'
}

# ==============================
# RUN
# ==============================
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='ur_md_new_ene',
    elasticity=False,
    weights=weights,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True,
    output_dir=OUTPUT_DIR,
    replicated=True
)
