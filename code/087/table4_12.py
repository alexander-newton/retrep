import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
from replication import replicate

# ---------- CONFIG ----------
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# ---------- LOAD DATA ----------
# Stata code imports: "../output/CleanData/paneldata_July2017.txt"
# It is tab-delimited with ISO-8859-1 encoding.
filepath = os.path.join(INPUT_DATA_DIR,'087/paneldata_July2017.txt')
df = pd.read_csv(filepath, sep='\t', encoding='ISO-8859-1')

# ---------- MATCH THE STATA TRANSFORMS ----------
# Stata destrings v* then renames; we mirror the final names:
rename_map = {
    'v1':'dates','v2':'permno','v3':'logrv','v4':'logK','v5':'logKAT','v6':'logKSA',
    'v7':'laglogK','v8':'laglogKAT','v9':'laglogKSA','v10':'logage','v11':'leverage',
    'v12':'concent','v13':'rdfrac','v14':'industry','v15':'herfout','v16':'herfin',
    'v17':'logME','v18':'logAT','v19':'logSA','v20':'laglogME','v21':'laglogAT','v22':'laglogSA'
}
df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})

# Coerce numerics like Stata's destring
for col in rename_map.values():
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Year filter: 1980–2012 inclusive
df = df[(df['dates'] >= 1980) & (df['dates'] <= 2012)]

# Construct logs as in .do
if 'logherfout' not in df.columns:
    df['logherfout'] = np.log(df['herfout'])

# Column 12 uses lagged log size = laglogK in their code
df['laglogsize'] = df['laglogK']

# Minimal NA drop for the variables used in col (12)
need = ['logrv', 'laglogsize', 'logherfout', 'industry', 'dates']
df = df.dropna(subset=need).copy()

# ---------- SPEC: Table 4, Column 12 ----------
# DV
y_col12 = np.exp(df['logrv'])

# Main regressors: log size (lagged) and log H_out
x_vars = ['laglogsize', 'logherfout']

# Industry FE (absorbed), cluster by year (dates)
fe_list = ['industry']
cluster_col = df['dates']  # cluster at "dates" like vce(cluster dates)

# X matrix: treatment vars first, constant added inside replicate (or here if needed)
X_col12 = sm.add_constant(df[x_vars], prepend=False)

# ---------- METADATA ----------
metadata_col12 = {
    'paper_id': '087',                 # your internal id for this paper
    'table_id': '4',
    'panel_identifier': '12',
    'model_type': 'log-log',
    'comments': (
        'Table 4 Column 12: log(firm volatility) on log size (lag) and log H_out; '
        'industry FE absorbed; SE clustered by year (dates); sample 1980–2012.'
    )
}

# ---------- RUN ----------
replicate(
    metadata=metadata_col12,
    y=y_col12,
    X=X_col12,
    interest=['laglogsize', 'logherfout'],
    elasticity=True,                   # log–log => elasticities
    fe=fe_list,                        # absorb industry FE
    #kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col}},
   # output=True, output_dir=OUTPUT_DIR, replicated=True
)
