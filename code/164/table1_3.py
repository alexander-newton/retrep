# === Table 1, Col (3): reg glpop6914 ldistTOP4 if year==1914, robust ===
import os, sys, yaml
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)
OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# Load
filepath = os.path.join(INPUT_DATA_DIR, '164/Districts_1914_1895_1869_final.dta')
df = pd.read_stata(filepath)

# Keep only variables we need
cols_needed = ['dcode1895','year','lpop','distBA','distRS','distLP','distBH']

# Build ldistTOP4 = ln(min(distBA, distRS, distLP, distBH)), avoiding log(0)
distTOP4 = pd.concat([df['distBA'], df['distRS'], df['distLP'], df['distBH']], axis=1).min(axis=1)
distTOP4 = distTOP4.replace(0, np.nan)  # log(0) undefined → like Stata, becomes missing
df['ldistTOP4'] = np.log(distTOP4)

# Extract needed years
lpop_1869 = df.loc[df['year'] == 1869, ['dcode1895','lpop']].rename(columns={'lpop':'lpop1869'})
x_1914     = df.loc[df['year'] == 1914, ['dcode1895','lpop','ldistTOP4']].rename(columns={'lpop':'lpop1914'})

# Merge & create DV: glpop6914 = lpop1914 - lpop1869
g = x_1914.merge(lpop_1869, on='dcode1895', how='inner')
g['glpop6914'] = g['lpop1914'] - g['lpop1869']

# DROP any non-finite rows to avoid statsmodels MissingDataError
mask = np.isfinite(g['glpop6914']) & np.isfinite(g['ldistTOP4'])
g = g.loc[mask].copy()

# Prepare DV in LEVELS (replicate() logs internally): log(y) = glpop6914
y = np.exp(g['glpop6914'])

# X: ldistTOP4 + constant (no FE)
X = g[['ldistTOP4']].copy()
X = sm.add_constant(X, prepend=False)

# (Optional) sanity checks
assert np.isfinite(X.values).all(), "X still has NaN/inf"
assert np.isfinite(y.values).all(), "y has NaN/inf"

metadata = {
    'paper_id': '164',
    'table_id': '1',
    'panel_identifier': '3',
    'model_type': 'log-linear',
    'comments': 'Table 1, Col (3): glpop6914 on ldistTOP4, year=1914. Dropped rows with log(0) in distances; DV passed in levels.'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='ldistTOP4',
    fe=None,
    elasticity=False,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
