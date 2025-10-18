import os, sys, yaml
import numpy as np
import pandas as pd
import statsmodels.api as sm

# import replicate()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# -----------------------------
# Config & paths
# -----------------------------
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)
OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# -----------------------------
# Load cleaned CPS dataset (folder 168)
# -----------------------------
data_path = os.path.join(INPUT_DATA_DIR, '168/data_cleaned_new.dta')
df = pd.read_stata(data_path)

# --- enforce preferred sample (drop universally licensed occs) ---
df.loc[df['gt_statecert'] == 1, 'sample'] = 0
df = df[df['sample'] == 1].copy()


# --- CPS window & basic filters (available vars only) ---
df = df[df['year'].between(2015, 2018)]
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df = df[(df['age'] >= 16) & (df['age'] <= 64)]

# -----------------------------
# Table 2 • Panel C • Col (3)
# -----------------------------
req = ['lwage','sh_statec','strata','statefip','occ','ind','my','state_occ','earnwt']
df = df.dropna(subset=[c for c in req if c in df.columns]).copy()

# Keep MORG wage subsample & clean weights
df = df[df['earnwt'] > 0].copy()
df['earnwt'] = df['earnwt'].astype(float).replace([np.inf, -np.inf], np.nan)
df = df[df['earnwt'].notna()].copy()

# Encode only state FEs (others already numeric)
df['statefip_fe'] = df['statefip'].astype('category').cat.codes.astype(np.int32)

# FE set and zero-weight FE-group drop
fe_cols = ['strata','statefip_fe','occ','ind','my']
grp_w = df.groupby(fe_cols, dropna=False)['earnwt'].transform('sum')
df = df[grp_w > 0].copy()

# Drop singleton FE groups to mimic reghdfe behavior (helps match N)
for fe in fe_cols:
    cnt = df.groupby(fe)[fe].transform('count')
    df = df[cnt > 1].copy()

# Normalize weights (optional, numerical stability)
m = df['earnwt'].mean()
if np.isfinite(m) and m > 0:
    df['earnwt'] = df['earnwt'] / m

# DV in levels (replicate() logs internally)
y = np.exp(df['lwage'])

# Build X and FE indices
X = df[['sh_statec'] + fe_cols].copy()
X = sm.add_constant(X, prepend=False)
fe_indices = [X.columns.get_loc(c) for c in fe_cols]

# Cluster + weights
cluster = df['state_occ']
weights = df['earnwt']

# Metadata
metadata = {
    'paper_id': '168',
    'table_id': '2',
    'panel_identifier': 'C_3',
    'model_type': 'log-linear',
    'comments': 'Table 2 Panel C Col 3: log hourly wage on % licensed; strata, state, occ, ind, month FE; cluster=state_occ; aw=earnwt'
}


#print("Diag — N:", len(df),
#      "| clusters(state_occ):", df['state_occ'].nunique(),
#      "| gt==1 (dropped):", int((df['gt'] == 1).sum() if 'gt' in df.columns else -1))

# Run
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='sh_statec',
    fe=fe_indices,
    weights=weights,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    elasticity=False,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
