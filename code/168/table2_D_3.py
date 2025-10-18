import os, sys, yaml
import numpy as np
import pandas as pd
import statsmodels.api as sm

# import replicate()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# --- config/paths ---
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)
OUTPUT_DIR = config['outputdata']; INPUT_DATA_DIR = config['intermediatedata']

# --- load data ---
df = pd.read_stata(os.path.join(INPUT_DATA_DIR, '168/data_cleaned_new.dta'))

# --- enforce preferred sample (drop universally licensed occs) ---
df.loc[df['gt_statecert'] == 1, 'sample'] = 0
df = df[df['sample'] == 1].copy()

# --- CPS window & basic filters (available vars only) ---
df = df[df['year'].between(2015, 2018)]
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df = df[(df['age'] >= 16) & (df['age'] <= 64)]

# --- vars needed for Panel D Col (3) ---
req = ['lhours','sh_statec','strata','statefip','occ','ind','my','state_occ','wtfinl']
df = df.dropna(subset=req).copy()
df = df[df['wtfinl'] > 0].copy()

# --- FE coding (state only; others already numeric) ---
df['statefip_fe'] = df['statefip'].astype('category').cat.codes.astype(np.int32)
fe_cols = ['strata','statefip_fe','occ','ind','my']

# --- DV (replicate() logs internally), X, FE indices ---
y = np.exp(df['lhours'])
X = sm.add_constant(df[['sh_statec'] + fe_cols], prepend=False)
fe_indices = [X.columns.get_loc(c) for c in fe_cols]

# --- cluster & weights ---
cluster = df['state_occ']
weights = df['wtfinl']

# Metadata
metadata = {
    'paper_id': '168',
    'table_id': '2',
    'panel_identifier': 'D_3',
    'model_type': 'log-linear',
    'comments': 'Table 2 Panel D Col 3: log employment on % licensed with state & occupation FE; clustered by state_occ; analytic weights = wtfinl'
}


# Run
replicate(
    metadata=metadata,
    y=y, X=X, interest='sh_statec',
    fe=fe_indices,
    weights=weights,
    kwargs_ols={'cov_type':'cluster','cov_kwds':{'groups': cluster}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
