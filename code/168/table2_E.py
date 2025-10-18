import os, sys, yaml
import numpy as np
import pandas as pd
import statsmodels.api as sm

# import replicate()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# ==============================
#  CONFIG & PATHS
# ==============================
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)
OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

data_path = os.path.join(INPUT_DATA_DIR, '168/data_cleaned_new.dta')
df = pd.read_stata(data_path)

# ==============================
#  SAMPLE: drop universally licensed occupations (gt_statecert == 1)
# ==============================
df.loc[df['gt_statecert'] == 1, 'sample'] = 0
df = df[df['sample'] == 1].copy()

# ==============================
#  PANEL E SETUP
#  Collapse to state–occupation level:
#  gcollapse (mean) sh_statec (rawsum) wtfinl [aw=wtfinl], by(statefip occ state_occ)
# ==============================
g = (
    df[['sh_statec', 'wtfinl', 'statefip', 'occ', 'state_occ', 'gt_statecert']]
      .dropna(subset=['sh_statec', 'wtfinl', 'statefip', 'occ', 'state_occ'])
      .groupby(['statefip', 'occ', 'state_occ'], as_index=False, observed=True)
      .agg({'sh_statec': 'mean', 'wtfinl': 'sum'})
)


# Drop zero or negative employment (as in Stata’s “drop if wtfinl <= 0”)
g = g[g['wtfinl'] > 0].copy()

# ==============================
#  FIXED EFFECTS (state & occupation only)
# ==============================
g['statefip_fe'] = g['statefip'].astype('category').cat.codes.astype(np.int32)
fe_cols = ['statefip_fe', 'occ']

# ==============================
#  REGRESSION VARIABLES
# ==============================
# DV: employment level (replicate logs internally)
y = g['wtfinl']
X = sm.add_constant(g[['sh_statec'] + fe_cols], prepend=False)
fe_indices = [X.columns.get_loc(c) for c in fe_cols]

# Cluster and analytic weights
cluster = g['state_occ']
weights = g['wtfinl']

# ==============================
#  METADATA & RUN
# ==============================
metadata = {
    'paper_id': '168',
    'table_id': '2',
    'panel_identifier': 'E',
    'model_type': 'log-linear',
    'comments': 'Table 2 Panel E: log employment on % licensed; FE = state + occupation; cluster = state_occ; aw = wtfinl'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='sh_statec',
    fe=fe_indices,
    weights=weights,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    elasticity=False,
    #output=True, output_dir=OUTPUT_DIR, replicated=True
)
