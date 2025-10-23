import yaml, os, pandas as pd, numpy as np, sys, statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Paths
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)
OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# Data: final port-level panel for Table IV
df = pd.read_stata(os.path.join(INPUT_DATA_DIR, '175/sr_ports_final.dta'))

# DV in levels (replicate() logs internally)
y = np.exp(df['lnships'])

# Fixed effects (Port FE; Country×Post FE)
df['fe_port'] = pd.Categorical(df['port_id']).codes
df['fe_ctrypost'] = pd.Categorical(
    df['countryoid'].astype(str) + "_P" + df['event_og'].astype(int).astype(str)
).codes

# Design matrix
X = df[[
    'post_fail_lon',          # Fail_po × post (coef of interest)
    'indic_ot',               # included as regressor in do-file
    'event_og', 'post_lon', 'indic_p',
    'post_capital', 'post_frac_uk', 'post_avg_age', 'post_avg_og',
    'fe_port', 'fe_ctrypost'  # FE placeholders
]].copy()
X = sm.add_constant(X, prepend=False)

# FE indices
fe_vars = [X.columns.get_loc('fe_port'), X.columns.get_loc('fe_ctrypost')]

# Keep complete cases for DV & interest
mask = y.notna() & X['post_fail_lon'].notna()
y = y[mask]
X = X.loc[mask].copy()

# Analytic weights (Stata: [aweight=port_size])
aw = df.loc[mask, 'port_size'].astype(float).to_numpy()

# One-way clustering at country level (Stata: cluster(countryoid))
cluster_groups = df.loc[mask, 'countryoid'].to_numpy()


metadata = {
    'paper_id': '175',
    'table_id': '4',
    'panel_identifier': 'short_run_ports',
    'model_type': 'log-linear ',
    'comments': 'Xu (2022) QJE — Table IV, Col 4: ln(ships) on post_fail_lon with Port FE & Country×Post FE; aweights=port_size; cluster by countryoid.'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='post_fail_lon',
    elasticity=False,
    fe=fe_vars,
    weights=aw,
    kwargs_ols = {'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_groups}},
    #output=True, output_dir=OUTPUT_DIR, replicated=True, overwrite=True,
)
# we just add some more controls but we dont get exact results!

