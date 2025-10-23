import yaml, os, pandas as pd, numpy as np, sys, statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# ---- Paths
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)
OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# ---- Load data
df = pd.read_stata(os.path.join(INPUT_DATA_DIR, '175/sr_ports_final.dta'))

# ---- Dependent variable (levels; replicate() logs internally)
y = np.exp(df['lnships'])

# ---- Fixed effects
df['fe_ctrypost'] = pd.Categorical(df['post_countryoid']).codes
df['fe_port'] = pd.Categorical(df['port_id']).codes

# ---- Design matrix: same as col 4 but *without* post_capital etc.
X = df[[
    'post_fail_lon',          # interest
    'indic_ot',               # additional regressor
    'event_og', 'post_lon', 'indic_p',  # $port_fe
    'fe_ctrypost', 'fe_port'            # FE placeholders
]].copy()
X = sm.add_constant(X, prepend=False)

# ---- FE indices
fe_vars = [X.columns.get_loc('fe_ctrypost'), X.columns.get_loc('fe_port')]

# ---- Filter, weights, cluster
mask = y.notna() & X['post_fail_lon'].notna()
y = y[mask]; X = X.loc[mask].copy()

# Analytic weights (Stata: [aweight=port_size])
aw = df.loc[mask, 'port_size'].astype(float).to_numpy()

# One-way clustering at country level (Stata: cluster(countryoid))
cluster_groups = df.loc[mask, 'countryoid'].to_numpy()

# ---- Metadata
metadata = {
    'paper_id': '175',
    'table_id': '4',
    'panel_identifier': '3',
    'model_type': 'log-linear',
    'comments': 'Xu (2022) QJE — Table IV Col 3: ln(ships) on post_fail_lon, event_og, post_lon, indic_p; '
                'aweights=port_size; FE=(port_id, country×post); cluster(countryoid).'
}

# ---- Run
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='post_fail_lon',
    elasticity=False,
    fe=fe_vars,
    weights=aw,
    kwargs_ols = {'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_groups}},
    output=True, output_dir=OUTPUT_DIR, replicated=True, overwrite=True,
)
