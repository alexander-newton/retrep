import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '009/regdata_3_yes_.33_.06.dta')
df = pd.read_stata(filepath)

# -------- Ensure needed columns present --------
cols_needed = [
    'c3diffln_gvapc',      # DV (log form; we feed levels so replicate() logs internally)
    'diffln_gvapc',        # lagged DV
    'inf_gva',             # key regressor (espionage)
    'diff_patents_gva',    # patent gap (patents)
    'year', 'branch',      # fixed effects (categorical variables)
    'weight_workers'       # analytic weights
]

df = df.dropna(subset=cols_needed).copy()

# -------- DV in LEVELS --------
y = np.exp(df['c3diffln_gvapc'])

# -------- Regressors: main vars + FE --------
X_vars = ['inf_gva', 'diff_patents_gva', 'diffln_gvapc', 'year', 'branch']
X = sm.add_constant(df[X_vars], prepend=False)

# -------- Clustering, weights, FE list --------
clusters = df['branch']
weights = df['weight_workers']
fes = ['year', 'branch']   # corresponds to yd_* and br_* in Stata

# -------- Metadata --------
metadata = {
    'paper_id': '009',
    'table_id': '2',
    'panel_identifier': '6',
    'model_type': 'log-linear',
    'comments': 'Table 2, Col (6): Δln(gvapc gap) on espionage(inf_gva) + patents(diff_patents_gva) + lagged Δln_gvapc with year & branch FE, clustered by branch, [aw=weight_workers]. Replicates: reg c3diffln_gvapc espionage patents diffln_gvapc yd_* br_* [aw=weight_workers], cluster(branch)'
}

# -------- Run replication --------
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='inf_gva',
    elasticity=False,
    fe=fes,
    weights=weights,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': clusters}},
    # output=True, output_dir=OUTPUT_DIR, replicated=True
)
