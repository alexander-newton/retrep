import numpy as np
import os
import sys
import json
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from replication import replicate
from utils.json_utils import ReplicationJSONBuilder, load_config

# Load configuration
config = load_config()
raw_data_folder = config.get('rawdata', './rawdata')
output_folder = config.get('output', './output')

# =====================================
# Paper 184 - Finkelstein, Notowidigdo, Schilbach & Zhang (2025), "Lives vs. Livelihoods"
# Main event study (Figure III), 2007-2016 average coefficient
# Replication 184_31: Multiple imputation draw 19 of 40
# Main build with randomly perturbed age splits and tiny-county totals (seed 20261013).
# Run all 40 draws (184_13 to 184_52) and combine with Rubin's rules.
#
# areg ln_age_adj_mort_plus1 ib2006.year##c.shock [aweight=czpop_2006], absorb(cz) cluster(cz)
# DV: log(age-adjusted mortality per 100k + 1); shock: 2007-09 CZ unemployment shock (pp)
# Interest: shock_x_post = shock x 1(year >= 2007). With the controls below (shock_x_t minus
#   shock_x_2007 for 2008-2016, and shock_x_t for 2003-2005) this is an exact reparametrisation
#   of the authors' regression: the coefficient on shock_x_post equals the average of their
#   2007-2016 event-study coefficients (the paper's headline number), and the fit is identical.
# Published 2007-16 (x100): -0.558 (SE 0.279)
# This data (weighted OLS, x100): -0.545 (SE 0.277)
# Closeness: Same sample as 184_1; spread across draws = imputation uncertainty.
# =====================================

# Load data
df = pd.read_stata(os.path.join(raw_data_folder, '184', '184_31_mi_draw19.dta'))

# Sample restriction: 2003-2016, CZs observed in every year (as the authors)
df = df[(df['year'] >= 2003) & (df['year'] <= 2016)].copy()
df = df[df.groupby('cz')['year'].transform('count') == 14].copy()
df = df.sort_values(['cz', 'year']).reset_index(drop=True)

# DV in levels: age-adjusted mortality per 100k + 1 (log of this is the authors' DV)
df['age_adj_mort_plus1'] = df['age_adj_mort'] + 1

# Interest: shock x post (2007-2016)
df['shock_x_post'] = df['shock'] * (df['year'] >= 2007)

# Controls: shock_x_t - shock_x_2007 for t = 2008-2016; shock_x_t for t = 2003-2005 (base 2006)
controls = []
for t in range(2008, 2017):
    df[f'shock_x_{t}_minus_2007'] = df['shock'] * (df['year'] == t) - df['shock'] * (df['year'] == 2007)
    controls.append(f'shock_x_{t}_minus_2007')
for t in range(2003, 2006):
    df[f'shock_x_{t}'] = df['shock'] * (df['year'] == t)
    controls.append(f'shock_x_{t}')

# Year FE
year_dummies = pd.get_dummies(df['year'].astype(int), prefix='yr', drop_first=True).astype(float)
year_cols = list(year_dummies.columns)
df = pd.concat([df.reset_index(drop=True), year_dummies.reset_index(drop=True)], axis=1)

# CZ FE (absorbs the shock main effect)
df['cz'] = df['cz'].astype(int)

# Drop missing
sample_vars = ['age_adj_mort_plus1', 'shock_x_post'] + controls + year_cols + ['cz', 'czpop_2006']
df = df.dropna(subset=sample_vars).reset_index(drop=True)

# =====================================
# Replication 184_31: mi_draw19
# =====================================

metadata = {
    'paper_id': '184',
    'table_id': 'FigureIII',
    'panel_identifier': 'mi_draw19',
    'model_type': 'log-linear'
}

# y in levels (age-adjusted mortality per 100k + 1)
y = pd.DataFrame(df['age_adj_mort_plus1'].values, columns=['age_adj_mort_plus1'])

# X: interest variable first (shock_x_post), then controls, year FE, CZ FE, weights (czpop_2006) last
x_cols = ['shock_x_post'] + controls + year_cols + ['cz']
x_cols = x_cols + ['czpop_2006']
X = df[x_cols].reset_index(drop=True)

# Cluster variable: CZ
cluster_col = df['cz'].values

# Save to parquet
out_dir = os.path.join(output_folder, '184', '184_31')
os.makedirs(out_dir, exist_ok=True)

y.to_parquet(os.path.join(out_dir, 'y.parquet'), index=False)
X.to_parquet(os.path.join(out_dir, 'X.parquet'), index=False)

with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

# replicate()
replicate(
    metadata=metadata,
    y=y.values,
    X=X[['shock_x_post'] + controls].values,
    interest='shock_x_post',
    endog_x=None,
    z=None,
    fe=X[year_cols + ['cz']].values,
    elasticity=True,
    replicated=True,
    kwargs_estimator={'estimator_type': 'ols'},
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col}},
    kwargs_ppml=None,
    fit_full_model=False,
    output=True,
    output_dir=output_folder
)
