import yaml, os
import pandas as pd
import numpy as np
import statsmodels.api as sm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# -------------------------------
# Load config and dataset
# -------------------------------
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

data_path = os.path.join(INPUT_DATA_DIR, '136/data with conflict.dta')
df = pd.read_stata(data_path)

# -------------------------------
# Sample: panel spec (1930 < year < 1990)
# -------------------------------
df = df[(df['year'] > 1930) & (df['year'] < 1990)].copy()

# -------------------------------
# Variables
# DV: logdeathpop40U
# Endogenous regressor: logmaddpop
# Instrument: compsjmhatit
# FE: country + year
# Cluster: ctrycluster
# -------------------------------
dv_log = 'logdeathpop40U'
endog_var = 'logmaddpop'
instrument = 'compsjmhatit'

# Clean data - remove rows with missing values for key variables
required_vars = [dv_log, endog_var, instrument, 'country', 'ctrycluster']
df = df.dropna(subset=required_vars)

# Remove infinite values
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required_vars)

# Remove singleton groups to help with convergence
# Remove countries with only one observation
country_counts = df.groupby('country').size()
singleton_countries = country_counts[country_counts == 1].index
df = df[~df['country'].isin(singleton_countries)].copy()

# Remove years with only one observation
year_counts = df.groupby('year').size()
singleton_years = year_counts[year_counts == 1].index
df = df[~df['year'].isin(singleton_years)].copy()

print(f"After removing singletons: {len(df)} observations")
print(f"Countries: {df['country'].nunique()}, Years: {df['year'].nunique()}")

# y in levels (replicate() will log internally)
y = np.exp(df[dv_log].astype(float).values)

# FE: country + year (try with data preprocessing to help convergence)
df['country_fe'] = pd.Categorical(df['country']).codes
df['year_fe'] = pd.Categorical(df['year'].astype(int)).codes
fe_vars = ['country_fe', 'year_fe']

# Design matrices
X = df[[endog_var] + fe_vars].copy()
X = sm.add_constant(X, prepend=False)

Z = df[[instrument] + fe_vars].copy()
Z = sm.add_constant(Z, prepend=False)

# Cluster groups
cluster = pd.Categorical(df['ctrycluster']).codes

# -------------------------------
# Metadata
# -------------------------------
metadata = {
    'paper_id': '136',
    'table_id': '4',
    'panel_identifier': 'B_4',
    'model_type': 'log-linear',
    'comments': (
        'Table 4, Column 4 (panel spec). DV = log(1 + battle deaths / pop_1940), '
        'endog = logmaddpop, IV = compsjmhatit. Country FE + year FE, clustered by ctrycluster. '
        'Sample: years 1931–1989. Singleton groups removed to help convergence.'
    )
}

# -------------------------------
# Run replication
# -------------------------------
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=endog_var,
    endog_x=[endog_var],
    z=Z,
    elasticity=False,
    fe=fe_vars,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True,
    output_dir=OUTPUT_DIR,
    replicated=True
)
