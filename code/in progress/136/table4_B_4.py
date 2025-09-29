import yaml
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '136/data with conflict.dta')
df = pd.read_stata(filepath)

# =====================================
# TABLE 4 PANEL B COLUMN 4: PANEL REGRESSIONS
# =====================================

# Panel B: Panel regressions, 1940s-1980s (unbalanced panels with one observation per decade)
df_panel = df[(df['year'] > 1930) & (df['year'] < 1990)].copy()

# Drop missing values for required variables
required_vars = ['logdeathpop40U', 'logmaddpop', 'compsjmhatit', 'ctrycluster']
df_clean = df_panel.dropna(subset=required_vars).copy()

# =====================================
# PREPARE DATA FOR REPLICATION
# =====================================

# y: Convert from log to levels (replicate function will take log internally)
y = np.exp(df_clean['logdeathpop40U'].values)

# Define variables
endog_var = 'logmaddpop'
instrument = 'compsjmhatit'

# Create year dummies for all years in Panel B (1940, 1950, 1960, 1970, 1980)
# Drop one year dummy to avoid perfect multicollinearity (drop 1940 as reference)
year_dummies = pd.get_dummies(df_clean['year'], prefix='yr')
year_cols = [col for col in year_dummies.columns if col in ['yr_1940.0', 'yr_1950.0', 'yr_1960.0', 'yr_1970.0', 'yr_1980.0']]
year_dummies = year_dummies[year_cols].astype(float)

# X matrix: endogenous variable + year dummies + fixed effects + constant
X = df_clean[[endog_var]].copy()
X = pd.concat([X, year_dummies], axis=1)
X = pd.concat([X, df_clean[['ctrycluster']]], axis=1)
X = sm.add_constant(X, prepend=False)

# z matrix: instrument + SAME year dummies + SAME fixed effects + constant
z = df_clean[[instrument]].copy()
z = pd.concat([z, year_dummies], axis=1)
z = pd.concat([z, df_clean[['ctrycluster']]], axis=1)
z = sm.add_constant(z, prepend=False)

# Fixed effects variables (country FE)
fe_vars = ['ctrycluster']

# Clustering variable
cluster_var = df_clean['ctrycluster'].values

# METADATA
metadata = {
    'paper_id': '136',
    'table_id': '4',
    'panel_identifier': 'B_4',
    'model_type': 'log-log',
    'comments': 'Table 4 Panel B Column 4: Panel regressions 2SLS, log(1+battle deaths/pop. 1940) on log population, clustered SEs'
}

# Run the replication
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=endog_var,
    endog_x=[endog_var],
    z=z,
    fe=fe_vars,
    elasticity=True,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_var}},

)
