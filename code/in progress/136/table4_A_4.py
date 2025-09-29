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
# TABLE 4 PANEL A COLUMN 4: LONG DIFFERENCES
# =====================================

# Panel A: Long differences - only 1940 and 1980 with newsample40==1
df_panel = df[((df['year'] == 1940) | (df['year'] == 1980)) & (df['newsample40'] == 1)].copy()

# Drop missing values for required variables
required_vars = ['logdeathpop40U', 'logmaddpop', 'compsjmhatit', 'yr1940', 'yr1980', 'ctrycluster']
df_clean = df_panel.dropna(subset=required_vars).copy()

# =====================================
# PREPARE DATA FOR REPLICATION
# =====================================

# y: Convert from log to levels (replicate function will take log internally)
y = np.exp(df_clean['logdeathpop40U'].values)

# Define variables
endog_var = 'logmaddpop'
instrument = 'compsjmhatit'
control_vars = ['yr1940', 'yr1980', 'ctrycluster']

# X matrix: endogenous variable + controls + constant
X = df_clean[[endog_var] + control_vars].copy()
X = sm.add_constant(X, prepend=False)

# z matrix: instrument + SAME controls + constant
z = df_clean[[instrument] + control_vars].copy()
z = sm.add_constant(z, prepend=False)

# Clustering variable
cluster_var = df_clean['ctrycluster'].values

# METADATA
metadata = {
    'paper_id': '136',
    'table_id': '4',
    'panel_identifier': 'A_4',
    'model_type': 'log-log',
    'comments': 'Table 4 Panel A Column 4: Long differences 2SLS, log(1+battle deaths/pop. 1940) on log population, clustered SEs'
}

# Run the replication
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=endog_var,
    endog_x=[endog_var],
    z=z,
    fe=['ctrycluster'],
    elasticity=True,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_var}},

)