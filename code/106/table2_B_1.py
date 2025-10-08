import yaml
import os
import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '106/socialmedia_data_main.dta')
df = pd.read_stata(filepath)

# =====================================
# Table 2 Panel B Column 1: IV regression
# Dependent variable: logprot (log protests)
# Endogenous variable: logvk (VK penetration)
# Instrument: logspbsu2 (SPbSU cohort 2)
# Additional regressors: logspbsu1, logspbsu3 (other SPbSU cohorts, excluding Durov)
# Control set: control4 variables
# Clustering: okato_reg
# =====================================

# Define control variable groups as per Stata script
control_pop = ['pop', 'pop_2', 'pop_3', 'pop_4', 'pop_5']

control1 = control_pop

control2 = control_pop + ['ac']

control3 = control2 + [
    'disttopiter', 'disttomoscow', 'city2', 'logwage11', 
    'lA20_24_y2010', 'lA25_29_y2010', 'lA30_34_y2010', 'lA35_39_y2010', 
    'lA40_44_y2010', 'lA45_49_y2010', 'lA50_more_y2010',
    'he2010_A20_24', 'he2010_A25_29', 'he2010_A30_34', 'he2010_A35_39', 
    'he2010_A40_44', 'he2010_A45_49', 'he2010_A50_more'
]

control4 = control3 + ['University_exists', 'ip_2011', 'logok2014', 'ef2010', 'educ2002']

# For Table 2 Panel B Column 1, we use control4
control_vars = control4

# Other SPbSU cohorts (excluding Durov cohort)
spbsu_cohorts_no_durov = ['logspbsu1', 'logspbsu3']

# Define all required variables for this regression
required_vars = ['logprot', 'logvk', 'logspbsu2'] + spbsu_cohorts_no_durov + control_vars + ['okato_reg']

# Drop missing values for required variables
df_clean = df.dropna(subset=required_vars)

print(f"Sample size after dropping missing values: {len(df_clean)}")

# Dependent variable:
y = np.exp(df_clean['logprot'])

# Endogenous variable: VK penetration (logvk)
endog_x = ['logvk']

# X matrix: endogenous variable + other SPbSU cohorts + control variables
X_vars = endog_x + spbsu_cohorts_no_durov + control_vars
X = df_clean[X_vars].copy()

# Add constant
X = sm.add_constant(X, prepend=False)

# Instruments matrix: SPbSU cohort 2 + other SPbSU cohorts + control variables  
# (the instrument replaces the endogenous variable)
z_vars = ['logspbsu2'] + spbsu_cohorts_no_durov + control_vars
z = df_clean[z_vars].copy()
z = sm.add_constant(z, prepend=False)

# Cluster variable
cluster = df_clean['okato_reg']

# Weights (aweight in Stata corresponds to frequency weights)
weights = df_clean['logpop']  
# METADATA
metadata = {
    'paper_id': '106',
    'table_id': '2',
    'panel_identifier': 'B_1',
    'model_type': 'log-log',
    'comments': 'Table 2 Panel B Column 1: 2SLS of log protests on VK penetration (logvk), instrumented by SPbSU cohort 2 (logspbsu2)'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata,
    y=y,  
    X=X,
    interest='logvk',  # Variable of interest
    endog_x=endog_x,  # Endogenous variables
    z=z,  # Instruments matrix
    elasticity=True,
    weights=weights,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    # output=True, 
    # output_dir=OUTPUT_DIR, 
    # replicated=True
)
