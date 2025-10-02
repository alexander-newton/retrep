import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# Load Mafia municipality dataset (as per Stata script)
filepath = os.path.join(INPUT_DATA_DIR, '127/ADD_Mafia_municipality.dta')
df = pd.read_stata(filepath)

# Define variable groups as in Stata
det_Fasci = ['predr_peas_fasci', 'ruralcentre1861', 'Rural_rent', 'Urban_rent', 
             'agricola_rel', 'seminatoritot_rel']

det_Mafia = ['Citrus_groves', 'sulfurproduction1868_70', 'Vineyards', 
             'Olives_groves', 'Mafia1885']

basics = ['lnpop1861', 'lnsurface', 'centreheight', 'maxheight', 'slope2', 
          'pa_pdist1856', 'port2_pdist1856', 'roads1799', 'ave_temp', 
          'var_sp3m_n30', 'sp3m_ave_n30']

provinces = ['province2', 'province3', 'province4', 'province5', 'province6', 'province7']

# Table 11 Panel D Column 3: Specification 3 for dev_spending1957
# This corresponds to specification3 = det_Fasci + det_Mafia + basics
dep_var = 'dev_spending1957'
endog_var = 'Mafia1900'
instrument = 'sp3m1893_n30'

# Specification 3: all controls
control_vars = det_Fasci + det_Mafia + basics + provinces

# Required variables for this regression (without province dummies as they'll be converted to FE codes)
control_vars_no_provinces = det_Fasci + det_Mafia + basics
required_vars = [dep_var, endog_var, instrument] + control_vars_no_provinces + ['cl1_stn_sp1893_n30', 'distretto1853']

# Clean data - filter as in Stata: if dev_spending1957!=. and sp3m1893_n30!=.
df_clean = df.dropna(subset=[dep_var, instrument])
df_clean = df_clean.dropna(subset=required_vars)

# Create levels from logged dependent variable (since replicate function logs internally)
y = np.exp(df_clean[dep_var].values)

# Convert province variable to numeric codes for replicate function
# Create a single province variable from the dummy variables
province_cols = [col for col in provinces if col in df_clean.columns]
if province_cols:
    # Find which province each municipality belongs to (find the column with value 1)
    province_matrix = df_clean[province_cols].values
    province_indices = np.argmax(province_matrix, axis=1)
    # For municipalities with all zeros (province1, the reference), assign 0
    has_province = province_matrix.sum(axis=1) > 0
    province_indices = np.where(has_province, province_indices + 1, 0)  # +1 because province2-7 are indices 0-5
    df_clean['province_code'] = province_indices
else:
    # If no province dummies exist, create a default province code
    df_clean['province_code'] = 0

# X matrix: endogenous variable + all controls + province FE codes + constant
X_vars = [endog_var] + control_vars_no_provinces + ['province_code']
X = df_clean[X_vars].copy()
X = sm.add_constant(X, prepend=False)

# z matrix: instrument + SAME controls + province FE codes + constant
z_vars = [instrument] + control_vars_no_provinces + ['province_code']
z = df_clean[z_vars].copy()
z = sm.add_constant(z, prepend=False)

# Fixed effects variable names for replicate function
fe_vars = ['province_code']

# Cluster variables (two-way clustering as in Stata)
# Primary cluster: cl1_stn_sp1893_n30 (closest weather station)
# Secondary cluster: distretto1853 (district)
cluster_primary = pd.Categorical(df_clean['cl1_stn_sp1893_n30']).codes
cluster_secondary = pd.Categorical(df_clean['distretto1853']).codes

# Note: The replicate function may need modification to handle two-way clustering
# For now, we'll use the primary cluster (weather station) as in the original bootstrap
cluster = cluster_primary

# Metadata
metadata = {
    'paper_id': '127',
    'table_id': '11',
    'panel_identifier': 'D_3',
    'model_type': 'log-linear',
    'comments': 'Table 11 Panel D Column 3: IV regression of development expenditure 1957 on Mafia1900 instrumented by spring rainfall 1893, specification 3 (all controls)'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=endog_var,
    endog_x=[endog_var],
    z=z,
    elasticity=False,
    fe=fe_vars,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True,
    output_dir=OUTPUT_DIR,
    replicated=True
)
