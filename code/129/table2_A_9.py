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

#==============================================================================
# LOAD PREPARED DATA FROM STATA
#==============================================================================

# Load the prepared dataset from Stata
filepath = os.path.join(INPUT_DATA_DIR, '129/table2_prepared_data.dta')
df = pd.read_stata(filepath)

#==============================================================================
# REGRESSION - Table 2 Panel A Column 9 (migrants9)
#==============================================================================

# Filter data for regression: period_start >= 1420, non-missing cityid and u1800
# Also drop observations with missing terr_id_1500alt (required for principality_x_year FE)
reg_data = df[
    (df['period_start'] >= 1420) & 
    (df['cityid'].notna()) & 
    (df['u1800'].notna()) &
    (df['terr_id_1500alt'].notna()) &
    (df['terr_id_1500alt'] != '')
].copy()

# The Stata prepared data already has log_migrants = log(d_migrants_b + 1)
# We need to use the exact same outcome variable as Stata

# Convert categorical variables to numeric codes for replicate function
reg_data['principality_x_year_code'] = pd.Categorical(reg_data['principality_x_year']).codes

# Use the exact outcome variable that Stata uses: log_migrants
# But since replicate expects levels and logs internally, convert back to levels
y = np.exp(reg_data['log_migrants'])

# Identify all city_trend_* columns dynamically
city_trend_cols = [col for col in reg_data.columns if col.startswith('city_trend_')]

# Define X matrix for Column 9: city-specific trends + principality-year FE (NO city FE)
# From table: "City FE: No", "City-Specific Trend: Yes", "Principality × Year FE: Yes"
X_vars = ['post', 'post_x_law', 'post_x_trend', 'laws_x_post_x_trend',
          'post_x_prot', 'post_x_prot_x_trend'] + city_trend_cols + ['principality_x_year_code']

# Clean data and create final matrices
reg_data = reg_data.dropna(subset=X_vars + ['log_migrants'])
y = np.exp(reg_data['log_migrants'])  # Convert log_migrants back to levels

X = reg_data[X_vars].copy()
X = sm.add_constant(X, prepend=False)

# Fixed effects variable names - only principality-year FE for Column 9
fe_vars = ['principality_x_year_code']

# Cluster variable
cluster = pd.Categorical(reg_data['cityid']).codes

# Metadata
metadata = {
    'paper_id': '129',
    'table_id': '2',
    'panel_identifier': 'A_9',
    'model_type': 'log-linear',  # Back to log-linear since we're giving levels
    'comments': 'Table 2 Panel A Column 9: OLS regression of log migrants on reformation laws with city trends and principality-year FE, 1420-1819'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='post_x_law',
    elasticity=False,
    fe=fe_vars,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True,
    output_dir=OUTPUT_DIR,
    replicated=True
)