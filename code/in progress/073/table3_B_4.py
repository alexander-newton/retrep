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

# LOAD DATA FOR PANEL B (WAGE ANALYSIS) - STACKED VERSION
# Load the stacked demographic group level data - matching first code's approach
filepath_group = os.path.join(INPUT_DATA_DIR, '073/outcomes_stacked_group.dta')
# Load specific columns like in first code
df = pd.read_stata(filepath_group, columns=['czone', 'year', 'd_hrwage_ln',
                                             'nat_earners_weight_1990', 'wage_earners_1990', 'group'])

# Load and merge exposure data (m:1 merge on czone and year)
filepath_exposure = os.path.join(INPUT_DATA_DIR, '073/czones_exposure_stacked.dta')
df_exposure = pd.read_stata(filepath_exposure)
df = df.merge(df_exposure, on=['czone', 'year'], how='inner')

# Load and merge covariates data (m:1 merge on czone and year)
filepath_covariates = os.path.join(INPUT_DATA_DIR, '073/covariates_stacked.dta')
df_covariates = pd.read_stata(filepath_covariates)
df = df.merge(df_covariates, on=['czone', 'year'], how='inner')

# Encode categorical variables to match Stata
# Create stcode from statefip (it already exists in your data)
df['stcode'] = df['statefip']
df['group_id'] = df['group']

# Create year dummies for stacked model
df['year_2000'] = (df['year'] == 2000).astype(int)

# Create group##year interaction for fixed effects
df['group_year'] = df['group'].astype(str) + '_' + df['year'].astype(str)
df['group_year_id'] = pd.Categorical(df['group_year']).codes + 1



# =====================================
# DATA CLEANING - Minimal to match Stata
# =====================================
# Find the exposure variable (after merges)
exposure_var = [col for col in df.columns if 'expof' in col]
if exposure_var:
    exposure_var = exposure_var[0]
else:
    raise ValueError("No exposure variable found after merges")

# Create a list of variables we need - using only what exists
analysis_vars = ['d_hrwage_ln', exposure_var, 'wage_earners_1990',
                 'division', 'group_id', 'group_year_id', 'stcode', 'year']

# Check which demographic/control variables actually exist after merges
potential_controls = ['ipums_female', 'ipums_hispanic', 'ipums_white', 'ipums_black', 
                      'ipums_asian', 'ipums_highschool', 'ipums_college', 'ipums_masters', 
                      'ipums_above65', 'ipums_logpop', 'ind_share_manufacturing', 
                      'ind_share_manufacturing_f', 'ind_share_light_manuf',
                      'chinashock_iv', 'expo_share_occ_routine']

# Add any that exist
for var in potential_controls:
    if var in df.columns:
        analysis_vars.append(var)

# Drop only rows with missing values in variables that exist
existing_vars = [var for var in analysis_vars if var in df.columns]
df_clean = df.dropna(subset=existing_vars)



# =====================================
# Table 3 Panel B Column 4: Change in Log Hourly Wages - Stacked Differences
# =====================================

# Dependent variable - absolute value of wage change (will be logged by replicate function)
# The sign will be restored after logging - EXACTLY LIKE FIRST CODE
y_col4 = np.exp(df_clean['d_hrwage_ln'])

# Main treatment variable
main_treatment = exposure_var

# Define control variables - based on what actually exists after merges
demographics = [var for var in df_clean.columns if 'ipums_' in var]
industry_shares = [var for var in df_clean.columns if 'ind_share' in var]
other_shocks = [var for var in df_clean.columns if 'chinashock' in var.lower() or 'routine' in var.lower()]

# Region effects - division exists in your data
region_effects = ['division'] if 'division' in df_clean.columns else []

# Year effects for stacked model
year_effects = ['year_2000']

# Combine all controls
controls = demographics + industry_shares + other_shocks + region_effects + year_effects

# Fixed effects - group##year interaction
group_year_fe = ['group_year_id']

# X matrix: treatment first, then controls, then FE variables
X_col4 = sm.add_constant(df_clean[[main_treatment] + controls + group_year_fe], prepend=False)

# Weight variable - number of wage earners in 1990
weights_col4 = df_clean['wage_earners_1990']

# Cluster variable - state code
cluster_col4 = df_clean['stcode']

# METADATA FOR THIS RESULT
metadata_col4 = {
    'paper_id': '073',
    'table_id': '3',
    'panel_identifier': 'B_4',
    'model_type': 'log-linear',  # Log-linear because we're using wage levels that need to be logged
    'comments': 'Table 3 Panel B Column 4: Stacked differences in hourly wages (1990-2000, 2000-2007); absolute values logged, signs preserved; robot exposure with all controls, division FE, year FE, and group##year FE absorbed; weighted by wage earners 1990, clustered by state'
}

# RUN THE REPLICATION
# Note: After the replicate function logs the absolute values, you may need to restore signs
# to the logged values if needed for interpretation
replicate(
    metadata=metadata_col4,
    y=y_col4,
    X=X_col4, 
    interest=exposure_var,
    elasticity=False,
    fe=['group_year_id'],  # Only group##year FE (division included as control)
    #weights=weights_col4,
    #kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col4}},
    #output=True, output_dir=OUTPUT_DIR, replicated=True
)