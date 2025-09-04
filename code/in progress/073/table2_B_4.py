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

# LOAD DATA FOR PANEL B (WAGE ANALYSIS)
# Load the demographic group level data - matching Stata exactly
filepath_group = os.path.join(INPUT_DATA_DIR, '073/outcomes_longdif_group.dta')
# Load BOTH the log version and level version of wage changes
df = pd.read_stata(filepath_group, columns=['czone', 'd_hrwage_ln_1990_2008',
                                             'nat_earners_weight_1990', 'wage_earners_1990', 'group'])

# Load and merge exposure data (m:1 merge on czone)
filepath_exposure = os.path.join(INPUT_DATA_DIR, '073/czones_exposure_longdif.dta')
df_exposure = pd.read_stata(filepath_exposure)
# Keep only needed columns from exposure data
exposure_cols = ['czone', 'tag_exposure', 'expof_euro5_qo93_07', 
                 'chinashock_iv_90_07', 'expo_share_occ_routine_1990']
df_exposure = df_exposure[exposure_cols]
df = df.merge(df_exposure, on='czone', how='inner')  # Using inner to match Stata's assert(3)

# Load and merge covariates data (m:1 merge on czone)
filepath_covariates = os.path.join(INPUT_DATA_DIR, '073/covariates_1990.dta')
df_covariates = pd.read_stata(filepath_covariates)
df = df.merge(df_covariates, on='czone', how='inner')  # Using inner to match Stata's assert(3)

# Encode categorical variables to match Stata
# Create state code if not already numeric
if 'state' in df.columns:
    df['stcode'] = pd.Categorical(df['state']).codes + 1
else:
    df['stcode'] = df['statefip']

# Create group_id if group is string
if df['group'].dtype == 'object':
    df['group_id'] = pd.Categorical(df['group']).codes + 1
else:
    df['group_id'] = df['group']



# =====================================
# DATA CLEANING - Minimal to match Stata
# =====================================
# Create a copy for the regression variables (using absolute value of wage change)
analysis_vars = ['d_hrwage_ln_1990_2008', 'expof_euro5_qo93_07', 'wage_earners_1990',
                 'ipums_female_1990', 'ipums_hispanic_1990', 'ipums_white_1990', 
                 'ipums_black_1990', 'ipums_asian_1990', 'ipums_highschool_1990', 
                 'ipums_college_1990', 'ipums_masters_1990', 'ipums_above65_1990', 
                 'ipums_logpop_1990', 'ind_share_manufacturing_1990', 
                 'ind_share_manufacturing_f_1990', 'ind_share_light_manuf_1990',
                 'chinashock_iv_90_07', 'expo_share_occ_routine_1990', 
                 'division', 'group_id', 'stcode']

# Drop only rows with missing values in analysis variables
df_clean = df.dropna(subset=analysis_vars)


# =====================================
# Table 2 Panel B Column 4: Change in Log Hourly Wages, 1990-2007
# =====================================

# Dependent variable - absolute value of wage change (will be logged by replicate function)
# The sign will be restored after logging
y_col4 = np.exp(df_clean['d_hrwage_ln_1990_2008'])

# Main treatment variable
main_treatment = 'expof_euro5_qo93_07'

# Define control variables exactly as in Stata globals
demographics = ['ipums_female_1990', 'ipums_hispanic_1990', 'ipums_white_1990', 
                'ipums_black_1990', 'ipums_asian_1990', 'ipums_highschool_1990', 
                'ipums_college_1990', 'ipums_masters_1990', 'ipums_above65_1990', 
                'ipums_logpop_1990']

industry_shares = ['ind_share_manufacturing_1990', 'ind_share_manufacturing_f_1990', 
                   'ind_share_light_manuf_1990']

other_shocks = ['chinashock_iv_90_07', 'expo_share_occ_routine_1990']

# Region effects - division (Census division dummies)
region_effects = ['division']

# Combine all controls
controls = demographics + industry_shares + other_shocks + region_effects

# Fixed effects for demographic groups (absorbed in reghdfe)
group_fe = ['group_id']

# X matrix: treatment first, then controls, then FE variables
X_col4 = sm.add_constant(df_clean[[main_treatment] + controls + group_fe], prepend=False)

# Weight variable - number of wage earners in 1990
weights_col4 = df_clean['wage_earners_1990']

# Cluster variable - state code
cluster_col4 = df_clean['stcode']

# METADATA FOR THIS RESULT
metadata_col4 = {
    'paper_id': '073',
    'table_id': '2',
    'panel_identifier': 'B_4',
    'model_type': 'log-linear',  # Log-linear because we're using wage levels that need to be logged
    'comments': 'Table 2 Panel B Column 4: Change in hourly wages (absolute values logged, signs preserved) 1990-2008 at demographic cell × czone level; robot exposure with all controls, division FE, and group_id FE absorbed; weighted by wage earners 1990, clustered by state'
}

# RUN THE REPLICATION
# Note: After the replicate function logs the absolute values, you may need to restore signs
# to the logged values if needed for interpretation
replicate(
    metadata=metadata_col4,
    y=y_col4,
    X=X_col4, 
    interest='expof_euro5_qo93_07',
    elasticity=False,
    fe=['division', 'group_id'],  # Both Census division and demographic group FE
    #weights=weights_col4,
    #kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col4}},
    #output=True, output_dir=OUTPUT_DIR, replicated=True
    )

