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

# Load the prepared dataset from Stata for Table 5
filepath = os.path.join(INPUT_DATA_DIR, '129/table5_prepared_data.dta')
df = pd.read_stata(filepath)

#==============================================================================
# REGRESSION - Table 5 Panel B Column 4 (hc_cantoni)
# Based on Stata script:
# areg ln_db_total1750 ln_db_total1470 ln_db_total1420
# ln_distance_wittenberg protestant_at_t t_protestant_at_t 
# ecclesiastical_at_t_1500 river hansa reichsstadt foundation 
# monasteries_nonmend_10km monasteries_a~m
# university pressby1517 
# pop1500* 
# if cityname_cantoni != "", absorb(polity1500) cluster(polity1500) ;
#==============================================================================

# Filter data: non-empty cityname_cantoni (note: no ln_u1800 filter in cantoni spec)
reg_data = df[
    (df['cityname_cantoni'].notna()) & 
    (df['cityname_cantoni'] != '')
].copy()

print(f"Sample size after filters: {len(reg_data)}")

# Convert categorical variables to numeric codes for replicate function
reg_data['polity1500_code'] = pd.Categorical(reg_data['polity1500']).codes

# Outcome variable: ln_db_total1750 (log of 1750 upper tail human capital)
# Convert back to levels since replicate logs internally
y = np.exp(reg_data['ln_db_total1750'])

# Identify dynamically all pop1500* variables and monasteries_a* variables
pop_cols = [col for col in reg_data.columns if col.startswith('pop1500')]
monasteries_aug_cols = [col for col in reg_data.columns if col.startswith('monasteries_a')]

print(f"Found {len(pop_cols)} population variables")
print(f"Found {len(monasteries_aug_cols)} monasteries_augustinian variables")

# Define X matrix for Panel B Column 4: Cantoni specification (NOTE: no ind_law in cantoni spec)
X_vars = ['protestant_at_t','ln_db_total1470', 'ln_db_total1420',
          'ln_distance_wittenberg_p1',  't_protestant_at_t',
          'ecclesiastical_at_t_1500', 'river', 'hansa', 'reichsstadt', 'foundation',
          'monasteries_nonmend_10km', 'university', 'pressby1517'] + monasteries_aug_cols + pop_cols + ['polity1500_code']

# Clean data and create final matrices
reg_data = reg_data.dropna(subset=X_vars + ['ln_db_total1750'])
y = np.exp(reg_data['ln_db_total1750'])

print(f"Final sample size: {len(reg_data)}")
print(f"Final unique polities: {reg_data['polity1500'].nunique()}")

X = reg_data[X_vars].copy()
X = sm.add_constant(X, prepend=False)

# Fixed effects variable names - absorb polity1500
fe_vars = ['polity1500_code']

# Cluster variable - cluster by polity1500
cluster = pd.Categorical(reg_data['polity1500']).codes

# Metadata
metadata = {
    'paper_id': '129',
    'table_id': '5',
    'panel_identifier': 'B_4',
    'model_type': 'log-linear',
    'comments': 'Table 5 Panel B Column 4: Cantoni specification - OLS regression of log 1750 upper tail on Cantoni variables, polity1500 FE, clustered SE'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='protestant_at_t',  # Main variable of interest in Cantoni specification
    elasticity=False,
    fe=fe_vars,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True,
    output_dir=OUTPUT_DIR,
    replicated=True
)
