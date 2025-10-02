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
# REGRESSION - Table 5 Panel B Column 2 (hc_control)
# Based on Stata script:
# areg ln_db_total1750 ind_law ln_db_total1470 ln_db_total1420
#	ind_inco_pre1517 ind_mkts_pre1517 books_0-books_1001
#	ind_freecity
#	ind_univ_by1517 students10*
#	pop1500* 
#	mean_plague1400_1499
#	if ln_u1800 != . & cityname_cantoni != ""
# , absorb(polity1500) cluster(polity1500);
#==============================================================================

# Filter data: non-missing ln_u1800 and non-empty cityname_cantoni
reg_data = df[
    (df['ln_u1800'].notna()) & 
    (df['cityname_cantoni'].notna()) & 
    (df['cityname_cantoni'] != '')
].copy()

print(f"Sample size after filters: {len(reg_data)}")

# Convert categorical variables to numeric codes for replicate function
reg_data['polity1500_code'] = pd.Categorical(reg_data['polity1500']).codes

# Outcome variable: ln_db_total1750 (log of 1750 upper tail human capital)
# Convert back to levels since replicate logs internally
y = np.exp(reg_data['ln_db_total1750'])

# Identify dynamically all books_* and students10* and pop1500* variables
books_cols = [col for col in reg_data.columns if col.startswith('books_')]
students_cols = [col for col in reg_data.columns if col.startswith('students10')]
pop_cols = [col for col in reg_data.columns if col.startswith('pop1500')]

print(f"Found {len(books_cols)} books variables")
print(f"Found {len(students_cols)} students variables") 
print(f"Found {len(pop_cols)} population variables")

# Define X matrix for Panel B Column 2: control specification
# Main variables + all controls (books, students, pop) + fixed effects
# Note: Panel B uses mean_plague1400_1499 (different from Panel A which uses mean_plague1349_1499)
X_vars = ['ind_law', 'ln_db_total1470', 'ln_db_total1420',
          'ind_inco_pre1517', 'ind_mkts_pre1517', 
          'ind_freecity', 'ind_univ_by1517',
          'mean_plague1400_1499'] + books_cols + students_cols + pop_cols + ['polity1500_code']

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
    'panel_identifier': 'B_2',
    'model_type': 'log-linear',
    'comments': 'Table 5 Panel B Column 2: OLS regression of log 1750 upper tail on reformation laws with controls, polity1500 FE, clustered SE'
}

replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='ind_law',
    elasticity=False,
    fe=fe_vars,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True,
    output_dir=OUTPUT_DIR,
    replicated=True
)
