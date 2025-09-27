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

# Load the prepared dataset from Stata for Table 6
# Note: This data was prepared using raw data files and Stata scripts located in this directory
# The data preparation Stata code (table6_datapreparation.do) has been included in this folder
# for transparency and reproducibility of the data construction process
filepath = os.path.join(INPUT_DATA_DIR, '129/table6_prepared_data.dta')
df = pd.read_stata(filepath)

#==============================================================================
# REGRESSION - Table 6 Panel B Column 1 (IV: Log Population 1800)
# Based on Stata script:
# ivreg2 ln_u1800 ln_db_total1470 ln_db_total1420
#	mean_plague1349_1499 
#	ind_inco_pre1517 ind_mkts_pre1517 ind_freecity
#	books_0-books_1001 ind_univ_by1517 students10*
#	pop1500* 
#	(test = plagues_years1500_1522)
#	if ln_u1800 != .&cityname_cantoni!=""
#	, cluster(polity1500)
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

# Outcome variable: ln_u1800 (log of 1800 population)
# Convert back to levels since replicate logs internally
y = np.exp(reg_data['ln_u1800'])

# Identify dynamically all books_*, students10*, and pop1500* variables
books_cols = [col for col in reg_data.columns if col.startswith('books_')]
students_cols = [col for col in reg_data.columns if col.startswith('students10')]
pop_cols = [col for col in reg_data.columns if col.startswith('pop1500')]

print(f"Found {len(books_cols)} books variables")
print(f"Found {len(students_cols)} students variables") 
print(f"Found {len(pop_cols)} population variables")

# Define exogenous variables (controls only - instrument is separate)
exog_vars = ['ln_db_total1470', 'ln_db_total1420',
             'mean_plague1349_1499',
             'ind_inco_pre1517', 'ind_mkts_pre1517', 'ind_freecity',
             'ind_univ_by1517'] + books_cols + students_cols + pop_cols

# Define endogenous variable: test (which is ind_law in the dataset)
endog_var = ['test']

# Instrument: plagues_years1500_1522
instrument = ['plagues_years1500_1522']

# Create full X matrix (exogenous + endogenous variables)
X_vars = endog_var + exog_vars

# Clean data and create final matrices
reg_data = reg_data.dropna(subset=X_vars + ['ln_u1800'])
y = np.exp(reg_data['ln_u1800'])

print(f"Final sample size: {len(reg_data)}")
print(f"Final unique polities: {reg_data['polity1500'].nunique()}")

# Create X matrix with all variables
X = reg_data[X_vars].copy()
X = sm.add_constant(X, prepend=False)

# Create instrument matrix (exogenous variables + instrument)
z_vars = exog_vars + ['plagues_years1500_1522']  # All exogenous variables + instrument
Z = reg_data[z_vars].copy()
Z = sm.add_constant(Z, prepend=False)

# Cluster variable - cluster by polity1500
cluster = pd.Categorical(reg_data['polity1500']).codes

# Metadata
metadata = {
    'paper_id': '129',
    'table_id': '6',
    'panel_identifier': 'B_1_population',
    'model_type': 'log-linear',
    'comments': 'Table 6 Panel B Column 1: IV regression of log 1800 population on law (instrumented by plagues 1500-1522), no territory FE, clustered SE'
}

# For IV regression, we need to specify:
# - endog_x: index/name of endogenous variable in X matrix
# - z: instrument matrix
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='test',  # This is our endogenous variable of interest
    endog_x='test',   # Specify which variable is endogenous
    z=Z,              # Instrument matrix
    elasticity=False,
    fe=None,          # No fixed effects in Column 1
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster}},
    output=True,
    output_dir=OUTPUT_DIR,
    replicated=True
)
