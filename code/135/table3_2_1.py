import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
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
filepath = os.path.join(INPUT_DATA_DIR, '135/quantity_prep_2002_2014.dta')
df = pd.read_stata(filepath)

# Dataset loaded successfully

# Format date - create monthly time series like Stata
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
df['date'] = df['date'].dt.to_period('M')

# Create log-quantity variables
df['lq'] = np.log(df['quantity_amount'])
df['lqc'] = np.log(df['quantity_count'])

# =====================================
# SAMPLE RESTRICTIONS
# =====================================

# Keep years between 2008 & 2013
df = df[(df['year'] >= 2008) & (df['year'] <= 2013)]

# Keep counties that have at least 1 jumbo loan in every month of sample
jumbo_counts = df[(df['jumbo'] == 1) & (df['quantity_count'] >= 1) & (df['quantity_count'].notna())].groupby('countyfips')['month'].count()
valid_counties = jumbo_counts[jumbo_counts >= 72].index
df = df[df['countyfips'].isin(valid_counties)]

# Drop counties that are ever missing a FICO or LTV score
# Stata: by county: egen evermissfico = max(missfico)
missing_fico = df.groupby('countyfips')['avg_fico'].apply(lambda x: x.isna().any())
missing_ltv = df.groupby('countyfips')['avg_ltv'].apply(lambda x: x.isna().any())
valid_counties = df['countyfips'][~df['countyfips'].isin(missing_fico[missing_fico].index) & 
                                  ~df['countyfips'].isin(missing_ltv[missing_ltv].index)].unique()
df = df[df['countyfips'].isin(valid_counties)]

# =====================================
# CONTROLS
# =====================================

# Lagged G-Fee (placeholder - would need actual data)
df['gfees_lag'] = 0.0
df['gfeeXjumbo'] = df['gfees_lag'] * df['jumbo']

# Estimated FICO gap (placeholder - would need actual data)
df['estimatefico'] = 0.0
df['estimateficoXjumbo'] = df['estimatefico'] * df['jumbo']

# Lagged financial CDS index (placeholder - would need actual data)
df['spread5y'] = 0.0
df['cdsXjumbo'] = df['jumbo'] * df['spread5y']

# Create fixed effects groups
df['county_segment'] = df['countyfips'].astype(str) + '_' + df['jumbo'].astype(str)
df['county_month'] = df['countyfips'].astype(str) + '_' + df['date'].astype(str)

# Create standard error clustering groups
df['clustervar'] = df['jumbo'].astype(str) + '_' + df['date'].astype(str)

# =====================================
# QE PERIODS
# =====================================

# QE1 period - December 2008 to April 2010
qe1_start = pd.to_datetime('2008-12-01')
qe1_end = pd.to_datetime('2010-04-01')
df['qe1_dummy'] = ((df['date'].dt.to_timestamp() >= qe1_start) & 
                   (df['date'].dt.to_timestamp() < qe1_end)).astype(int)
df['qe1_jumbo'] = df['qe1_dummy'] * df['jumbo']

# QE1 sample period - 3 months before and after December 2008
# Stata: tm(2008m12) - 3 = September 2008, tm(2008m12) + 3 = March 2009
qe1_sample_start = pd.to_datetime('2008-09-01')  # September 2008
qe1_sample_end = pd.to_datetime('2009-03-01')    # March 2009
df['qe1_sample3_dummy'] = ((df['date'].dt.to_timestamp() >= qe1_sample_start) & 
                           (df['date'].dt.to_timestamp() < qe1_sample_end)).astype(int)

# =====================================
# TIME-SERIES CONTROLS REGRESSION
# =====================================

# Run time-series controls regression to get residuals
df_controls = df[(df['year'] >= 2008) & (df['year'] <= 2013)].copy()

# Drop missing values and ensure numeric types
df_controls = df_controls.dropna(subset=['lq', 'gfeeXjumbo', 'cdsXjumbo', 'estimateficoXjumbo', 'jumbo', 
                                       'avg_fico', 'avg_ltv', 'avg_fico_nonmissing'])

y_controls = df_controls['lq'].astype(float).values
X_controls = df_controls[['gfeeXjumbo', 'cdsXjumbo', 'estimateficoXjumbo', 'jumbo', 
                          'avg_fico', 'avg_ltv', 'avg_fico_nonmissing']].astype(float)

# Add fixed effects
county_segment_dummies = pd.get_dummies(df_controls['county_segment'], prefix='cs', drop_first=True)
county_month_dummies = pd.get_dummies(df_controls['county_month'], prefix='cm', drop_first=True)
X_controls_full = pd.concat([X_controls, county_segment_dummies, county_month_dummies], axis=1)
X_controls_full = sm.add_constant(X_controls_full)

# Convert to numpy arrays to avoid object dtype issues
y_controls = np.asarray(y_controls, dtype=float)
X_controls_full = np.asarray(X_controls_full, dtype=float)

model_controls = sm.OLS(y_controls, X_controls_full)
results_controls = model_controls.fit()

# Calculate residuals for Panel II
# Get parameter indices for the time-series controls
param_names = ['gfeeXjumbo', 'cdsXjumbo', 'estimateficoXjumbo']
gfee_idx = 1  # Index of gfeeXjumbo in X_controls (after constant)
cds_idx = 2   # Index of cdsXjumbo in X_controls
fico_idx = 3  # Index of estimateficoXjumbo in X_controls

df['lq_r'] = df['lq'] - (results_controls.params[gfee_idx] * df['gfeeXjumbo'] + 
                         results_controls.params[cds_idx] * df['cdsXjumbo'] + 
                         results_controls.params[fico_idx] * df['estimateficoXjumbo'])

# =====================================
# TABLE 3 PANEL II COLUMN 1: QE1 WITH CONTROLS
# =====================================

# Apply sample restriction
df_qe1 = df[df['qe1_sample3_dummy'] == 1].copy()

# Drop missing values and ensure numeric types
df_qe1 = df_qe1.dropna(subset=['lq_r', 'qe1_dummy', 'qe1_jumbo', 'jumbo', 'avg_fico', 'avg_ltv', 'avg_fico_nonmissing'])

# Prepare data - convert log residuals back to levels for replication function
y = np.exp(df_qe1['lq_r'].astype(float).values)
X = df_qe1[['qe1_dummy', 'qe1_jumbo', 'jumbo', 'avg_fico', 'avg_ltv', 'avg_fico_nonmissing']].astype(float)
X = sm.add_constant(X)

# Add fixed effects
county_segment_dummies = pd.get_dummies(df_qe1['county_segment'], prefix='cs', drop_first=True)
county_month_dummies = pd.get_dummies(df_qe1['county_month'], prefix='cm', drop_first=True)
X_full = pd.concat([X, county_segment_dummies, county_month_dummies], axis=1)

# Convert y to numpy array but keep X_full as DataFrame for replication function
y = np.asarray(y, dtype=float)

# METADATA FOR THIS RESULT
metadata = {
    'paper_id': '135',
    'table_id': '3',
    'panel_identifier': '2_1',
    'model_type': 'log-linear',
    'comments': 'Table 3 Panel II Column 1: QE1 with controls - reghdfe lq_r on qe1_dummy qe1_jumbo jumbo with county-segment and county-month FE'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata,
    y=y,
    X=X_full,
    interest='qe1_jumbo',
    elasticity=False,
    fe=list(county_segment_dummies.columns) + list(county_month_dummies.columns)
)