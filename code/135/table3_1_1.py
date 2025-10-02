import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
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
missing_fico = df.groupby('countyfips')['avg_fico'].apply(lambda x: x.isna().any())
missing_ltv = df.groupby('countyfips')['avg_ltv'].apply(lambda x: x.isna().any())
valid_counties = df['countyfips'][~df['countyfips'].isin(missing_fico[missing_fico].index) & 
                                  ~df['countyfips'].isin(missing_ltv[missing_ltv].index)].unique()
df = df[df['countyfips'].isin(valid_counties)]

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
# TABLE 3 PANEL I COLUMN 1: QE1 WITHOUT CONTROLS
# =====================================

# Apply sample restriction
df_qe1 = df[df['qe1_sample3_dummy'] == 1].copy()

# Drop missing values and ensure numeric types
df_qe1 = df_qe1.dropna(subset=['lq', 'qe1_dummy', 'qe1_jumbo', 'jumbo'])

# Prepare data - convert log to levels for replication function
y = np.exp(df_qe1['lq'].astype(float).values)
X = df_qe1[['qe1_jumbo','qe1_dummy',  'jumbo']].astype(float)
X = sm.add_constant(X)

# METADATA FOR THIS RESULT
metadata = {
    'paper_id': '135',
    'table_id': '3',
    'panel_identifier': '1_1',
    'model_type': 'log-linear',
    'comments': 'Table 3 Panel I Column 1: QE1 without controls - reg lq on qe1_dummy qe1_jumbo jumbo'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='qe1_jumbo',
    elasticity=False,
    output=True, output_dir=OUTPUT_DIR, replicated=True,
)
