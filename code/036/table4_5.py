import numpy as np
from loglinearcorrection.correction_estimator import DoublyRobustElasticityEstimator
import os
from datetime import datetime
import json
import pandas as pd
import yaml
import statsmodels.api as sm
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# Load data
filepath = os.path.join(INPUT_DATA_DIR, '036/2a.canton_dataset_HHexpenditure.dta')
df = pd.read_stata(filepath)

# Drop missing values
df = df.dropna(subset=['HH_expend', 'share_refractory', 'lnpopulation', 
                       'wheat_suit', 'lndist_Paris', 'lnsubs', 
                       'department_id', 'district_id'])

# Dependent variable (in levels)
y_col5 = df['HH_expend'].values

# Main treatment variable
main_treatment = 'share_refractory'

# Control variables
control_vars = ['lnpopulation', 'wheat_suit', 'lndist_Paris', 'lnsubs']

# Fixed effect variable
fe_var = 'department_id'

# Create X matrix
X_vars = [main_treatment] + control_vars + [fe_var]
X_col5 = sm.add_constant(df[X_vars], prepend=False)
X_col5['department_id'] = pd.Categorical(df['department_id'])

# Cluster variable
cluster_col5 = df['district_id'].values

# Metadata
metadata_col5 = {
    'paper_id': '036',
    'table_id': '4',
    'panel_identifier': '5',
    'model_type': 'log-linear',
    'comments': 'Table 4 Column 5: Log HH expenditure on share_refractory with all controls and department FE'
}

# Run replication
replicate(
    metadata=metadata_col5,
    y=y_col5,
    X=X_col5,
    interest=main_treatment,
    elasticity=False,
    fe='department_id',
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col5}},
    output=True, output_dir=OUTPUT_DIR, replicated=True

)