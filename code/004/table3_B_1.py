import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

import sys
import os
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '004/AG_Corp_Prod_Database.dta')
df = pd.read_stata(filepath).dropna(subset=['RevperWorker', 'Form', 'RegIndYearGroup'])

# =====================================
# Table 3 Panel B Column 1: Revenue per Worker
# =====================================

# Define dependent variable
y_col1 = df['RevperWorker']

# Main treatment variable (Form)
main_treatment = 'Form'

# Define control and fixed effect variables
control_set = ['ProvinceFactor', 'IndustryFactor', 'YearFactor']

# Prepare X matrix - treatment variable first for easy identification
X_col1 = sm.add_constant(df[[main_treatment] + control_set], prepend=False)

# Define cluster variable
cluster_col1 = df['RegIndYearGroup']

# METADATA FOR THIS RESULT
metadata_col1 = {
    'paper_id': '004',
    'table_id': '3',
    'panel_identifier': 'B_1',
    'model_type': 'log-linear',
    'comments': 'Table 3 Panel B Column 1: Revenue per Worker on Form with Province, Industry, Year FE'
}

# RUN THE REPLICATION
# Just checking results without saving
replicate(
    metadata=metadata_col1, 
    y=y_col1, 
    X=X_col1, 
    interest='Form', 
    elasticity=False, 
    fe=['ProvinceFactor', 'IndustryFactor', 'YearFactor'],
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col1}},

    output=True, output_dir=OUTPUT_DIR, replicated=True
)


