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

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '004/AG_Corp_Prod_Database.dta')
df = pd.read_stata(filepath).dropna(subset=['PowerperWorker', 'Form', 'RegIndYearGroup'])

# =====================================
# Table 3 Panel B Column 2: Power per Worker (log(HP/L))
# Sample: exclude 1900; include 1908
# =====================================

df = df[(df['PowerperWorker'] > 0) & (df['YearFactor'] != 1900)]

# Dependent variable
y_col2 = df['PowerperWorker']  # log(HP/L)

# Main treatment variable (Corporation indicator)
main_treatment = 'Form'


# Controls / FE — no age controls for Panel B cols (1)–(3)
control_set = ['ProvinceFactor', 'IndustryFactor', 'YearFactor']

# X matrix: treatment first (for easy identification)
X_col2 = sm.add_constant(df[[main_treatment] + control_set], prepend=False)

# Cluster variable
cluster_col2 = df['RegIndYearGroup']

# METADATA FOR THIS RESULT
metadata_col2 = {
    'paper_id': '004',
    'table_id': '3',
    'panel_identifier': 'B_2',
    'model_type': 'log-linear',
    'comments': 'Table 3 Panel B Column 2: Power per Worker (log(HP/L)) on Form with Province, Industry, Year FE; 1900 excluded, 1908 included; no age controls'
}

# RUN THE REPLICATION
# Just checking results without saving
replicate(
    metadata=metadata_col2,
    y=y_col2,
    X=X_col2,
    interest='Form',
    elasticity=False,
    fe=['ProvinceFactor', 'IndustryFactor', 'YearFactor'],
    # kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col2}},

    output=True, output_dir=OUTPUT_DIR, replicated=True
)
