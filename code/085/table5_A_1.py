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
filepath = os.path.join(INPUT_DATA_DIR, '085/boat_anon.dta')  # Adjust path as needed
df = pd.read_stata(filepath)

# =====================================
# Table 5 Panel A Column 1: Offload on VI
# Fixed Effects: boat_plant, date
# Clustered SE: boat_num
# =====================================

# Filter to remove missing values for the variables we need
df_clean = df[['offload', 'VI', 'boat_plant', 'date', 'boat_num']].dropna()

# Dependent variable
y_col1 = np.exp(df_clean['offload'])

# Main treatment variable
main_treatment = 'VI'

# Fixed effects variables
fe_variables = ['boat_plant', 'date']

# Create X matrix: VI first (as treatment/interest variable), then FEs
X_col1 = sm.add_constant(df_clean[[main_treatment] + fe_variables], prepend=False)

# Cluster variable
cluster_col1 = df_clean['boat_num']

# METADATA FOR THIS RESULT
metadata_col1 = {
    'paper_id': '085',  # Update with actual paper ID
    'table_id': '5',
    'panel_identifier': 'A_1',
    'model_type': 'log-linear',  # Change to 'log-linear' if offload is in logs
    'comments': 'Table 5 Panel A Column 1: Effect of VI on offload with boat_plant and date FE; clustered SE at boat_num level'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_col1,
    y=y_col1,
    X=X_col1,
    interest='VI',
    elasticity=False,
    fe=fe_variables, 
    #kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col1}},
    output=True, output_dir=OUTPUT_DIR, replicated=True, overwrite=True

)

