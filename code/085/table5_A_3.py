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
filepath = os.path.join(INPUT_DATA_DIR, '085/boat_anon.dta')  # Adjust path as needed
df = pd.read_stata(filepath)

# =====================================
# Table 5 Panel A Column 3: Time at Sea on VI
# Fixed Effects: boat_plant, date
# Clustered SE: boat_num
# No filter condition for this column
# =====================================

# Filter to remove missing values for the variables we need
df_clean = df[['time_at_sea', 'VI', 'boat_plant', 'date', 'boat_num']].dropna()

# Dependent variable (time_at_sea is already log-transformed based on Table 5 title)
y_col3 = np.exp(df_clean['time_at_sea'])

# Main treatment variable
main_treatment = 'VI'

# Fixed effects variables
fe_variables = ['boat_plant', 'date']

# Create X matrix: VI first (as treatment/interest variable), then FEs
X_col3 = sm.add_constant(df_clean[[main_treatment] + fe_variables], prepend=False)

# Cluster variable
cluster_col3 = df_clean['boat_num']

# METADATA FOR THIS RESULT
metadata_col3 = {
    'paper_id': 'boat_study',  # Update with actual paper ID
    'table_id': '5',
    'panel_identifier': 'A_3',
    'model_type': 'log-linear',  # log-linear since time_at_sea is log of time
    'comments': 'Table 5 Panel A Column 3: Log(Total Time Spent at Sea) on VI with boat_plant and date FE; clustered SE at boat_num level'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_col3,
    y=y_col3,
    X=X_col3,
    interest='VI',
    elasticity=False,
    fe=fe_variables,
    #kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col3}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)