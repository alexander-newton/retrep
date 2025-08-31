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
# Table 5 Panel A Column 2: Distance from Port on VI
# Fixed Effects: boat_plant, date
# Clustered SE: boat_num
# Filter: moving == 0 (only when boat is stationary)
# =====================================

# Filter to only include observations where moving == 0 AND remove missing values
df_clean = df[(df['moving'] == 0)][['distance_port2', 'VI', 'boat_plant', 'date', 'boat_num']].dropna()

# Dependent variable (distance_port2 is already log-transformed based on Table 5 title)
y_col2 = np.exp(df_clean['distance_port2'])

# Main treatment variable
main_treatment = 'VI'

# Fixed effects variables
fe_variables = ['boat_plant', 'date']

# Create X matrix: VI first (as treatment/interest variable), then FEs
X_col2 = sm.add_constant(df_clean[[main_treatment] + fe_variables], prepend=False)

# Cluster variable
cluster_col2 = df_clean['boat_num']

# METADATA FOR THIS RESULT
metadata_col2 = {
    'paper_id': 'boat_study',  # Update with actual paper ID
    'table_id': '5',
    'panel_identifier': 'A_2',
    'model_type': 'log-linear',  # log-linear since distance_port2 is log of distance
    'comments': 'Table 5 Panel A Column 2: Log(Maximum Distance from Plant Port) on VI with boat_plant and date FE; filtered for moving==0; clustered SE at boat_num level'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_col2,
    y=y_col2,
    X=X_col2,
    interest='VI',
    elasticity=False,
    fe=fe_variables,
    #kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col2}},
    output=True, output_dir=OUTPUT_DIR, replicated=True
)