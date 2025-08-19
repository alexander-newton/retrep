import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
import sys
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '002/measures.dta')
df = pd.read_stata(filepath)

# =====================================
# Table 6 Column 1: Cognition (OLS)
# =====================================

# Create log transformations for all variables needed
df['log_material'] = np.log(df['fci_play_mat_type1'] + 1)
df['log_time'] = np.log(df['fci_play_act1'] + 1)
df['log_cog_baseline'] = np.log(df['b_tot_cog0'] + 1)
df['log_socio_baseline'] = np.log(df['bates_difficult0'] + 1)
df['log_mother_cog'] = np.log(df['raventot'] + 1)
df['log_mother_socio'] = np.log(df['cesdA0'] + 1)
df['log_nchildren'] = np.log(df['nkids1'] + 1)

# Clean data
df_clean = df.dropna(subset=['b_tot_cog1', 'treat', 'log_material', 'log_time', 
                              'log_cog_baseline', 'log_socio_baseline', 
                              'log_mother_cog', 'log_mother_socio', 'log_nchildren'])

# Define dependent variable
y_col1 = df_clean['b_tot_cog1']

# Main treatment variables
main_treatments = ['treat', 'log_material', 'log_time']

# Define control variables
control_set = ['log_cog_baseline', 'log_socio_baseline', 'log_mother_cog', 
               'log_mother_socio', 'log_nchildren']

# Prepare X matrix - treatment variables first for easy identification
X_col1 = sm.add_constant(df_clean[main_treatments + control_set], prepend=False)

# Define cluster variable
#cluster_col1 = df_clean['cod_dane']

# METADATA FOR THIS RESULT
metadata_col1 = {
    'paper_id': '002',
    'table_id': '6',
    'panel_identifier': '1',
    'model_type': 'log-log',
    'comments': 'Table 6 Column 1: OLS estimation of production function for cognitive skills'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_col1, 
    y=y_col1, 
    X=X_col1, 
    interest= 'treat', 
    elasticity=False, 
    fe=None,
    #kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col1}},
)