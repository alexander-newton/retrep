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
filepath = os.path.join(INPUT_DATA_DIR, '055/gravity_analysis_final.dta')
df = pd.read_stata(filepath)

# =====================================
# Table 1 Panel A Column 1: OLS
# Dependent variable: uprob (probability of commuting)
# Independent variable: lttrans1921 (log transit travel time)
# Fixed effects: Origin and Destination borough indicators
# Sample: sample==1 (positive commuting flows, excluding self-commuting)
# =====================================

# Apply sample restriction and drop missing values
df_sample = df[df['sample'] == 1].copy()
df_sample = df_sample.dropna(subset=['uprob', 'lttrans1921', 'G_UNIT_o', 'G_UNIT_d'])

# Dependent variable
y_col1 = df_sample['uprob'].values

# Main independent variable
main_variable = 'lttrans1921'

# Create fixed effects as dummy variables
origin_dummies = pd.get_dummies(df_sample['G_UNIT_o'], prefix='origin', drop_first=True)
dest_dummies = pd.get_dummies(df_sample['G_UNIT_d'], prefix='dest', drop_first=True)

# Combine main variable with fixed effects
X_col1 = pd.concat([
    df_sample[[main_variable]],
    origin_dummies,
    dest_dummies
], axis=1)

# Add constant
X_col1 = sm.add_constant(X_col1, prepend=False)

# Get the list of FE variables for metadata
fe_vars = origin_dummies.columns.tolist() + dest_dummies.columns.tolist()

# METADATA FOR THIS RESULT
metadata_col1 = {
    'paper_id': '055',
    'table_id': '1',
    'panel_identifier': 'A_1',
    'model_type': 'log-log',
    'comments': 'Table 1 Panel A Column 1: OLS of log probability of commuting (uprob) on log transit travel time (lttrans1921) with origin and destination FE'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata_col1,
    y=y_col1,
    X=X_col1,
    interest=main_variable,
    elasticity=True,
    fe=fe_vars,
    output=True, output_dir=OUTPUT_DIR, replicated=True, overwrite=True

)