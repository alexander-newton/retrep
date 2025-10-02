import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '131/cc_delays.dta')
df = pd.read_stata(filepath)

# =====================================
# Table 2 Panel B Column 4: areg with plu_* maj controls and family FE
# Dependent variable: loglength (log of length)
# Independent variable: pm_party_b
# Controls: plu_* variables and maj
# Fixed effects: countrynumber and family
# Sample: winner==1
# =====================================

# Apply sample restriction and drop missing values
df_sample = df[df['winner'] == 1].copy()

# Get all plu_* variables
plu_vars = [col for col in df_sample.columns if col.startswith('plu_')]
control_vars = plu_vars + ['maj']

# Drop missing values for all required variables
required_vars = ['loglength', 'pm_party_b', 'countrynumber', 'family'] + control_vars
df_sample = df_sample.dropna(subset=required_vars)

# Dependent variable
y = np.exp(df_sample['loglength'].values)

# Main independent variable
main_variable = 'pm_party_b'

# Create country and family fixed effects as dummy variables
country_dummies = pd.get_dummies(df_sample['countrynumber'], prefix='country', drop_first=True)
family_dummies = pd.get_dummies(df_sample['family'], prefix='family', drop_first=True)

# Combine main variable, controls, and fixed effects
X = pd.concat([
    df_sample[[main_variable] + control_vars],
    country_dummies,
    family_dummies
], axis=1)

# Add constant
X = sm.add_constant(X, prepend=False)

# Get the list of FE variables for metadata
fe_vars = country_dummies.columns.tolist() + family_dummies.columns.tolist()

# METADATA FOR THIS RESULT
metadata = {
    'paper_id': '131',
    'table_id': '2',
    'panel_identifier': 'B_4',
    'model_type': 'log-linear',
    'comments': 'Table 2 Panel B Column 4: areg loglength on pm_party_b with plu_* and maj controls, country and family FE, winner==1 sample'
}

# RUN THE REPLICATION
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=main_variable,
    elasticity=False,
    fe=fe_vars,
    #output=True, output_dir=OUTPUT_DIR, replicated=True, overwrite=True
)
