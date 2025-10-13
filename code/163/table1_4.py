import yaml
import os
import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# =====================================
# Table 1 Column 4 (Martínez 2022 JPE)
# Stata baseline:
# reghdfe lngdp14 lndn13 fiw fiw2 lndn13_fiw, absorb(countrycode year) cluster(countrycode)
# =====================================

# LOAD DATA
filepath = os.path.join(INPUT_DATA_DIR, '163/Estimations.dta')
df = pd.read_stata(filepath)

# Keep the main estimating sample (as in Stata: if fiw != .)
df = df[~df['fiw'].isna()].copy()

# Required variables
required_vars = [
    'lngdp14', 'lndn13', 'fiw', 'fiw2', 'lndn13_fiw',
    'countrycode', 'year'
]
df = df.dropna(subset=required_vars).copy()

# ---- Fixed effects as numeric codes ----
df['country_fe'] = pd.Categorical(df['countrycode']).codes
df['year_fe'] = pd.Categorical(df['year']).codes

# ---- Dependent variable in LEVELS (replicate logs internally) ----
# Stata DV is lngdp14; convert to levels for replicate
y = np.exp(df['lngdp14'])

# ---- X matrix: lndn13, fiw, fiw2, lndn13_fiw + FE ----
X = df[['lndn13', 'fiw', 'fiw2', 'lndn13_fiw', 'year_fe', 'country_fe']].copy()

# Add constant (replicate expects a design matrix; fine to include constant)
X = sm.add_constant(X, prepend=False)

# Indices of FE columns within X (year and country)
fe_indices = [
    X.columns.get_loc('year_fe'),
    X.columns.get_loc('country_fe')
]

# ---- METADATA ----
metadata = {
    'paper_id': '163',
    'table_id': '1',
    'panel_identifier': '4',  # column (4)
    'model_type': 'log-log',
    'comments': 'Replicates Table 1 Column 4: ln(GDP) on ln(NTL), FiW, FiW^2, and ln(NTL)*FiW with country and year FE. DV passed in levels because replicate() logs internally.'
}

# ---- RUN THE REPLICATION ----
# Variable of interest in col (4) is the interaction ln(NTL)*FiW
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='lndn13_fiw',   # key coefficient reported in col (4)
    fe=fe_indices,           # absorb year and country FE
    elasticity=True,         # log–log setting for interpretation (keeps consistency with your framework)
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
