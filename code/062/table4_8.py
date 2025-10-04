# 062/table4_col8_zfps_single.py
import yaml, os, sys, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# ----- CONFIG -----
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config['outputdata']    
INPUT_DATA_DIR = config['intermediatedata']

# ----- LOAD DATA -----
fp = os.path.join(INPUT_DATA_DIR, '062/ZFPS_clean.dta')
df = pd.read_stata(fp)

# --- Filters from Stata code ---
# keep if H_lobolacharged==1
df = df[df['H_lobolacharged'] == 1].copy()

# --- Rename variables to keep names consistent for table output ---
# rename primary elementary_plus
if 'primary' in df.columns:
    df['elementary_plus'] = df['primary']

# rename junior_sec junior_plus  
if 'junior_sec' in df.columns:
    df['junior_plus'] = df['junior_sec']

# rename H_primary husband_elementary
if 'H_primary' in df.columns:
    df['husband_elementary'] = df['H_primary']

# rename H_junior_sec husband_junior
if 'H_junior_sec' in df.columns:
    df['husband_junior'] = df['H_junior_sec']

# rename H_secondary husband_college
if 'H_secondary' in df.columns:
    df['husband_college'] = df['H_secondary']

# rename secondary college
if 'secondary' in df.columns:
    df['college'] = df['secondary']

# --- Define needed variables for regression ---
needed = [
    'H_ln_lobolacharged_amnt',  # dependent variable
    'elementary_plus', 'junior_plus', 'college',
    'marriage_age', 'marriage_age2',  # marriage age variables
    'mar_year', 'mar_year2',  # marriage year variables
    'ethnicity'  # ethnicity for fixed effects (using 'ethnicity' as in Stata code)
]

# Drop observations with missing values in needed variables
df = df.dropna(subset=needed)

# --- y variable (dependent variable) ---
y = np.exp(df['H_ln_lobolacharged_amnt'].astype(float).values)

# --- Controls (NO FE dummies here) ---
X_core = df[['elementary_plus', 'junior_plus', 'college',
             'marriage_age', 'marriage_age2',
             'mar_year', 'mar_year2']].astype(float)

# Create fixed effects variables
df['ethnicity_fe'] = pd.Categorical(df['ethnicity'].astype(str)).codes  # stable codes

fe_vars = ['ethnicity_fe']

# Build X with FE codes included (plus constant)
X = pd.concat([X_core, df[fe_vars]], axis=1)
X = sm.add_constant(X, prepend=False)

# --- Metadata ---
metadata = {
    'paper_id': '062',
    'table_id': '4',
    'panel_identifier': '8',
    'model_type': 'log-linear',
    'comments': 'Table 4, Col (8): ZFPS Hedonic regression, H_ln_lobolacharged_amnt on schooling + marriage-age + marriage-year poly with ethnicity FE. '
}

# --- Run ---
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='college',
    elasticity=False,
    fe=fe_vars,
    output=True, output_dir=OUTPUT_DIR, replicated=True
)
