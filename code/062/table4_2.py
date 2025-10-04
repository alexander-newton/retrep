# 062/table4_col2_ifls_single.py
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
INPUT_DATA_DIR = config['intermediatedata']



# ----- LOAD DATA -----
fp = os.path.join(INPUT_DATA_DIR, '062/clean_IFLS.dta')
df = pd.read_stata(fp)




# Additional filters from Stata code that we missed:





needed = [
    'log_dowry',
    'elementary_plus', 'junior_plus', 'college',
    'mar_age', 'mar_age2',
    'mar_year', 'mar_year2',
    'year', 'ethnicity'
]
df = df.dropna(subset=needed)

df = df[df['ethnicity'] != 9].copy()

# --- y in LEVELS (replicate logs internally) ---
y = np.exp(df['log_dowry'].astype(float).values)

# --- Controls (NO FE dummies here) ---
X_core = df[['elementary_plus', 'junior_plus', 'college',
             'mar_age', 'mar_age2',
             'mar_year', 'mar_year2']].astype(float)


df['year_fe'] = df['year'].astype(int)  # already numeric, keep as int
df['ethnicity_fe'] = pd.Categorical(df['ethnicity'].astype(str)).codes  # stable codes

fe_vars = ['year_fe', 'ethnicity_fe']

# Build X with FE codes included (plus constant)
X = pd.concat([X_core, df[fe_vars]], axis=1)
X = sm.add_constant(X, prepend=False)

# --- Metadata ---
metadata = {
    'paper_id': '062',
    'table_id': '4',
    'panel_identifier': '2',
    'model_type': 'log-linear',
    'comments': 'Table 4, Col (2): OLS, log_dowry on schooling + marriage-age + marriage-year poly with year & ethnicity FE. y in levels; FE absorbed via pyhdfe.'
}

# --- Run ---
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='college',
    elasticity=False,
    fe=fe_vars,            # pass identifiers; replicate/pyhdfe will absorb them

)
