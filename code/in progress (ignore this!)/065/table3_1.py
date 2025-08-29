import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
import sys

# Add parent directory to path for replication module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# =====================================
# LOAD DATA
# =====================================
# Based on the Stata code, we need the dataset_main_st_AUG2018.dta file
filepath = os.path.join(INPUT_DATA_DIR, '065/dataset_main_st_AUG2018.dta')
df = pd.read_stata(filepath)

# =====================================
# DATA PREPARATION (following Stata code)
# =====================================

# Create entry4 variable
df['entry4'] = 0
df.loc[(df['WN_Qentered'] >= df['QasaPE']) & 
       (df['WN_Qentered'] < df['QasaPE'] + 5), 'entry4'] = 1

# Transform variables (only if not already done in the data)
df['distance_km'] = df['distance'] / 1000  # Convert to thousands of km
df['avgpop_mil'] = df['avgpop'] / 1000000  # Convert to millions
df['distsq'] = (df['distance']/1000) ** 2
df['avgpopsq'] = (df['avgpop']/1000000) ** 2
df['popsq'] = df['avgpopsq']  # Alias

# Fuel variables (already exist as 'fuel' and 'lagged_fuel')
df['fueldist'] = df['fuel'] * (df['distance']/1000)
df['fueldistsq'] = df['fuel'] * df['distsq']

# Calculate mean fuel distance by quarter and deviation
df['meanfueldist'] = df.groupby('quarter')['fueldist'].transform('mean')
df['deviationsq'] = (df['fueldist'] - df['meanfueldist']) ** 2

# Market size adjustments
df['mktsize_adj'] = df['mktsize'] / 1000

# Create PEmktsize variables
df['PEmktsize'] = 0
df.loc[df['quarter'] == df['QasaPE'], 'PEmktsize'] = df.loc[df['quarter'] == df['QasaPE'], 'mktsize_adj']
df['PEmktsize'] = df.groupby(['origin', 'destination'])['PEmktsize'].transform('max')
df['PEmktsizesq'] = df['PEmktsize'] ** 2

# Create duplicate indicator for unique obs per market-quarter
df['dupid'] = df.groupby(['origin', 'destination', 'quarter']).cumcount() == 0

# Create PEroutepass - average market size in phase 1
def calc_phase1_var(group, var_name):
    """Calculate phase 1 average for a given variable"""
    if len(group) == 0:
        return np.nan
    
    QasaPE = group['QasaPE'].iloc[0]
    Qstart = group['quarter'].min()
    
    # Phase 1 condition from Stata code
    mask = ((group['quarter'] <= QasaPE - 4) | 
            ((group['quarter'] < QasaPE) & (QasaPE <= 4)) |
            ((group['quarter'] < QasaPE) & (QasaPE - Qstart <= 4) & (Qstart > 4))) & group['dupid']
    
    if mask.any():
        return group.loc[mask, var_name].mean()
    return np.nan

# Calculate PEroutepass
grouped = df.groupby(['origin', 'destination'])
df['PEroutepass'] = grouped.apply(lambda x: calc_phase1_var(x, 'mktsize_adj')).reindex(df.groupby(['origin', 'destination']).ngroup()).values
df['PEroutepass'] = df.groupby(['origin', 'destination'])['PEroutepass'].transform('first')
df['PEroutepasssq'] = df['PEroutepass'] ** 2

# Calculate phase1 variables for existing columns
phase1_vars = {
    'INCpresORGvary': 'INCpresORG_phase1',
    'INCpresDESTvary': 'INCpresDEST_phase1',
    'WNpresORGvary': 'WNpresORG_phase1',
    'WNpresDESTvary': 'WNpresDEST_phase1',
    'HHIvary': 'HHI_phase1'
}

for orig_var, new_var in phase1_vars.items():
    df[new_var] = grouped.apply(lambda x: calc_phase1_var(x, orig_var)).reindex(df.groupby(['origin', 'destination']).ngroup()).values
    df[new_var] = df.groupby(['origin', 'destination'])[new_var].transform('first')
    if orig_var != 'HHIvary':
        df[new_var + 'sq'] = df[new_var] ** 2

# Long distance indicator (distance > 2000 km)
df['longdistance'] = (df['distance'] > 2000).astype(int)

# =====================================
# FIRST STAGE PROBIT (for predicted probabilities)
# =====================================
# Keep one observation per market for probit
probit_df = df[df['dupid']].copy()
probit_df = probit_df[(probit_df['QasaPE'] != 1) & 
                       (probit_df['QasaPE'] < 69) & 
                       (probit_df['dominant_route'] == 0)]

# Prepare probit variables
probit_vars = ['distance_km', 'distsq', 'longdistance', 'avgpop_mil', 'popsq',
                'PEroutepass', 'PEroutepasssq', 'slot', 'tourist', 'big_city',
                'second_Q', 'HHI_phase1', 'primary_airportO', 'primary_airportD',
                'secondary_airportO', 'secondary_airportD',
                'INCpresORG_phase1', 'INCpresORG_phase1sq',
                'INCpresDEST_phase1', 'INCpresDEST_phase1sq',
                'WNpresORG_phase1', 'WNpresORG_phase1sq',
                'WNpresDEST_phase1', 'WNpresDEST_phase1sq']

# Add quarter dummies (quarters 2-24 as in Stata)
for q in range(2, 25):
    probit_df[f'quarter_{q}'] = (probit_df['QasaPE'] == q).astype(int)
    probit_vars.append(f'quarter_{q}')

# Run probit and get predicted probabilities
from statsmodels.discrete.discrete_model import Probit
X_probit = probit_df[probit_vars].dropna()
y_probit = probit_df.loc[X_probit.index, 'entry4']
X_probit = sm.add_constant(X_probit)

probit_model = Probit(y_probit, X_probit)
probit_results = probit_model.fit()
probit_df.loc[X_probit.index, 'phat'] = probit_results.predict(X_probit)

# Merge predicted probabilities back to main dataset
df = df.merge(probit_df[['origin', 'destination', 'phat']], 
              on=['origin', 'destination'], 
              how='left')

# =====================================
# FILTER DATA FOR SECOND STAGE
# =====================================
# Keep only dominant firms
df = df[df['dominant_firm'] == 1]

# Keep only pre-entry observations
df = df[(df['quarter'] < df['WN_Qentered']) | (df['WN_Qentered'] == 0)]

# Remove missing phat values
df = df.dropna(subset=['phat'])

# =====================================
# CREATE FIXED EFFECTS AND CONTROL VARIABLES
# =====================================
df['market_FE'] = df.groupby(['origin', 'destination']).ngroup()
df['firm_FE'] = df.groupby('carrier').ngroup()
df['marketcarrier_FE'] = df.groupby(['origin', 'destination', 'carrier']).ngroup()

# Create control variable (threat period indicator)
df['notice'] = 0  # As specified in Stata code
df['control'] = 0

# Control = 1 if in threat period (after PE announcement but before entry)
df.loc[((df['quarter'] >= df['QasaPE'] - df['notice']) & 
        (df['quarter'] < df['WN_Qentered'])) |
       ((df['quarter'] >= df['QasaPE'] - df['notice']) & 
        (df['WN_Qentered'] == 0)), 'control'] = 1

# Apply WNEXIT condition
if 'WNEXIT' in df.columns:
    df.loc[df['quarter'] > df['WNEXIT'], 'control'] = 0

# =====================================
# CREATE DEPENDENT VARIABLE
# =====================================


# =====================================
# PREPARE VARIABLES FOR REGRESSION
# =====================================

# Remove rows with missing values for key variables
required_vars = ['routefare', 'control', 'fuel', 'fueldist', 'deviationsq',
                 'competitor_d', 'competitor_i', 'PotEntrants_AND', 
                 'market_FE', 'quarter']
df_clean = df.dropna(subset=required_vars)



# Prepare dependent variable
y_col1 = df_clean['routefare'].values

# Prepare independent variables
# Main variable of interest: control (threat period)
# Control variables from the regression
X_vars = ['control', 'fuel', 'fueldist', 'deviationsq',
          'competitor_d', 'competitor_i', 'PotEntrants_AND']
X_col1 = df_clean[X_vars].copy()

# Add quarter fixed effects
quarter_dummies = pd.get_dummies(df_clean['quarter'], prefix='quarter', drop_first=True)
X_col1 = pd.concat([X_col1, quarter_dummies], axis=1)

# Add market_FE for absorption
X_col1['market_FE'] = df_clean['market_FE'].values

# Add constant
X_col1 = sm.add_constant(X_col1, prepend=False)

# Fixed effects to absorb: market_FE
fe_vars = ['market_FE']

# =====================================
# METADATA FOR THIS RESULT
# =====================================
metadata_col1 = {
    'paper_id': '065',
    'table_id': '3',
    'panel_identifier': 'c1',
    'model_type': 'log-linear',
    'comments': 'Table 3 Column 1: Effect of Southwest entry threat on incumbent fares, dominant firms only, with market FE'
}

# =====================================
# RUN THE REPLICATION
# =====================================
replicate(
    metadata=metadata_col1,
    y=y_col1,
    X=X_col1,
    interest=['control'],  # Main coefficient of interest
    elasticity=False,
    fe=fe_vars,  # Market fixed effects
)