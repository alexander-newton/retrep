import yaml
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# -------------------------------
# Load config
# -------------------------------
with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# -------------------------------
# Load data
# -------------------------------
data_path = os.path.join(INPUT_DATA_DIR, '170/state_data.dta')
df = pd.read_stata(data_path)

# -------------------------------
# Stata-sample alignment:
# First keep 1939–2000 so L. controls exist for 1940, then exclude states, then build lags.
# Finally restrict to 1940–2000 for the regression.
# -------------------------------
df = df[(df['year'] >= 1939) & (df['year'] <= 2000)].copy()
df = df[~df['stateabbr'].isin(["", "HI", "AK", "LA"])].copy()

# Ensure proper sort for lags
df = df.sort_values(['statenum', 'year'])

# Build L. controls (within state)
df['L1_real_gdp_pc'] = df.groupby('statenum')['real_gdp_pc'].shift(1)
df['L1_population_density'] = df.groupby('statenum')['population_density'].shift(1)

# Analysis years
df = df[df['year'] >= 1940].copy()

# -------------------------------
# Transform tax variables to ln net-of-tax (ln(1 - tau)), tau is in percent
# Stata does: replace mtr90_lag3 = ln(1 - mtr90_lag3/100)
#             replace top_corp_lag3 = ln(1 - top_corp_lag3/100)
# -------------------------------
def ln_net(series):
    return np.log(1.0 - (series / 100.0))

df['ln_net_mtr90_lag3']   = ln_net(df['mtr90_lag3'])
df['ln_net_topcorp_lag3'] = ln_net(df['top_corp_lag3'])

# -------------------------------
# Two-way clustering keys: statenum × fiveyear, and year
# -------------------------------
df['fiveyear'] = (5 * np.floor(df['year'] / 5)).astype(int)
# create a numeric group id like egen group(statenum fiveyear)
st_fe = df['statenum'].astype(int).astype(str)
fy_fe = df['fiveyear'].astype(int).astype(str)
df['statenum_fiveyear_gid'] = (st_fe + "_" + fy_fe).astype('category').cat.codes.astype(int)

# -------------------------------
# Keep complete cases for regression variables
# -------------------------------
controls = ['rd_credit_lag3', 'L1_real_gdp_pc', 'L1_population_density']
main_vars = ['ln_net_mtr90_lag3', 'ln_net_topcorp_lag3']
need = ['lnpat', 'statenum', 'year', 'pop1940'] + controls + main_vars
df = df.dropna(subset=need).copy()

# -------------------------------
# Dependent variable: replicate() logs internally → pass exp(lnpat)
# -------------------------------
y = np.exp(df['lnpat'].to_numpy())

# -------------------------------
# Fixed effects: state and year (dummies)
# -------------------------------
fe_state = pd.get_dummies(df['statenum'].astype(int), prefix='fe_state', drop_first=True)
fe_year  = pd.get_dummies(df['year'].astype(int),      prefix='fe_year',  drop_first=True)
fe_vars = fe_state.columns.tolist() + fe_year.columns.tolist()

# -------------------------------
# Design matrix X = main vars + controls + FE
# -------------------------------
X = pd.concat([df[main_vars + controls], fe_state, fe_year], axis=1)
X = sm.add_constant(X, prepend=False)

# -------------------------------
# Weights and clustering (if your replicate() supports kwargs_fit)
# -------------------------------
aw = df['pop1940'].to_numpy()


# -------------------------------
# Metadata
# -------------------------------
metadata = {
    'paper_id': '170',
    'table_id': '2',
    'panel_identifier': 'A_1',
    'model_type': 'log-log',
    'comments': 'Table 2 Panel A Col (1): lnpat ~ ln(1 - mtr90_lag3) + ln(1 - top_corp_lag3) + rd_credit_lag3 + L.real_gdp_pc + L.population_density; state & year FE; [aw=pop1940]; two-way cluster.'
}

# -------------------------------
# Run replication
# interest: ln_net_mtr90_lag3 (personal net-of-tax elasticity)
# -------------------------------
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest='ln_net_mtr90_lag3',
    elasticity=False,
    fe=fe_vars,
    weights=aw,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': [df['statenum_fiveyear_gid'].to_numpy(), df['year'].astype(int).to_numpy()]}},
    output=True, output_dir=OUTPUT_DIR, replicated=True, overwrite=True,
)