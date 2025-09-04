import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# Load schooling data
filepath_school = os.path.join(INPUT_DATA_DIR, '060/schooling_withnames.dta')
df_school = pd.read_stata(filepath_school)
df_school = df_school[['countrycode', 'schooling']]

# Load R&D stock data
filepath_rd = os.path.join(INPUT_DATA_DIR, '060/data_chh_2009.dta')
df_rd = pd.read_stata(filepath_rd)
df_rd = df_rd[df_rd['year'] == 2004].copy()

# Map country codes
country_mapping = {
    1: "AUS", 2: "AUT", 3: "BEL", 4: "CAN", 5: "DNK",
    6: "FIN", 7: "FRA", 8: "DEU", 9: "GRC", 10: "ISL",
    11: "IRL", 12: "ISR", 13: "ITA", 14: "JPN", 15: "KOR",
    16: "NLD", 17: "NZL", 18: "NOR", 19: "PRT", 20: "ESP",
    21: "SWE", 22: "CHE", 23: "GBR", 24: "USA"
}
df_rd['name'] = df_rd['country'].map(country_mapping)
df_rd = df_rd.rename(columns={'sd': 'rd_stock'})[['name', 'rd_stock']]

# =====================================
# Load and process trade flows matrix
# =====================================

# Load the trade flows matrix (your file)
filepath_trade_matrix = os.path.join(INPUT_DATA_DIR, '060/Trade flows, market prices, origin is row, destination is column.dta')
df_trade_matrix = pd.read_stata(filepath_trade_matrix)

# Set first column as index if needed
if df_trade_matrix.index.name is None:
    first_col = df_trade_matrix.columns[0]
    if df_trade_matrix[first_col].dtype == 'object':
        df_trade_matrix = df_trade_matrix.set_index(first_col)

# Remove 'total' column if it exists
if 'total' in df_trade_matrix.columns:
    df_trade_matrix = df_trade_matrix.drop('total', axis=1)

# Convert matrix to bilateral format
countries = df_trade_matrix.index.tolist()
df_bilateral = []

for orig in countries:
    for dest in df_trade_matrix.columns:
        if dest in countries:
            trade_value = df_trade_matrix.loc[orig, dest]
            df_bilateral.append({
                'orig_name': orig.upper(),  # Convert to uppercase to match other data
                'dest_name': dest.upper(),
                'trade': trade_value
            })

df_trade = pd.DataFrame(df_bilateral)

# =====================================
# Load the full GTAP dataset for other variables
# =====================================

# Load the full trade data with gravity variables
filepath_gtap = os.path.join(INPUT_DATA_DIR, '060/trade_2004_gtap.dta')
df_gtap = pd.read_stata(filepath_gtap)

# Merge our trade matrix with GTAP gravity variables (excluding trade values)
# Keep all gravity variables except 'trade' from GTAP
gtap_vars = [col for col in df_gtap.columns if col != 'trade']
df = df_trade.merge(df_gtap[gtap_vars], on=['orig_name', 'dest_name'], how='left')

# Filter by GDP threshold
df = df[(df['gdp_o'] >= 0) & (df['gdp_d'] >= 0)].copy()

# Merge R&D data
df = df.merge(df_rd.rename(columns={'name': 'orig_name', 'rd_stock': 'orig_rd_stock'}), 
              on='orig_name', how='left')
df = df.merge(df_rd.rename(columns={'name': 'dest_name', 'rd_stock': 'dest_rd_stock'}), 
              on='dest_name', how='left')

# Merge schooling data
df = df.merge(df_school.rename(columns={'countrycode': 'orig_name', 'schooling': 'orig_schooling'}),
              on='orig_name', how='left')
df = df.merge(df_school.rename(columns={'countrycode': 'dest_name', 'schooling': 'dest_schooling'}),
              on='dest_name', how='left')

# Create trade flow variables
df['ln_dist'] = np.log(df['distw'])

# Get own trade flows
df['own_trade'] = df.apply(lambda x: x['trade'] if x['orig_name'] == x['dest_name'] else np.nan, axis=1)
df['temp2'] = df.groupby('dest_name')['own_trade'].transform('mean')

# Normalized trade flows
df['ln_Xij_Xjj'] = np.log(df['trade']) - np.log(df['temp2'])

# Fix variables for own trade flows
df.loc[df['orig_name'] == df['dest_name'], 'distw'] = 0
df.loc[df['orig_name'] == df['dest_name'], 'ln_dist'] = 0
df.loc[df['orig_name'] == df['dest_name'], 'comcur'] = 0

# Create continent fixed effects
if 'cont_dceania_d' in df.columns:
    df = df.rename(columns={'cont_dceania_d': 'cont_oceania_d'})

# Create continent_d variable
df['continent_d'] = 0
df.loc[df['cont_africa_d'] == 1, 'continent_d'] = 1
df.loc[df['cont_asia_d'] == 1, 'continent_d'] = 2
df.loc[df['cont_europe_d'] == 1, 'continent_d'] = 3
df.loc[df['cont_oceania_d'] == 1, 'continent_d'] = 4
df.loc[df['cont_north_america_d'] == 1, 'continent_d'] = 5
df.loc[df['cont_south_america_d'] == 1, 'continent_d'] = 6

# Create continent_o variable
df['continent_o'] = 0
df.loc[df['cont_africa_o'] == 1, 'continent_o'] = 1
df.loc[df['cont_asia_o'] == 1, 'continent_o'] = 2
df.loc[df['cont_europe_o'] == 1, 'continent_o'] = 3
df.loc[df['cont_oceania_o'] == 1, 'continent_o'] = 4
df.loc[df['cont_north_america_o'] == 1, 'continent_o'] = 5
df.loc[df['cont_south_america_o'] == 1, 'continent_o'] = 6

# Create continent pair FE
df['FEcontpair'] = df.groupby(['continent_o', 'continent_d']).ngroup()

# Create distance bins using percentiles
df['FE2dist'] = 0
mask_not_own = df['orig_name'] != df['dest_name']
df.loc[mask_not_own, 'FE2dist'] = pd.qcut(df.loc[mask_not_own, 'distw'], 
                                           q=10, labels=False) + 1

# Create country dummies for gravity regression
unique_countries = sorted(df['dest_name'].unique())
for country in unique_countries:
    df[f'pi_{country}'] = 0
    df.loc[df['dest_name'] == country, f'pi_{country}'] = 1
    df.loc[df['orig_name'] == country, f'pi_{country}'] = -1

# Create interaction terms for distance and continent pairs
for dist in df['FE2dist'].unique():
    for cont in df['FEcontpair'].unique():
        col_name = f'dist{dist}_cont{cont}'
        df[col_name] = ((df['FE2dist'] == dist) & (df['FEcontpair'] == cont)).astype(int)

# Prepare variables for gravity regression
interaction_cols = [col for col in df.columns if col.startswith('dist') and '_cont' in col]
pi_vars = [f'pi_{c}' for c in unique_countries]

# Run gravity regression
y_gravity = df['ln_Xij_Xjj']
X_gravity = df[interaction_cols + pi_vars]

model_gravity = sm.OLS(y_gravity, X_gravity)
results_gravity = model_gravity.fit()

# Extract country fixed effects
country_fe_dict = {}
for country in unique_countries:
    pi_var = f'pi_{country}'
    if pi_var in results_gravity.params.index:
        country_fe_dict[country] = results_gravity.params[pi_var]
    else:
        country_fe_dict[country] = 0

# Create country-level dataset
df_country = df.drop_duplicates(subset=['dest_name']).copy()
df_country['dest_pi'] = df_country['dest_name'].map(country_fe_dict)

# Calculate lambda (own expenditure share)
df_trade_totals = df.groupby('dest_name').agg({
    'trade': 'sum',
    'own_trade': 'first'
}).reset_index()
df_trade_totals['lambda'] = df_trade_totals['own_trade'] / df_trade_totals['trade']

# Merge lambda with country data
df_country = df_country.merge(df_trade_totals[['dest_name', 'lambda']], on='dest_name', how='left')

# Create variables for regression - keep Y in levels
df_country['Y'] = df_country['gdp_d']  # Income in levels (from GTAP data)
df_country['ln_Y'] = np.log(df_country['Y'])  # Log income for regression
df_country['ln_lambda'] = np.log(df_country['lambda'])  # Log own expenditure share

# Remove any rows with missing values
df_country = df_country.dropna(subset=['dest_pi', 'ln_Y', 'ln_lambda'])

# =====================================
# Table 2, Column 1: OLS Regression
# =====================================

# Dependent variable: gravity fixed effects (dest_pi is ln p, so we need p in levels)
# Convert from log to levels for the replication function
y_col1 = np.exp(df_country['dest_pi'].values)

# Independent variables: log income and log own expenditure share  
X_col1 = df_country[['ln_Y', 'ln_lambda']].values
X_col1 = sm.add_constant(X_col1)

# Create DataFrame with proper column names
X_col1_df = pd.DataFrame(X_col1, columns=['const', 'ln_Y', 'ln_lambda'])

# Metadata
metadata_col1 = {
    'paper_id': '060',
    'table_id': '2',
    'panel_identifier': 'col1_ols',
    'model_type': 'log-linear',
    'comments': 'Table 2 Column 1: OLS regression of log gravity fixed effects (ln p_i) on log income and log own expenditure share, no controls, robust standard errors'
}

# Run replication
replicate(
    metadata=metadata_col1,
    y=y_col1,
    X=X_col1_df,
    interest=['ln_Y', 'ln_lambda'],
    elasticity=False,
    fe=None,

)