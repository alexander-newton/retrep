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
filepath = os.path.join(INPUT_DATA_DIR, '066/Accounts_matched_beforeafter.dta')
df_ba = pd.read_stata(filepath)

# =====================================
# Table 5 Column 3: Log employment on CEO behavior interaction with after
# eststo: xi: areg ly c.ceo_behavior##after lemp lempm cons active i.year i.cty emp_imputed $noise_basic_collapse [aw=r_averagewk], rob abs(company_id)
# =====================================

# Define noise controls
noise_controls = ['pa', 'reliability']
noise_controls += [col for col in df_ba.columns if col.startswith('ww')]
noise_controls += [col for col in df_ba.columns if col.startswith('aa')]


# Collapse by cty, company_id, sic, after
agg_dict = {
    'emp_imputed': 'max',
    'ly': 'mean',
    'ceo_behavior': 'mean',
    'lemp': 'mean',
    'cons': 'mean',
    'active': 'mean',
    'year': 'mean',
    'r_averagewk': 'mean'
}
for col in noise_controls:
    if col in df_ba.columns:
        agg_dict[col] = 'mean'

df = df_ba.groupby(['cty', 'company_id', 'sic', 'after']).agg(agg_dict).reset_index()



# Process variables - match exact Stata order
df['year'] = df['year'].astype(int)  # replace year=int(year)
df.loc[df['lemp'].isna(), 'lemp'] = -99  # replace lemp=-99 if lemp==.
# cap drop lempm (already handled by groupby)
df['lempm'] = (df['lemp'] == -99).astype(float)  # gen lempm=lemp==-99

# Create interaction term for c.ceo_behavior##after
df['ceo_behavior_after'] = df['ceo_behavior'] * df['after']

# Filter to remove missing values
required_vars = ['ly', 'ceo_behavior', 'after', 'lemp', 'lempm', 'cons', 'active', 
                'year', 'cty', 'emp_imputed', 'company_id', 'r_averagewk'] + noise_controls
df_clean = df.dropna(subset=[v for v in required_vars if v in df.columns])

# Remove singletons BEFORE regression to avoid weight mismatch
# Identify singleton groups in company_id (the absorbed fixed effect)
company_counts = df_clean.groupby('company_id').size()
singleton_companies = company_counts[company_counts == 1].index
non_singleton_mask = ~df_clean['company_id'].isin(singleton_companies)

# Filter data to remove singletons
df_clean = df_clean[non_singleton_mask].copy()
print(f"Removed {(~non_singleton_mask).sum()} singleton company observations")
print(f"Final sample size: {len(df_clean)} observations")

# Dependent variable
y = np.exp(df_clean['ly'])

# Variables in exact Stata order: c.ceo_behavior##after creates main effects + interaction
# Then: lemp lempm cons active emp_imputed $noise_basic_collapse
stata_order_vars = ['ceo_behavior', 'after', 'ceo_behavior_after', 'lemp', 'lempm', 'cons', 'active', 'emp_imputed']

# RESTORE: Back to original specification that gave correct lemp coefficient
fe_variables = ['year', 'cty', 'company_id']  # Absorb company_id like original

# Create X matrix in exact Stata order - core variables only
all_vars = stata_order_vars + noise_vars + fe_variables
X = sm.add_constant(df_clean[all_vars], prepend=False)

# Weights
weights = df_clean['r_averagewk']

# METADATA
metadata = {
    'paper_id': '066',
    'table_id': '5',
    'panel_identifier': '3',
    'model_type': 'log-linear',
    'comments': 'Table 5 Column 3: Log employment on CEO behavior interaction with after, with year, cty FE and company_id absorption, weighted by r_averagewk'
}



# Maybe we need to manually handle the multicollinearity differently
# Let's see if we can modify the replication function behavior
replicate(
    metadata=metadata,
    y=y,
    X=X,
    interest=['ceo_behavior'],
    fe=fe_variables,  # Back to absorbing company_id
    weights=weights,
    #output=True, output_dir=OUTPUT_DIR, replicated=True
)