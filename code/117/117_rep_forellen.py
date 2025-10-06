import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
import sys
#from replication import replicate

sys.path.append('/Users/ellenmunroe/Desktop/retrep')
from replication import replicate

import os
os.chdir('/Users/ellenmunroe/Desktop/retrep')
sys.path.append('/Users/ellenmunroe/Desktop/retrep')

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']


# LOAD DATA
#filepath = os.path.join(INPUT_DATA_DIR, '117/reg_data.dta')
#current filepath
#root_path = "/Users/ellenmunroe/Downloads"
##filepath = os.path.join(root_path, "/144601-V1-all/fires-hazard/02_Data/reg_data.dta")
df = pd.read_stata("/Users/ellenmunroe/Downloads/144601-V1-all/fires-hazard/02_Data/reg_data.dta")


# Convert categorical columns to numeric
for col in df.columns:
    if df[col].dtype.name == 'category':
        df[col] = df[col].cat.codes

print(df.columns)

# Define variables needed
sample_vars = ['lncost', 'cost', 
               'distanceNM', 'unitid', 'state', 'year', 'mofy']

# Drop missing values
df = df.dropna(subset=sample_vars)

# Create distance bins (equivalent to R's cut function and Stata's gen/replace)
def create_distance_bins(distance):
    """Create distance bins matching the Stata code"""
    # Use np.inf for the last bin to capture everything 40+
    bins = pd.cut(distance, bins=[0, 10, 20, 30, 40, np.inf], 
                  labels=['[0,10)', '[10,20)', '[20,30)', '[30,40)', '[40+)'],
                  include_lowest=True, right=False)
    return bins

df['distancebin'] = create_distance_bins(df['distanceNM'])

# Create dummy variables for distance bins, dropping the first (reference category)
distance_dummies = pd.get_dummies(df['distancebin'], prefix='distance', drop_first=True)

# This creates: distance_[10,20), distance_[20,30), distance_[30,40), distance_[40+)
# Rename to cleaner names: distance1, distance2, distance3, distance4
distance_dummies.columns = ['distance1', 'distance2', 'distance3', 'distance4']

# Add to dataframe
df = pd.concat([df, distance_dummies], axis=1)

# Create dummy variables for distance bins (equivalent to tab distancebin, gen(dist_))
distance_dummies = pd.get_dummies(df['distancebin'], prefix='dist', drop_first=True)
df = pd.concat([df, distance_dummies], axis=1)


# Create interaction variables (equivalent to Stata's egen group)
# state_mofy = group(state mofy)
df['state_mofy'] = df.groupby(['state', 'mofy']).ngroup()

# state_year = group(state year)  
df['state_year'] = df.groupby(['state', 'year']).ngroup()

# Dependent variable - cost
y = df['cost']

# X matrix
X_vars = ['distance1', 'distance2', 'distance3', 'distance4', 'unitid', 'state_mofy', 'state_year']
X_col1 = sm.add_constant(df[X_vars], prepend=False)
print(X_col1)

# Cluster variable
cluster_col1 = df['unitid']

print(df.columns)
print(df['state_mofy'])

# METADATA
metadata_col1 = {
    'paper_id': '117',
    'table_id': '1',
    'panel_identifier': 'A',
    'model_type': 'log-lin'
}


# RUN REPLICATION
replicate(
    metadata=metadata_col1,
    y=y,
    X=X_col1,
    interest='distance1',
    elasticity=True,
    fe=['unitid', 'state_mofy', 'state_year'],
    #kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col1}},
    output=True, output_dir=OUTPUT_DIR, replicated=True,
    overwrite =True
)


replicator, ols_results = replicate(
    metadata=metadata_col1,
    y=y,
    X=X_col1,
    interest='distance1',
    elasticity=True,
    fe=['unitid', 'state_mofy', 'state_year'],
    kwargs_estimator={
        'estimator_type': 'nn',
        'nn_params': {
            'hidden_layers': [64, 32],
            'epochs': 100,
            'learning_rate': 0.001,
            'batch_size': 32
        },
        'density_estimator': 'nn',
        'density_params': {
            'hidden_layers': [32, 16],
            'epochs': 50
        }
    },
    kwargs_fit={'compute_asymptotic_variance': False},
    output=False,  # Skip saving for now
    replicated=True
)

# Now access the full doubly robust results
dr_results = replicator.estimator.fit(method='ols', compute_asymptotic_variance=False)
print("\nDoubly Robust Neural Network Results:")
print(dr_results.summary())