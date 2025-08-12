import numpy as np
from loglinearcorrection.correction_estimator import DoublyRobustElasticityEstimator
import os
import sys
from datetime import datetime
import json
import pandas as pd
import yaml
from pathlib import Path


# Add repository root to path to import modules
# Goes up two levels from papers/004/ to reach the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from replication import replicate
from utils.json_utils import ReplicationJSONBuilder, load_config

# Load configuration
config = load_config()
raw_data_folder = config.get('rawdata', './rawdata')
intermediate_data_folder = config.get('intermediatedata', './intermediate_data')
final_data_folder = config.get('finaldata', './final_data')
output_folder = config.get('output', './output')

# Load data
data_path = os.path.join(intermediate_data_folder, 'AG_Corp_Prod_Database.dta')
df = pd.read_stata(data_path)



# Initialize storage for JSON output
json_output = {
    "paper_id": "004",
    "tables": []
}

# Function to capture results and create JSON entry
def create_json_entry(table_id, column, y, X, fe, interest_idx, 
                      fe_indices, iv_indices=None, elasticity=False, 
                      model_type="log-linear"):
    """Create a JSON entry for a single regression table"""
    
    # Combine X and FE for full X matrix
    if fe is not None:
        X_full = np.hstack([X.reshape(-1, 1) if X.ndim == 1 else X, fe])
    else:
        X_full = X.reshape(-1, 1) if X.ndim == 1 else X
    
    # Create the JSON entry according to specification
    entry = {
        "table_id": table_id,
        "column": column,
        "binary": 1,  # Form variable is binary
        "model": model_type,
        "elasticity": 1 if elasticity else 0,
        "FEs": fe_indices if fe_indices else [],
        "IVs": iv_indices if iv_indices else [],
        "interest": [interest_idx],  # Index of the variable of interest
        "y": y.flatten().tolist(),
        "X": [X_full[:, i].tolist() for i in range(X_full.shape[1])]
    }
    
    return entry

# =====================================
# Column 1: Revenue per Worker
# =====================================

metadata_col1 = {
    'paper_id': '004',
    'table_id': 'Table3',
    'panel_identifier': 'PanelB_Col1',
    'model_type': 'OLS'
}

y_col1 = df['RevperWorker'].values.reshape(-1, 1)
X_col1 = df[['Form']].values
fe_col1 = df[['ProvinceFactor', 'IndustryFactor', 'YearFactor']].values
cluster_col1 = df['RegIndYearGroup'].values

# Run the replication
replicate(
    metadata=metadata_col1,
    y=y_col1,
    X=X_col1,
    interest='Form',
    endog_x=None,
    z=None,
    fe=fe_col1,
    elasticity=False,
    replicated=True,
    kwargs_estimator={'estimator_type': 'ols'},
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col1}},
    kwargs_ppml=None,
    cluster=cluster_col1,
    comments='Table 3 Panel B Column 1: RevperWorker on Form with FE',
    fit_full_model=False,
    output=True,
    data_dir='./data',
    output_dir='./output'
)

# Create JSON entry for Column 1
json_entry_col1 = create_json_entry(
    table_id="3.B",  # Table 3 Panel B
    column=1,
    y=y_col1,
    X=X_col1,
    fe=fe_col1,
    interest_idx=0,  # Form is at index 0
    fe_indices=[1, 2, 3],  # Province, Industry, Year factors
    elasticity=False,
    model_type="log-linear"
)
json_output["tables"].append(json_entry_col1)

# =====================================
# Column 2: Power per Worker
# =====================================

metadata_col2 = {
    'paper_id': '004',
    'table_id': 'Table3',
    'panel_identifier': 'PanelB_Col2',
    'model_type': 'OLS'
}

y_col2 = df['PowerperWorker'].values.reshape(-1, 1)
X_col2 = df[['Form']].values
fe_col2 = df[['ProvinceFactor', 'IndustryFactor', 'YearFactor']].values
cluster_col2 = df['RegIndYearGroup'].values

# Run the replication
replicate(
    metadata=metadata_col2,
    y=y_col2,
    X=X_col2,
    interest='Form',
    endog_x=None,
    z=None,
    fe=fe_col2,
    elasticity=False,
    replicated=True,
    kwargs_estimator={'estimator_type': 'ols'},
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': cluster_col2}},
    kwargs_ppml=None,
    cluster=cluster_col2,
    comments='Table 3 Panel B Column 2: PowerperWorker on Form with FE',
    fit_full_model=False,
    output=True,
    data_dir='./data',
    output_dir='./output'
)

# Create JSON entry for Column 2
json_entry_col2 = create_json_entry(
    table_id="3.B",  # Table 3 Panel B
    column=2,
    y=y_col2,
    X=X_col2,
    fe=fe_col2,
    interest_idx=0,  # Form is at index 0
    fe_indices=[1, 2, 3],  # Province, Industry, Year factors
    elasticity=False,
    model_type="log-linear"
)
json_output["tables"].append(json_entry_col2)

# =====================================
# Save JSON Output
# =====================================
output_filename = os.path.join(output_folder, '004.json')

with open(output_filename, 'w') as f:
    json.dump(json_output, f, indent=2)
print(f"JSON output saved to {output_filename}")
