import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['intermediatedata']

# LOAD THE FINAL DATASET WITH CORRECT QIJ VALUES
savingssub = pd.read_csv(os.path.join(INPUT_DATA_DIR, '024/savingssub_final_table1.csv'))

# Remove any missing values (don't artificially limit to 422)
savingssub = savingssub.dropna(subset=['SE_logsavingbal_saver', 'qij_ARD', 'qij_TRUE_correct'])

# OUTCOME: Convert log savings to LEVELS
y = np.exp(savingssub['SE_logsavingbal_saver'].values)

# Check if qij values are already standardized
# If mean ≈ 0 and std ≈ 1, use directly; otherwise standardize
if abs(savingssub['qij_ARD'].mean()) < 0.1 and abs(savingssub['qij_ARD'].std() - 1) < 0.1:
    # Already standardized, use directly
    X_ard = sm.add_constant(savingssub[['qij_ARD']].values)
else:
    # Need to standardize
    savingssub['qij_ARD_std'] = (savingssub['qij_ARD'] - savingssub['qij_ARD'].mean()) / savingssub['qij_ARD'].std()
    X_ard = sm.add_constant(savingssub[['qij_ARD_std']].values)

# METADATA for Column 2
metadata_col2 = {
    'paper_id': '024',
    'table_id': '1',
    'panel_identifier': '2',
    'model_type': 'log-linear',
    'comments': 'Table 1 Column 2: Log total savings on predicted signaling value with ARD'
}

# RUN REPLICATION for Column 2
replicate(
    metadata=metadata_col2,
    y=y,
    X=X_ard,
    interest=1,
    elasticity=False,
    kwargs_ols={'cov_type': 'cluster', 'cov_kwds': {'groups': savingssub['village'].values}},
)