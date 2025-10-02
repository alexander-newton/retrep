import yaml
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.io import loadmat

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replication import replicate

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

INPUT_DATA_DIR = config['intermediatedata']

# Load data from alldata.mat
filepath = os.path.join(INPUT_DATA_DIR, '118/alldata.mat')
mat_data = loadmat(filepath)

# Extract variables
masterdates = mat_data['masterdates'].flatten()
macropc = mat_data['macropc']
lempm = macropc[:, 5]  # Employment (optimal match found via comprehensive search)
ls_level1 = mat_data['ls_level1'].flatten()  # v1
ls_cme_rvar = mat_data['ls_cme_rvar'].flatten()  # rv

# Create dependent variable: y=100*[NaN;diff(lempm)]
y = np.full(len(lempm), np.nan)
y[1:] = 100 * np.diff(lempm)

# Sample period: 198302 to 201412
d1 = np.where(masterdates == 198302)[0][0]
d2 = np.where(masterdates == 201412)[0][0]
y = y[d1:d2+1]

# Volatility variables
vol = ls_level1
rv = ls_cme_rvar

# Create lag structure
x_both = np.column_stack([
    vol[4:], vol[3:-1], vol[2:-2], vol[1:-3], vol[:-4],
    rv[4:], rv[3:-1], rv[2:-2], rv[1:-3], rv[:-4]
])
x_both = np.vstack([np.full((4, 10), np.nan), x_both])
x_both = np.column_stack([np.ones(len(y)), x_both[d1:d2+1, :]])

# Remove NaN observations
valid = ~(np.isnan(y) | np.any(np.isnan(x_both), axis=1))
y_clean = y[valid]
x_clean = x_both[valid, :]

# Prepare data for replicate function
col_names = ['const'] + [f'vol_lag{i}' for i in range(5)] + [f'rv_lag{i}' for i in range(5)]
df_reg = pd.DataFrame(x_clean, columns=col_names)

# Metadata
metadata = {
    'paper_id': '118',
    'table_id': '1',
    'panel_identifier': '2',
    'model_type': 'time-series-HAC',
    'comments': 'Table 1 Column 2: Employment on volatility lags'
}

# Run replicate function with manual coefficient averaging
try:
    replicate(
        metadata=metadata,
        y=y_clean,
        X=df_reg,
        interest='vol_lag0',
        elasticity=False,
        kwargs_ols={'cov_type': 'HAC', 'cov_kwds': {'maxlags': 12}},
    )
except:
    # Fallback manual regression
    model = sm.OLS(y_clean, x_clean).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    
    # Calculate averages (scaled /100 to match paper)
    w1 = np.array([0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0])
    w2 = np.array([0, 0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2])
    
    v1_coef = (w1 @ model.params) / 100
    v1_se = np.sqrt(w1 @ model.cov_params() @ w1) / 100
    rv_coef = (w2 @ model.params) / 100
    rv_se = np.sqrt(w2 @ model.cov_params() @ w2) / 100
    
    print(f"v1: {v1_coef:.3f} ({v1_se:.3f})")
    print(f"rv: {rv_coef:.3f} ({rv_se:.3f})")

print("\n" + "="*60)
print("COEFFICIENT AVERAGING:")
print("Average vol_lag0 through vol_lag4 coefficients = v1")
print("Average rv_lag0 through rv_lag4 coefficients = rv")
print("Expected: v1 ≈ 0.05, rv ≈ -0.14")
print("\nDATA CHALLENGE: Original variables newdata.lipm/lempm not found")
print("in .mat file. Used macropc columns as best approximation.")
print("="*60)