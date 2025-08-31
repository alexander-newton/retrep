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

# =====================================
# Paper 084 - Table 5, Column 4
# =====================================

# Load data
df = pd.read_stata(os.path.join(INPUT_DATA_DIR, '084/SCF_plus.dta'))

# Replace ffaequ = ffaequ + mfun (as in Stata)
df['ffaequ'] = df['ffaequ'] + df['mfun']

# First collapse: Calculate thresholds (p90 and means by year)
thresholds = []
for year in sorted(df['yearmerge'].unique()):
    year_data = df[df['yearmerge'] == year]
    valid_mask = ~year_data['wgtI95W95'].isna()
    year_data_valid = year_data[valid_mask]
    
    thresholds.append({
        'yearmerge': year,
        'p90wealth': np.quantile(year_data['ffanw'], 0.90),
        'meanincome': np.average(year_data_valid['tinc'], weights=year_data_valid['wgtI95W95']),
        'meanwealth': np.average(year_data_valid['ffanw'], weights=year_data_valid['wgtI95W95']),
        'meanhouse': np.average(year_data_valid['house'], weights=year_data_valid['wgtI95W95']),
        'meanstocks': np.average(year_data_valid['ffaequ'], weights=year_data_valid['wgtI95W95'])
    })
thresholds = pd.DataFrame(thresholds)

# Merge thresholds back and keep only top 10%
df = df.merge(thresholds, on='yearmerge')
df_top10 = df[df['ffanw'] > df['p90wealth']]

# Second collapse: Calculate top 10% statistics
collapsed = []
for year in sorted(df_top10['yearmerge'].unique()):
    year_data = df_top10[df_top10['yearmerge'] == year]
    valid_mask = ~year_data['wgtI95W95'].isna()
    year_data_valid = year_data[valid_mask]
    weights = year_data_valid['wgtI95W95']
    
    # Get threshold values (meanincome, meanwealth, etc. from first collapse)
    year_thresholds = thresholds[thresholds['yearmerge'] == year].iloc[0]
    
    collapsed.append({
        'year': year,  # Rename yearmerge to year here
        'p90wealth': np.average(year_data_valid['ffanw'], weights=weights),
        'p90income': np.average(year_data_valid['tinc'], weights=weights),
        'meanincome': year_thresholds['meanincome'],  # From first collapse
        'meanwealth': year_thresholds['meanwealth'],  # From first collapse
        'house': np.average(year_data_valid['house'], weights=weights),
        'stocks': np.average(year_data_valid['ffaequ'], weights=weights),
        'meanhouse': year_thresholds['meanhouse'],  # From first collapse
        'meanstocks': year_thresholds['meanstocks']  # From first collapse
    })
collapsed = pd.DataFrame(collapsed)

# Generate shares (as in Stata)
collapsed['top10wealthshare'] = collapsed['p90wealth'] / collapsed['meanwealth'] * 0.1
collapsed['top10incomeshare'] = collapsed['p90income'] / collapsed['meanincome'] * 0.1
collapsed['p90stockshare'] = collapsed['stocks'] / collapsed['p90wealth']
collapsed['p90houseshare'] = collapsed['house'] / collapsed['p90wealth']
collapsed['stockshare'] = collapsed['meanstocks'] / collapsed['meanwealth']
collapsed['houseshare'] = collapsed['meanhouse'] / collapsed['meanwealth']

# Merge with asset prices
assetprices = pd.read_stata(os.path.join(INPUT_DATA_DIR, '084/assetprices.dta'))
df_final = collapsed.merge(assetprices, on='year', how='inner')

# Sort by year and set timeline
df_final = df_final.sort_values('year').reset_index(drop=True)
df_final['timeline'] = range(1, len(df_final) + 1)
df_final['timeline2'] = df_final['timeline'] ** 2

# Generate variables (following Stata exactly)
df_final['wealthgrowth'] = df_final['meanwealth'].shift(-1) / df_final['meanwealth']
df_final['Y2W'] = df_final['meanincome'] / df_final['meanwealth']
df_final['p90Y2W'] = df_final['p90income'] / df_final['p90wealth']

# Growth variables (F. means forward/lead in Stata, which is shift(-1) in pandas)
df_final['stockgrowth'] = np.log(df_final['stockprice'].shift(-1) / df_final['stockprice'])
df_final['housegrowth'] = np.log(df_final['houseprice'].shift(-1) / df_final['houseprice'])
df_final['top10growth'] = df_final['top10wealthshare'].shift(-1) / df_final['top10wealthshare']
df_final['incineq'] = df_final['top10incomeshare']

# Prepare regression data for Column 4
# Stata: regress top10growth stockgrowth housegrowth incineq Y2W
# But table shows housegrowth first, so we order it that way
reg_data = df_final[['top10growth', 'housegrowth', 'stockgrowth', 'incineq', 'Y2W']].dropna()

# For replicate function: y needs to be in levels (it will take log)
# So we need to convert back from log growth to growth factor
y_col4 = reg_data['top10growth'].values
X_col4 = sm.add_constant(reg_data[['housegrowth', 'stockgrowth', 'incineq', 'Y2W']], prepend=False).values

# METADATA
metadata_col4 = {
    'paper_id': '084',
    'table_id': '5',
    'panel_identifier': '4',
    'model_type': 'log-linear',  # Replicate function will take log of y
    'comments': 'Table 5 Column 4: log(top10 wealth share growth) ~ housegrowth + stockgrowth + incineq + Y2W'
}

# RUN REPLICATION
replicate(
    metadata=metadata_col4,
    y=y_col4,
    X=X_col4,
    interest=[0, 1],  # housegrowth and stockgrowth
    elasticity=False,
    fe=None,

)