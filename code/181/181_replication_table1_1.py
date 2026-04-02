import numpy as np
import os
import sys
import json
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from replication import replicate
from utils.json_utils import ReplicationJSONBuilder, load_config

# Load configuration
config = load_config()
output_folder = config.get('output', './output')

# =====================================
# Paper 181 - Table 1, Column 1
# DV: ln(COW_importsUSA / total_gdp)
# newey2 ln_COW_importsUSA USinfluence RUSinfluence ln_per_capita_income
#        new_leader_dum year_in_office cgv_democracy i.year i.isocode
#        if year<=1989 & year>=1947, lag(40)
# =====================================

# Load data
df = pd.read_stata('/Users/ellenmunroe/Downloads/112606-V1/MS_AER-2010-0514-R3_Replication_Files/Country_year_dataset_final.dta')

# Drop USA and Russia
df = df[~df['isocode'].isin(['USA', 'RUS'])].copy()

# Drop countries with no import data
counts = df.groupby('isocode')['COW_importsWORLD'].transform('count')
df = df[counts > 0].copy()

# Drop missing isocode/year
df = df.dropna(subset=['isocode', 'year']).copy()

# Sample restriction: 1947-1989
df = df[(df['year'] >= 1947) & (df['year'] <= 1989)].copy()

# Create DV: ln(imports from USA / total GDP)
df['ln_COW_importsUSA'] = np.log(df['COW_importsUSA'] / df['total_gdp'])

# DV in levels for pipeline: imports from USA / total GDP
df['COW_importsUSA_ratio'] = df['COW_importsUSA'] / df['total_gdp']

# Drop observations where log is -inf (zero imports)
df = df[np.isfinite(df['ln_COW_importsUSA'])].copy()

# Regressors
controls = ['USinfluence', 'RUSinfluence', 'ln_per_capita_income',
            'new_leader_dum', 'year_in_office', 'cgv_democracy']

# Year FE
year_dummies = pd.get_dummies(df['year'].astype(int), prefix='yr', drop_first=True).astype(float)
year_cols = list(year_dummies.columns)
df = pd.concat([df.reset_index(drop=True), year_dummies.reset_index(drop=True)], axis=1)

# Country FE (isocode)
# Store as column for parquet; pipeline handles dummification
df['isocode_num'] = df['isocode'].astype('category').cat.codes

# Drop missing
sample_vars = ['COW_importsUSA_ratio', 'ln_COW_importsUSA'] + controls + year_cols + ['isocode_num']
df = df.dropna(subset=sample_vars).reset_index(drop=True)

# =====================================
# Table 1, Column 1
# =====================================

metadata = {
    'paper_id': '181',
    'table_id': 'Table1',
    'panel_identifier': 'Col1',
    'model_type': 'log-linear'
}

# y in levels (imports/GDP ratio)
y = pd.DataFrame(df['COW_importsUSA_ratio'].values, columns=['imports_usa_gdp'])

# X: interest variable first (USinfluence), then controls, year FE, country FE
x_cols = controls + year_cols + ['isocode_num']
X = df[x_cols].reset_index(drop=True)

# Save to parquet
out_dir = os.path.join(output_folder, '181', 'Table1_Col1')
os.makedirs(out_dir, exist_ok=True)

y.to_parquet(os.path.join(out_dir, 'y.parquet'), index=False)
X.to_parquet(os.path.join(out_dir, 'X.parquet'), index=False)

with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

# replicate()
replicate(
    metadata=metadata,
    y=y.values,
    X=X[controls].values,
    interest='USinfluence',
    endog_x=None,
    z=None,
    fe=X[year_cols + ['isocode_num']].values,
    elasticity=True,
    replicated=True,
    kwargs_estimator={'estimator_type': 'ols'},
    kwargs_ols=None,
    kwargs_ppml=None,
    fit_full_model=False,
    output=True,
    output_dir=output_folder
)
