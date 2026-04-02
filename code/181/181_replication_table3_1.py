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
# Paper 181 - Table 3, Column 1
# DV: ln(industry imports from USA / total GDP)
# cnsreg ln_importsUSA RUSinfluence USinfluence USinfluence_x_US_RCA USinfluence_x_importer_RCA
#        US_RCA importer_RCA [controls] [multilateral_resistance] i.year i.isocode i.sitc2
#        if year<=1989 & year>=1947, cluster(isocode)
# Data: industry-level (country x year x sitc2)
# =====================================

base_path = '/Users/ellenmunroe/Downloads/112606-V1/MS_AER-2010-0514-R3_Replication_Files'

# ===== Step 1: Build world export shares by country-year-sitc2 =====
wexp = pd.read_stata(f'{base_path}/World_exports_by_industry_annual_final.dta')
wexp = wexp.dropna(subset=['sitc4', 'isocode', 'year'])
wexp = wexp[wexp['isocode'] != 'WORLD']
wexp['sitc2'] = wexp['sitc4'].str[:2]
wexp = wexp[~wexp['sitc2'].str.match(r'.A')]
wexp = wexp[~wexp['sitc2'].str.match(r'.X')]
for i in range(1, 10):
    wexp = wexp[wexp['sitc2'] != f'{i}0']

wexp_collapsed = wexp.groupby(['isocode', 'year', 'sitc2'])['exportsWORLD'].sum().reset_index()

total_exp = wexp_collapsed.groupby(['year', 'sitc2'])['exportsWORLD'].sum().reset_index()
total_exp.rename(columns={'exportsWORLD': 'total_exportsWORLD'}, inplace=True)
wexp_collapsed = wexp_collapsed.merge(total_exp, on=['year', 'sitc2'])
wexp_collapsed['world_export_share'] = wexp_collapsed['exportsWORLD'] / wexp_collapsed['total_exportsWORLD']

agg_exp = wexp_collapsed.groupby(['isocode', 'year'])['exportsWORLD'].sum().reset_index()
agg_exp.rename(columns={'exportsWORLD': 'aggregate_exportsWORLD'}, inplace=True)
total_agg = wexp_collapsed.groupby('year')['exportsWORLD'].sum().reset_index()
total_agg.rename(columns={'exportsWORLD': 'total_aggregate_exportsWORLD'}, inplace=True)
agg_exp = agg_exp.merge(total_agg, on='year')
agg_exp['aggregate_world_export_share'] = agg_exp['aggregate_exportsWORLD'] / agg_exp['total_aggregate_exportsWORLD']
wexp_collapsed = wexp_collapsed.merge(agg_exp[['isocode', 'year', 'aggregate_world_export_share']], on=['isocode', 'year'])

# US exports
us_exp = wexp_collapsed[wexp_collapsed['isocode'] == 'USA'][['year', 'sitc2', 'world_export_share', 'aggregate_world_export_share']].copy()
us_exp.columns = ['year', 'sitc2', 'US_world_export_share', 'US_aggregate_world_export_share']

world_exp = wexp_collapsed[wexp_collapsed['isocode'] != 'USA']

# ===== Step 2: Build imports by industry =====
imp = pd.read_stata(f'{base_path}/Imports_by_industry_annual_final.dta')
imp = imp.dropna(subset=['sitc4', 'isocode', 'year'])
imp['sitc2'] = imp['sitc4'].str[:2]
imp = imp[~imp['sitc2'].str.match(r'.A')]
imp = imp[~imp['sitc2'].str.match(r'.X')]
for i in range(1, 10):
    imp = imp[imp['sitc2'] != f'{i}0']

mean_cols = [c for c in ['USinfluence', 'RUSinfluence', 'cgv_democracy', 'ln_per_capita_income', 'total_gdp', 'year_in_office'] if c in imp.columns]
imp_agg = imp.groupby(['isocode', 'year', 'sitc2']).agg(
    {**{'importsUSA': 'sum', 'importsWORLD': 'sum'}, **{c: 'mean' for c in mean_cols}}
).reset_index()
imp_agg = imp_agg[imp_agg['isocode'] != 'WORLD']

# ===== Step 3: Merge and construct RCA =====
imp_agg = imp_agg.merge(world_exp[['isocode', 'year', 'sitc2', 'world_export_share', 'aggregate_world_export_share']],
                         on=['isocode', 'year', 'sitc2'], how='left')
imp_agg = imp_agg.merge(us_exp, on=['year', 'sitc2'], how='left')

imp_agg['US_RCA'] = imp_agg['US_world_export_share'] / imp_agg['US_aggregate_world_export_share']
imp_agg['importer_RCA'] = imp_agg['world_export_share'] / imp_agg['aggregate_world_export_share']
imp_agg = imp_agg.dropna(subset=['importsUSA', 'importsWORLD'])

# Normalize RCA to [0,1] using (x - min) / max (matching Stata code)
for v in ['US_RCA', 'importer_RCA']:
    mn = imp_agg[v].min()
    mx = imp_agg[v].max()
    imp_agg[v] = (imp_agg[v] - mn) / mx

# Drop USA and Russia
imp_agg = imp_agg[~imp_agg['isocode'].isin(['USA', 'RUS'])]

# Interactions
imp_agg['USinfluence_x_US_RCA'] = imp_agg['USinfluence'] * imp_agg['US_RCA']
imp_agg['USinfluence_x_importer_RCA'] = imp_agg['USinfluence'] * imp_agg['importer_RCA']

# New leader dummy
imp_agg['new_leader_dum'] = np.where(imp_agg['year_in_office'] == 1, 1,
                             np.where(imp_agg['year_in_office'].notna() & (imp_agg['year_in_office'] > 1), 0, np.nan))

# Sample restriction
imp_agg = imp_agg[(imp_agg['year'] >= 1947) & (imp_agg['year'] <= 1989)]
counts = imp_agg.groupby('isocode')['importsWORLD'].transform('count')
imp_agg = imp_agg[counts > 0]

# ===== Step 4: B&B correction terms =====
bb = pd.read_stata(f'{base_path}/Baier_Bergstrand_MR_approx_terms_final.dta')
imp_agg['importer_isocode'] = imp_agg['isocode']
imp_agg['exporter_isocode'] = 'USA'
imp_agg = imp_agg.merge(bb, on=['importer_isocode', 'exporter_isocode', 'year'], how='left', suffixes=('', '_bb'))

imp_agg['x1'] = imp_agg['ln_distance'] - imp_agg['distance_correction']
imp_agg['x2'] = imp_agg['contiguous'] - imp_agg['border_correction']
imp_agg['x3'] = imp_agg['same_language'] - imp_agg['language_correction']
imp_agg['x4'] = imp_agg['gatt'] - imp_agg['gatt_correction']
imp_agg['x5'] = imp_agg['rta'] - imp_agg['rta_correction']

# ===== Step 5: DV =====
imp_agg['ln_importsUSA'] = np.log(imp_agg['importsUSA'] / imp_agg['total_gdp'])
imp_agg['importsUSA_ratio'] = imp_agg['importsUSA'] / imp_agg['total_gdp']

# Drop inf (zero imports)
imp_agg = imp_agg[np.isfinite(imp_agg['ln_importsUSA'])].copy()
imp_agg = imp_agg[imp_agg['importsUSA_ratio'] > 0].copy()

# ===== Regression variables =====
controls = ['USinfluence', 'USinfluence_x_US_RCA', 'USinfluence_x_importer_RCA',
            'US_RCA', 'importer_RCA', 'RUSinfluence',
            'ln_per_capita_income', 'new_leader_dum', 'year_in_office', 'cgv_democracy',
            'x1', 'x2', 'x3', 'x4', 'x5']

# FE identifiers
imp_agg['year_int'] = imp_agg['year'].astype(int)
fe_cols = ['isocode', 'year_int', 'sitc2']

# Drop missing
sample_vars = ['importsUSA_ratio'] + controls + fe_cols
reg_df = imp_agg[sample_vars].dropna().reset_index(drop=True)

# =====================================
# Table 3, Column 1
# =====================================

metadata = {
    'paper_id': '181',
    'table_id': 'Table3',
    'panel_identifier': 'Col1',
    'model_type': 'log-linear'
}

# y in levels (imports/GDP ratio)
y = pd.DataFrame(reg_df['importsUSA_ratio'].values, columns=['imports_usa_gdp'])

# X: interest variable first (USinfluence), then controls, then FE identifiers
x_cols = controls + fe_cols
X = reg_df[x_cols].reset_index(drop=True)

# Save to parquet
out_dir = os.path.join(output_folder, '181', 'Table3_Col1')
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
    fe=X[fe_cols].values,
    elasticity=True,
    replicated=True,
    kwargs_estimator={'estimator_type': 'ols'},
    kwargs_ols=None,
    kwargs_ppml=None,
    fit_full_model=False,
    output=True,
    output_dir=output_folder
)
