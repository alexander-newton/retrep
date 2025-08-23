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
# LOAD DATA
# =====================================
# Using Expanded_dataset which contains the available variables
filepath = os.path.join(INPUT_DATA_DIR, '045/Expanded_dataset.dta')
df = pd.read_stata(filepath)

# =====================================
# CREATE NECESSARY VARIABLES FROM AVAILABLE DATA
# =====================================

# Create GScites_mean (asinh transformation of Google Scholar citations)
df['GScites_mean'] = np.arcsinh(df['GScites_2019'])

# Create Year from yearsubmit
df['Year'] = df['yearsubmit']

# Create Journal_ID from Journal name
df['Journal_ID'] = pd.Categorical(df['Journal']).codes

# Create NAuthors from auth_count
df['NAuthors'] = df['auth_count']

# Rename Pubs35_l5yr to Pubs35_l5yr_AuthMax (assuming they're the same)
df['Pubs35_l5yr_AuthMax'] = df['Pubs35_l5yr']

# Create JEL field fractions with proper names (renaming from fr_* to JEL_*_fr)
jel_mapping = {
    'fr_micro': 'JEL_Micro_fr',
    'fr_theory': 'JEL_Theory_fr', 
    'fr_metrics': 'JEL_Metrics_fr',
    'fr_macro': 'JEL_Macro_fr',
    'fr_internat': 'JEL_Intl_fr',
    'fr_fin': 'JEL_Fin_fr',
    'fr_pub': 'JEL_Public_fr',
    'fr_labor': 'JEL_Labor_fr',
    'fr_healthurblaw': 'JEL_HealthUrbanLaw_fr',
    'fr_hist': 'JEL_Hist_fr',
    'fr_io': 'JEL_IO_fr',
    'fr_dev': 'JEL_Dev_fr',
    'fr_lab': 'JEL_Exp_fr',
    'fr_other': 'JEL_Other_fr'
}

for old_name, new_name in jel_mapping.items():
    if old_name in df.columns:
        df[new_name] = df[old_name]

# =====================================
# DATA FILTERING
# =====================================

# For Column 2, we don't need DROP_NDR filtering since we don't have it
# Just drop missing GScites_mean
df = df.dropna(subset=['GScites_mean'])

# Keep only years 2003-2013 (if specified in the paper)
# Uncomment if needed:
# df = df[(df['Year'] >= 2003) & (df['Year'] <= 2013)]

# =====================================
# VARIABLE PREPARATION (using EXACT Stata variable names for spec2)
# =====================================

# Create gender dummies from Gender_Auth_2 (drop first for reference category)
# Assuming Gender_Auth_2 codes: 1=All male (reference), 2=All female, 3=Mixed senior female, 4=Mixed other, 5=Undetermined
gender_dummies = pd.get_dummies(df['Gender_Auth_2'], prefix='Gender_Auth_2', drop_first=True)
df = pd.concat([df, gender_dummies], axis=1)

# Create author publication dummies from Pubs35_l5yr_AuthMax
# Creating categories: 0 (reference), 1, 2, 3, 4-5, 6+
df['Pubs35_l5yr_AuthMax'] = df['Pubs35_l5yr_AuthMax'].fillna(0)
pub_dummies = pd.get_dummies(df['Pubs35_l5yr_AuthMax'].astype(int), prefix='Pubs35_l5yr_AuthMax', drop_first=True)
df = pd.concat([df, pub_dummies], axis=1)

# Create number of authors dummies from NAuthors
nauth_dummies = pd.get_dummies(df['NAuthors'].astype(int), prefix='NAuthors', drop_first=True)
df = pd.concat([df, nauth_dummies], axis=1)

# Create Journal-Year interaction dummies (i.Year#i.Journal_ID in Stata)
df['Year_Journal'] = df['Year'].astype(str) + '_' + df['Journal_ID'].astype(str)
year_journal_dummies = pd.get_dummies(df['Year_Journal'], prefix='YearXJournal', drop_first=True)
df = pd.concat([df, year_journal_dummies], axis=1)

# =====================================
# Define variable groups for Column 2 (spec2 in Stata)
# =====================================

# Gender variables (dummies created from i.Gender_Auth_2)
Genders = [col for col in df.columns if col.startswith('Gender_Auth_2_')]

# Author publications (dummies created from i.Pubs35_l5yr_AuthMax)
AuthPubs = [col for col in df.columns if col.startswith('Pubs35_l5yr_AuthMax_')]

# Number of authors (dummies created from i.NAuthors)
NAuthors = [col for col in df.columns if col.startswith('NAuthors_')]

# Field variables (JEL fields + gender field controls)
Fields = ['JEL_Micro_fr', 'JEL_Theory_fr', 'JEL_Metrics_fr', 'JEL_Macro_fr',
          'JEL_Intl_fr', 'JEL_Fin_fr', 'JEL_Public_fr', 'JEL_Labor_fr',
          'JEL_HealthUrbanLaw_fr', 'JEL_Hist_fr', 'JEL_IO_fr', 'JEL_Dev_fr',
          'JEL_Exp_fr', 'JEL_Other_fr', 'Field_frF_5yr', 'Field_frGender']

# Year-Journal interaction dummies
Years = [col for col in df.columns if col.startswith('YearXJournal_')]

# =====================================
# Column 2 Specification (spec2 in Stata)
# =====================================
# spec2 = Genders + AuthPubs + NAuthors + Fields + Years

spec2 = Genders + AuthPubs + NAuthors + Fields + Years

# Filter to available columns
available_vars = [v for v in spec2 if v in df.columns]

# Clean data
df_clean = df.dropna(subset=['GScites_mean'] + available_vars[:5])

# =====================================
# RUN OLS FOR COLUMN 2
# =====================================

# Dependent variable in asinh form
y = df_clean['GScites_mean']

# Independent variables for column 2
X = sm.add_constant(df_clean[available_vars], prepend=False)

# Identify interest variables (Gender dummies)
interest_indices = [i for i, col in enumerate(X.columns) if col in Genders]

# Identify fixed effects (Year-Journal dummies)
fe_indices = [i for i, col in enumerate(X.columns) if col in Years]

# =====================================
# METADATA
# =====================================
metadata = {
    'paper_id': '045',
    'table_id': 'IV',
    'panel_identifier': '2',
    'model_type': 'asinh-linear',
    'comments': 'Table IV Column 2: GScites_mean (asinh) with gender, author pubs, fields, and year-journal FE'
}

# =====================================
# RUN THE REPLICATION
# =====================================
print("\n" + "="*60)
print("Replicating Table IV, Column 2")
print("="*60)
print(f"Sample size: {len(df_clean)}")
print(f"Number of regressors: {X.shape[1]}")
print(f"Dependent variable: GScites_mean (asinh of Google Scholar citations)")
print(f"Specification: Gender + AuthPubs + NAuthors + Fields + Years")
print("Note: No clustering since we don't have editor IDs")


