import yaml
from replication import replicate
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np




with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)


OUTPUT_DIR = config['outputdata']
INPUT_DATA_DIR = config['rawdata'] # Could be intermediate data as well

# LOAD DATA, happy if you want to use a function

filepath = os.path.join(INPUT_DATA_DIR, '011/Radio_LRA_DB125.dta')
df = pd.read_stata(filepath)


# COPYING YOUR CODE
# Btw in this case I wouldn't have done so many results - probably just the one with only treatment, and the one with most controls

df_filtered = df[df['year'] > 2007].copy()


# Define dependent variables (LRA fatalities)
# DOING ONLY 1 DEPENDENT VARIABLE FOR NOW
dependent_vars = ['lnC_LRAfatalities']

# Define control variables
dist_controls = ['min_dist','min_dist2','med_dist','med_dist2',
    'min_dist_rugged','min_dist2_rugged','med_dist_rugged','med_dist2_rugged']

# Main treatment variable
main_treatment = 'stdmessagingpct_fn'

# PREPARE DATA FOR REGRESSION TABLE 1 PANEL A

y_A = df_filtered[dependent_vars[0]]

control_set = ['cell_id'] + dist_controls + ['year']

# For now try to keep treatment variable as the first one in the X matrix so its easy for you to see. I will try to add names here.
X_A = sm.add_constant(df_filtered[[main_treatment] + control_set], prepend=False)




# NOW BEFORE EVERY REGRESSION YOU DO, NEED TO PROVIDE A SET OF METADATA (can be done in a loop for multiple tables)

metadata_result_1 = {
    'paper_id': '011Example', #Put only the 3 digit code, here more because example.
    'table_id': '2',
    'panel_identifier': 'A1_2', # THIS WILL NEED TO CHANGE FOR EACH RESULT
    'model_type': 'semi-elasticity',
    'comments': 'Replication of Table 2 from Armand et al. (2020)' # WHATEVER YOU THINK IS RELEVANT
}

# Note y has to be the raw variable (i.e not logged)
replicate(metadata=metadata_result_1, y=np.exp(y_A), X=X_A, interest='stdmessagingpct_fn', elasticity=False, fe=['cell_id', 'year'])

# Find attached PDF for more info on this function

# If you want to save the results, you can do so by setting output=True. DO THIS ONLY AFTER YOU ARE HAPPY WITH THE RESULTS
# Do these modifications in the previous command
# In particular, specify an output directory, specify output=True, and confirm replicated=True
# This will save the results in the output directory specified in the config file.
# It will save metadata as well as the data.


replicate(metadata=metadata_result_1, y=np.exp(y_A), X=X_A, interest='stdmessagingpct_fn', elasticity=False, fe=['cell_id', 'year'],
          output=True, output_dir=OUTPUT_DIR, replicated=True)



