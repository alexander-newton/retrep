import os
import yaml
import json
import numpy as np
import pandas as pd
from ppml import fit_ppml_or_ivppml
from loglinearcorrection.correction_estimator import DoublyRobustElasticityEstimator

# Load configuration

with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

DIR = config['outputdata']


# get everything in DIR
def run_replications(DIR):
    files = os.listdir(DIR)
    all_results = []

    for file in files:
        if file.endswith('.ini'):
            continue

        folder_path = os.path.join(DIR, file)

        results_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if not f.endswith('.ini')]

        for result_path in results_paths:
            stripped_path = result_path.split('\\')

            # TODO: fix how kwargs are stored and loaded. Right now they are not stored appropriately.
            print(f"Processing paper {stripped_path[-2]}, {stripped_path[-1]}")
            # if stripped_path[-2] in ['018', '050']:
            #     print("Skipping paper for now due to data issues.")
            #     continue
            try:
                metadata = os.path.join(result_path, 'metadata.json')
                X = pd.read_parquet(os.path.join(result_path, 'X.parquet'))
                y = pd.read_parquet(os.path.join(result_path, 'y.parquet'))
                if os.path.exists(os.path.join(result_path, 'z.parquet')):
                    z = pd.read_parquet(os.path.join(result_path, 'z.parquet'))
                else:
                    z = None

                with open(metadata, 'r') as f:
                    metadata = json.load(f)
                print(metadata)
                results = replicate_full(X, y, z, metadata)

                new_path = os.path.join(result_path, 'replication_results.pkl')
                if os.path.exists(new_path):
                    os.remove(new_path)
                    continue
                pd.Series(results).to_pickle(new_path)

                results['paper'] = stripped_path[-2]
                results['panel'] = stripped_path[-1]

                all_results.append(results)
            except Exception as e:
                print(f"Error loading data for {result_path}: {e}")
                continue
        #
        # all_results_df = pd.DataFrame(all_results)
        # all_results_df.to_csv('./trial_results_se.csv')



def replicate_full(X,y, z, metadata):
    elasticity = metadata.get('elasticity', False)
    interest = metadata.get('interest', [0])
    weights = metadata.get('weights', None)
    if weights is not None:
        weights = np.array(float(weights))

    fe = metadata.get('fe', None)
    fe = [int(fixed_effect) for fixed_effect in fe] if fe is not None else None
    kwargs_ols = metadata.get('kwargs_ols', {})
    endog_x = metadata.get('endogenous_regressors', None)

    # nnparams = {
    #     'num_layers': 8,
    #     'num_units': 128
    #
    # }

    estimator = DoublyRobustElasticityEstimator(endog=y, exog=X, interest=interest, endog_x=endog_x, instruments=z, fe=fe, estimator_type='nn', elasticity=elasticity, weights=weights)
    fit = estimator.fit(method='ols', bootstrap=True, bootstrap_reps=100)
    print(fit.summary())
    # res_ppml = fit_ppml_or_ivppml(y, X, instruments=z, fe_cols=fe, endog_cols=endog_x, interest_var=interest, hdfe=fe is not None, return_meta=True)
    res_dict = {
        'beta': fit.parametric_coef[fit.interest[0]],
        'beta_se': fit.parametric_results.bse[fit.interest[0]],
        'average_estimate': fit.average_estimate(),
        'estimate_at_average': fit.estimate_at_average(),
        'bootstrap_se': fit.bootstrap_se_dict[fit.interest[0]][2],
        'correction': np.mean(fit.correction[fit.interest[0]]),
        # 'ppml_beta': res_ppml[2]['beta']
    }
    print(res_dict)
    return res_dict

if __name__ == "__main__":
    run_replications(DIR)






