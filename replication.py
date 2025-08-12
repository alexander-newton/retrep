import numpy as np
from loglinearcorrection.correction_estimator import DoublyRobustElasticityEstimator
import os
from datetime import datetime
import json
import pandas as pd
# VERY MUCH WORK IN PROGRESS

class Replicator:
    def __init__(self, metadata,  y, X, interest, fe = None, endog_x = None, z = None, elasticity=False, replicated=False, kwargs_estimator=None, kwargs_fit=None, kwargs_ols = None, kwargs_ppml=None, cluster=None, comments=None):
        self.y = y # Outcome of interest
        self.X = X # Exogenous variables, including interest and endogenous x variables
        self.instruments = z
        self.interest = interest # Name of the exogenous variable of interest
        self.cluster = cluster
        self.kwargs_ols = kwargs_ols if kwargs_ols is not None else {'cov_type':'HC0'}
        self.kwargs_ppml = kwargs_ppml if kwargs_ppml is not None else {'cov_type':'HC0'}
        self.comments = comments if comments is not None else 'No comments provided'

        if cluster is not None:
            print('Clustered standard errors not yet implemented, will use non-clustered standard errors')



        self.metadata = metadata
        self.elasticity = elasticity
        self.replicated = replicated
        self.kwargs_estimator = kwargs_estimator if kwargs_estimator is not None else {'estimator_type':'ols'}
        self.kwargs_fit = kwargs_fit if kwargs_fit is not None else {'compute_asymptotic_variance': False}
        self.estimator =  DoublyRobustElasticityEstimator(
            endog=y,
            exog=X,
            endog_x = endog_x,
            interest=interest,
            fe=fe,
            instruments=z,
            elasticity=elasticity,
            **self.kwargs_estimator
        )
# Make sure x and z are together
    def replicate_ols(self, weights=None):
        results_ols = self.estimator._fit_base_ols(weights=weights, **self.kwargs_ols)
        return results_ols

    def replicate_ppml(self, weights=None, **kwargs):
        results_ppml = self.estimator._fit_base_ppml(weights=weights, **self.kwargs_ppml)
        return results_ppml

    def save_output(self, output_dir):


        if not self.replicated:
            raise ValueError("Results must be replicated before saving output.")

        paper_id = self.metadata['paper_id']

        metadata_dict = {}
        metadata_dict['table_id'] = self.metadata['table_id']
        metadata_dict['panel_identifier'] = self.metadata['panel_identifier']
        metadata_dict['model_type'] = self.metadata['model_type']
        metadata_dict['elasticity'] = self.elasticity
        if isinstance(self.interest, np.ndarray):
            metadata_dict['interest'] = list(self.interest)
        elif isinstance(self.interest, list):
            metadata_dict['interest'] = self.interest
        else:
            metadata_dict['interest'] = [self.interest]
        metadata_dict['comments'] = self.comments

        if self.estimator.fe is not None:
            metadata_dict['fe'] = list(self.estimator.fe_indices)

        if self.estimator.instruments is not None:
            metadata_dict['endogenous_regressors'] = list(self.estimator.endog_x)

        metadata_dict['kwargs_ols'] = self.kwargs_ols
        metadata_dict['kwargs_ppml'] = self.kwargs_ppml
        self.metadata['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.cluster is not None:
            metadata_dict['cluster'] = list(self.cluster)

        # fnc to save metadata based on paper_id
        output_folder = os.path.join(output_dir, str(paper_id))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        results_folder = os.path.join(output_folder, f'result_table_{metadata_dict['table_id']}_{metadata_dict['panel_identifier']}')
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        metadata_file = os.path.join(results_folder, 'metadata.json')
        if os.path.exists(metadata_file):
            validation = input(f'METADATA FILE FOR RESULT EXISTS ALREADY, \n file_name = {metadata_file} \n If you want to proceed, type Y')
            if not validation.lower() == 'y':
                print('Terminating saving')
                return

        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, default=str)


        data_y = pd.DataFrame(self.y)
        data_X = pd.DataFrame(self.X)
        data_instruments = pd.DataFrame(self.estimator.instruments) if self.estimator.instruments is not None else None


        # save data as parquet
        data_y.to_parquet(os.path.join(results_folder, f'y.parquet'))
        data_X.to_parquet(os.path.join(results_folder, f'X.parquet'))

        if data_instruments is not None:
            data_instruments.to_parquet(os.path.join(results_folder, f'z.parquet'))



def replicate(metadata, y, X, interest, endog_x=None, z=None, fe=None, elasticity=False, replicated=False, kwargs_estimator=None,
              kwargs_fit=None, kwargs_ols=None, kwargs_ppml=None, fit_full_model=False, output=False,
             output_dir=None):
    """
    Replicate the results using the provided metadata and data.
    """
    # Add covtype hc3 here maybe?

    replicator = Replicator(metadata, y, X, interest, endog_x=endog_x, z=z, fe=fe, elasticity=elasticity, replicated=replicated, kwargs_estimator=kwargs_estimator, kwargs_fit=kwargs_fit, kwargs_ols=kwargs_ols, kwargs_ppml=kwargs_ppml)
    ols_results = replicator.replicate_ols(weights=None)
    print('OLS res:')
    print(ols_results.summary(yname=replicator.estimator.endog_names,xname=list(replicator.estimator.exog_names)))
    # ppml_results = replicator.replicate_ppml(weights=None)
    # print('PPML res:')
    # print(ppml_results.summary())
    # if fe is not None:
    #     print('PPML results currently inaccurate with fixed effects')

    if fit_full_model:
        print('Full model fitting will be added soon, patience!')

    if output:
        if output_dir is None:
            raise ValueError("Output directory must be specified if output is True.")
        replicator.save_output(output_dir)

