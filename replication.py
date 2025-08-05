from loglinearcorrection.correction_estimator import DoublyRobustElasticityEstimator
import os
from datetime import datetime
# VERY MUCH WORK IN PROGRESS

"""
Metadata format:
{
'PaperID' : Numeric,
'Table': Numeric,
'Result Number': Numeric,
'Binary_x': Boolean
}
"""
class Replicator:
    def __init__(self, metadata,  y, X, interest, fe = None, endog_x = None, z = None, elasticity=False, replicated=False, kwargs_estimator=None, kwargs_fit=None, kwargs_ols = None, kwargs_ppml=None, cluster=None, comments=None):
        self.y = y # Outcome of interest
        self.X = X # Exogenous variables, including interest and endogenous x variables
        self.instruments = z
        self.interest = interest # Name of the exogenous variable of interest
        self.cluster = cluster
        self.kwargs_ols = kwargs_ols if kwargs_ols is not None else {'cov_type':'HC0_se'}
        self.kwargs_ppml = kwargs_ppml if kwargs_ppml is not None else {'cov_type':'HC0_se'}
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
        results_ols = self.estimator._base_ols_fit(weights=weights, **self.kwargs_ols)
        return results_ols

    def replicate_ppml(self, weights=None, **kwargs):
        results_ppml = self.estimator._base_ppml_fit(weights=weights, **self.kwargs_ppml)
        return results_ppml

    def save_output(self, data_dir, output_dir):


        if not self.replicated:
            raise ValueError("Results must be replicated before saving output.")

        paper_id = self.metadata['paper_id']

        metadata_dict = {}
        metadata_dict['table_id'] = self.metadata['table_id']
        metadata_dict['panel_identifier'] = self.metadata['panel_identifier']
        metadata_dict['model_type'] = self.metadata['model_type']
        metadata_dict['elasticity'] = self.elasticity
        metadata_dict['interest'] = self.interest
        metadata_dict['comments'] = self.comments

        if self.estimator.fe is not None:
            metadata_dict['fe'] = self.estimator.fe_indices

        if self.estimator.instruments is not None:
            metadata_dict['endogenous_regressors'] = self.estimator.endog_x

        metadata_dict['kwargs_ols'] = self.kwargs_ols
        metadata_dict['kwargs_ppml'] = self.kwargs_ppml
        self.metadata['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.cluster is not None:
            metadata_dict['cluster'] = self.cluster

        # fnc to save metadata based on paper_id

        # fnc to save data based on paper_id
        data_y = self.y
        data_X = self.X
        data_instruments = self.estimator.instruments if self.estimator.instruments is not None else None

        # save data as parquet












def replicate(metadata, y, X, interest, fe=None, z=None, elasticity=False, replicated=False, kwargs_estimator=None, kwargs_fit=None, kwargs_ols=None, kwargs_ppml=None, fit_full_model=False, output=False, data_dir= None, output_dir=None):
    """
    Replicate the results using the provided metadata and data.
    """
    # Add covtype hc3 here maybe?

    replicator = Replicator(metadata, y, X, interest, fe=fe, z=z, elasticity=elasticity, replicated=replicated, kwargs_estimator=kwargs_estimator, kwargs_fit=kwargs_fit, kwargs_ols=kwargs_ols, kwargs_ppml=kwargs_ppml)
    ols_results = replicator.replicate_ols(weights=None, **kwargs_ols)
    print('OLS res:')
    print(ols_results.summary())
    ppml_results = replicator.replicate_ppml(weights=None, **kwargs_ppml)
    print('PPML res:')
    print(ppml_results.summary())
    if fe is not None:
        print('PPML results currently inaccurate with fixed effects')

    if fit_full_model:
        print('Full model fitting will be added soon, patience!')

    if output:
        if output_dir is None or data_dir is None:
            raise ValueError("Output directory and data directory must be specified if output is True.")
        replicator.save_output(data_dir, output_dir)

