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
        self.kwargs_ols = kwargs_ols if kwargs_ols is not None else {'cov_type':'HC3'}
        self.kwargs_ppml = kwargs_ppml if kwargs_ppml is not None else {'cov_type':'HC3'}
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

    def save_output(self, output_dir, overwrite=False):


        if not self.replicated:
            raise ValueError("Results must be replicated before saving output.")

        required_keys = ["paper_id", "table_id", "panel_identifier", "model_type"]
        missing = [k for k in required_keys if k not in self.metadata]
        if missing:
            raise ValueError(f"Missing required metadata keys: {missing}")

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

        results_folder = os.path.join(output_folder, f"result_table_{metadata_dict['table_id']}_{metadata_dict['panel_identifier']}")
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        metadata_file = os.path.join(results_folder, 'metadata.json')
        if os.path.exists(metadata_file) and not overwrite:
            raise FileExistsError(
                f"Metadata already exists at {metadata_file}. Set overwrite=True to replace."
            )

        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, default=str, indent=2)


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
             output_dir=None, overwrite=False):
    """
        Run a replication of an econometric result (OLS by default) using the provided
        metadata, data, and estimation settings.

        This is the main entry point for running replications in the project.
        It wraps the `Replicator` class to:
          1. Set up the model from your data and metadata
          2. Estimate results (currently OLS; PPML code is present but not active)
          3. Optionally save the replication bundle (metadata + minimal data) to disk

        Parameters
        ----------
        metadata : dict
            Information about the model/table being replicated.
            Must include: 'paper_id', 'table_id', 'panel_identifier', 'model_type'.
        y : array-like
            Dependent variable.
        X : array-like
            Independent variables matrix.
        interest : str or list
            Variable(s) of primary interest.
        endog_x : list of int or list of str, optional
            Endogenous regressors, specified either by column index (int) or
            column name (str). If provided, `z` must also be given.
        z : array-like, optional
            Instrumental variable matrix. Required if `endog_x` is not None.
        fe : array-like, optional
            Fixed effects identifiers.
        elasticity : bool, default False
            Whether to compute and report elasticities for variables of interest.
        replicated : bool, default False
            Set True if this replication worked out as expected.
        kwargs_estimator : dict, optional
            Extra keyword arguments for the estimator initialization.
        kwargs_fit : dict, optional
            Extra keyword arguments for the estimator's `.fit()` method.
        kwargs_ols : dict, optional
            Keyword arguments for OLS replication (e.g., `{'cov_type': 'HC3'}`).
        kwargs_ppml : dict, optional
            Keyword arguments for PPML replication (currently unused in this function).
        fit_full_model : bool, default False
            If True, placeholder for running a "full model" (not yet implemented).
        output : bool, default False
            If True, save replication metadata and data to `output_dir`.
        output_dir : str or Path, optional
            Directory where replication output will be saved. Required if `output=True`.
        overwrite : bool, default False
            If True, overwrite existing replication output files in `output_dir`.

        Raises
        ------
        ValueError
            If `output=True` but `output_dir` is not provided.

        Notes
        -----
        - This function currently only runs OLS. PPML replication is available
          in the `Replicator` class but commented out here until fixed effects
          support is finalised.
        - Saved output includes:
            * metadata.json — replication settings
            * y.parquet — dependent variable
            * X.parquet — independent variables
            * z.parquet — instruments (if provided)
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
        replicator.save_output(output_dir, overwrite=overwrite)
    return replicator, ols_results

