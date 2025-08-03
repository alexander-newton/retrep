from loglinearcorrection.correction_estimator import DoublyRobustElasticityEstimator
import configparser
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
    def __init__(self, metadata,  y, X, interest, fe = None, z = None, elasticity=False, replicated=False, kwargs_estimator=None, kwargs_fit=None):
        self.metadata = metadata
        self.elasticity = elasticity
        self.replicated = replicated
        self.kwargs_estimator = kwargs_estimator if kwargs_estimator is not None else {'estimator_type':'ols'}
        self.kwargs_fit = kwargs_fit if kwargs_fit is not None else {'compute_asymptotic_variance': False}
        self.estimator =  DoublyRobustElasticityEstimator(
            endog=y,
            exog=X,
            interest=interest,
            fe=fe,
            instruments=z,
            elasticity=elasticity,
            **self.kwargs_estimator
        )

    def replicate_ols(self, weights=None, **kwargs):
        results_ols = self.estimator._base_ols_fit(weights=weights, **kwargs)
        return results_ols

    def replicate_ppml(self, weights=None, **kwargs):
        results_ppml = self.estimator._base_ppml_fit(weights=weights, **kwargs)
        return results_ppml

    def replicate(self, weights=None, **kwargs):
        pass

    def save_output(self, output):
        """
        Save the output to a file or database.
        """
        # Implement saving logic here
        pass




# FOR KIMIA:

def replicate(metadata, y, X, interest, fe=None, z=None, elasticity=False, replicated=False, kwargs_estimator=None, kwargs_fit=None, kwargs_ols=None, kwargs_ppml=None, fit_full_model=False):
    """
    Replicate the results using the provided metadata and data.
    """
    # Add covtype hc3 here maybe?
    kwargs_ols = kwargs_ols if kwargs_ols is not None else {}
    kwargs_ppml = kwargs_ppml if kwargs_ppml is not None else {}

    replicator = Replicator(metadata, y, X, interest, fe=fe, z=z, elasticity=elasticity, replicated=replicated, kwargs_estimator=kwargs_estimator, kwargs_fit=kwargs_fit)
    ols_results = replicator.replicate_ols(weights=None, **kwargs_ols)
    print('OLS res:')
    ppml_results = replicator.replicate_ppml(weights=None, **kwargs_ppml)
    if fe is not None:
        print('PPML results currently inaccurate with fixed effects')

    if fit_full_model:
        full_results = replicator.estimator.fit(**replicator.kwargs_fit)
        return ols_results, ppml_results, full_results
    else:
        return ols_results, ppml_results

