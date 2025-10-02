import os
import yaml
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ppml import fit_ppml_or_ivppml
from loglinearcorrection.correction_estimator import DoublyRobustElasticityEstimator

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)
DIR = config['outputdata']

def process_single_result(result_path, paper_id):
    """
    Worker function to process a single result path.
    Returns a dictionary of results or None if processing fails.
    """
    try:
        stripped_path = result_path.split('\\')
        panel_id = stripped_path[-1]
        
        print(f"Processing paper {paper_id}, {panel_id}")
        
        # Check if results already exist
        new_path = os.path.join(result_path, 'replication_results.pkl')
        if os.path.exists(new_path):
            print(f"Skipping {paper_id}/{panel_id} - results already exist")
            return None
        
        # Load data
        metadata = os.path.join(result_path, 'metadata.json')
        X = pd.read_parquet(os.path.join(result_path, 'X.parquet'))
        y = pd.read_parquet(os.path.join(result_path, 'y.parquet'))
        
        if os.path.exists(os.path.join(result_path, 'z.parquet')):
            z = pd.read_parquet(os.path.join(result_path, 'z.parquet'))
        else:
            z = None
        
        with open(metadata, 'r') as f:
            metadata = json.load(f)
        
        print(f"Metadata for {paper_id}/{panel_id}:", metadata)
        
        # Run replication
        results = replicate_full(X, y, z, metadata)
        
        # Save results
        pd.Series(results).to_pickle(new_path)
        
        # Add identifiers
        results['paper'] = paper_id
        results['panel'] = panel_id
        
        return results
        
    except Exception as e:
        print(f"Error processing {result_path}: {e}")
        return None

def collect_result_paths(DIR):
    """
    Collect all result paths that need to be processed.
    Returns a list of tuples: (result_path, paper_id)
    """
    files = os.listdir(DIR)
    all_paths = []
    
    for file in files:
        if file.endswith('.ini'):
            continue
        
        folder_path = os.path.join(DIR, file)
        if not os.path.isdir(folder_path):
            continue
            
        results_paths = [
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if not f.endswith('.ini') and os.path.isdir(os.path.join(folder_path, f))
        ]
        
        for result_path in results_paths:
            all_paths.append((result_path, file))
    
    return all_paths

def run_replications(DIR, n_jobs=-1, verbose=10):
    """
    Run replications in parallel across all papers.
    
    Parameters:
    -----------
    DIR : str
        Directory containing the data
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means use all available cores.
    verbose : int, default=10
        Verbosity level for joblib
    """
    # Collect all paths to process
    result_paths = collect_result_paths(DIR)
    print(f"Found {len(result_paths)} result paths to process")
    
    # Process in parallel
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(process_single_result)(path, paper_id) 
        for path, paper_id in result_paths
    )
    
    # Filter out None results (failures or skipped)
    all_results = [r for r in results if r is not None]
    
    # Save aggregated results
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv('./trial_results_se.csv', index=False)
        print(f"\nSuccessfully processed {len(all_results)} replications")
        print(f"Results saved to ./trial_results_se.csv")
    else:
        print("\nNo new results to save")
    
    return all_results

def replicate_full(X, y, z, metadata):
    """
    Replicate the full analysis for a single dataset.
    (Unchanged from original)
    """
    elasticity = metadata.get('elasticity', False)
    interest = metadata.get('interest', [0])
    weights = metadata.get('weights', None)
    
    if weights is not None:
        weights = np.array(float(weights))
    
    fe = metadata.get('fe', None)
    fe = [int(fixed_effect) for fixed_effect in fe] if fe is not None else None
    kwargs_ols = metadata.get('kwargs_ols', {})
    endog_x = metadata.get('endogenous_regressors', None)
    
    estimator = DoublyRobustElasticityEstimator(
        endog=y, 
        exog=X, 
        interest=interest, 
        endog_x=endog_x, 
        instruments=z, 
        fe=fe, 
        estimator_type='nn', 
        elasticity=elasticity, 
        weights=weights
    )
    
    fit = estimator.fit(method='ols', bootstrap=True, bootstrap_reps=100)
    print(fit.summary())
    
    res_dict = {
        'beta': fit.parametric_coef[fit.interest[0]],
        'beta_se': fit.parametric_results.bse[fit.interest[0]],
        'average_estimate': fit.average_estimate(),
        'estimate_at_average': fit.estimate_at_average(),
        'bootstrap_se': fit.bootstrap_se_dict[fit.interest[0]][2],
        'correction': np.mean(fit.correction[fit.interest[0]]),
    }
    
    print(res_dict)
    return res_dict

if __name__ == "__main__":
    # Run with all available cores, adjust n_jobs as needed
    run_replications(DIR, n_jobs=25, verbose=10)
