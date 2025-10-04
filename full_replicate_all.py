import os
import yaml
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

print("=" * 60)
print("Multi-GPU Setup with Loky Backend (True Process Isolation)")
print("=" * 60)

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)
DIR = config['outputdata']

# Make sure 'ppml.py' is in the same directory or accessible in the Python path
def run_ppml_estimation(X, y, z, metadata):
    """
    Runs only the PPML estimation using the provided ppml.py module.
    
    Returns a dictionary with PPML beta and standard error.
    """
    from ppml import fit_ppml_or_ivppml

    interest_var = metadata.get('interest', [0])
    if isinstance(interest_var, list) and len(interest_var) == 1:
        interest_var = interest_var[0]

    fe_cols = metadata.get('fe', None)
    endog_cols = metadata.get('endogenous_regressors', None)

    try:
        print("Running fit_ppml_or_ivppml...")
        # We assume hdfe=True is the desired default for PPML with fixed effects
        ppml_res, _, interest_dict = fit_ppml_or_ivppml(
            y=y,
            X=X,
            instruments=z,
            fe_cols=fe_cols,
            endog_cols=endog_cols,
            interest_var=interest_var,
            hdfe=True,
            return_meta=True
        )
        
        if interest_dict and 'beta' in interest_dict and 'se' in interest_dict:
            print("PPML estimation successful.")
            return {
                'ppml_beta': interest_dict['beta'],
                'ppml_se': interest_dict['se']
            }
        else:
            print("PPML estimation ran but failed to find the interest variable.")
            return {'ppml_beta': np.nan, 'ppml_se': np.nan}

    except Exception as e:
        print(f"An error occurred during PPML estimation: {e}")
        return {'ppml_beta': np.nan, 'ppml_se': np.nan}


def process_single_result(result_path, paper_id, gpu_id):
    """
    Worker function to process a single result path.
    Runs in an isolated process with its own TensorFlow instance.
    """
    # CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing TensorFlow
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_CUDA_COMPUTE_CAPABILITIES'] = '12.0'
    os.environ['CUDA_CACHE_MAXSIZE'] = '2147483648'

    # NOW import TensorFlow (will only see GPU specified above)
    import tensorflow as tf
    from loglinearcorrection.correction_estimator import DoublyRobustElasticityEstimator

    # Configure TensorFlow for this worker
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"[Process {os.getpid()}] GPU {gpu_id} - TensorFlow sees {len(gpus)} GPU(s): {gpus}")
    else:
        print(f"[Process {os.getpid()}] WARNING: GPU {gpu_id} assigned but TensorFlow sees no GPUs!")

    # Set thread limits for this process
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    try:
        stripped_path = result_path.split('\\')
        panel_id = stripped_path[-1]

        print(f"[GPU {gpu_id}, PID {os.getpid()}] Processing paper {paper_id}, {panel_id}")

        main_required_fields = [
            'beta', 'beta_se', 'average_estimate', 'ae_bootstrap_se',
            'conf_int_05', 'conf_int_95', 'estimate_at_average',
            'ea_bootstrap_se', 'ea_conf_05', 'ea_conf_95',
            'correction', 'correction_se', 'correction_05', 'correction_95',
            'semi_elasticity', 'semi_elasticity_se', 'semi_elasticity_ci_05', 'semi_elasticity_ci_95'
        ]
        ppml_required_fields = ['ppml_beta', 'ppml_se']

        # Check if results already exist and handle different scenarios
        new_path = os.path.join(result_path, 'replication_results.pkl')
        if os.path.exists(new_path):
            try:
                existing_results = pd.read_pickle(new_path)
                if isinstance(existing_results, pd.Series):
                    existing_results = existing_results.to_dict()

                missing_main = any(field not in existing_results for field in main_required_fields)
                missing_ppml = any(field not in existing_results for field in ppml_required_fields)

                if not missing_main and not missing_ppml:
                    print(f"[GPU {gpu_id}] Skipping {paper_id}/{panel_id} - complete results already exist.")
                    return None
                
                elif not missing_main and missing_ppml:
                    print(f"[GPU {gpu_id}] Updating {paper_id}/{panel_id} with PPML results only.")
                    # Load data just for the PPML estimation
                    X = pd.read_parquet(os.path.join(result_path, 'X.parquet'))
                    y = pd.read_parquet(os.path.join(result_path, 'y.parquet'))
                    z = pd.read_parquet(os.path.join(result_path, 'z.parquet')) if os.path.exists(os.path.join(result_path, 'z.parquet')) else None
                    with open(os.path.join(result_path, 'metadata.json'), 'r') as f:
                        metadata = json.load(f)
                    
                    # Run only PPML
                    ppml_results = run_ppml_estimation(X, y, z, metadata)
                    
                    # Add new results and save
                    existing_results.update(ppml_results)
                    pd.Series(existing_results).to_pickle(new_path)
                    print(f"[GPU {gpu_id}] Successfully updated {paper_id}/{panel_id} with PPML results.")
                    return None # Stop further processing for this file

                else:
                    print(f"[GPU {gpu_id}] Re-estimating {paper_id}/{panel_id} - main results are missing.")
                    os.remove(new_path)
            except Exception as e:
                print(f"[GPU {gpu_id}] Error reading existing results for {paper_id}/{panel_id}: {e}. Re-estimating.")
                os.remove(new_path)

        # Load data for a full run
        metadata = os.path.join(result_path, 'metadata.json')
        X = pd.read_parquet(os.path.join(result_path, 'X.parquet'))
        y = pd.read_parquet(os.path.join(result_path, 'y.parquet'))
        z = pd.read_parquet(os.path.join(result_path, 'z.parquet')) if os.path.exists(os.path.join(result_path, 'z.parquet')) else None
        with open(metadata, 'r') as f:
            metadata = json.load(f)

        print(f"[GPU {gpu_id}] Metadata for {paper_id}/{panel_id}:", metadata)

        # Run full replication (which now includes PPML)
        results = replicate_full(X, y, z, metadata, gpu_id)

        # Save results
        pd.Series(results).to_pickle(new_path)
        
        results['paper'] = paper_id
        results['panel'] = panel_id
        results['gpu_used'] = gpu_id
        results['process_id'] = os.getpid()

        return results

    except Exception as e:
        print(f"[GPU {gpu_id}] Error processing {result_path}: {e}")
        import traceback
        traceback.print_exc()
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

def run_replications(DIR, n_jobs=30, verbose=10, start_from=None, exclude_papers = ['141']):
    """
    Run replications in parallel across all papers using Loky backend.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(gpus)

    print(f"\nConfiguration:")
    print(f"  Available GPUs: {num_gpus}")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"  Jobs per GPU: ~{n_jobs / num_gpus:.1f}")
    print(f"  Backend: loky (true process isolation)")
    print()

    result_paths = collect_result_paths(DIR)
    
    if exclude_papers is None:
        exclude_papers = []
    
    if exclude_papers:
        result_paths = [(path, paper_id) for path, paper_id in result_paths if paper_id not in exclude_papers]
        print(f"Excluded papers: {', '.join(exclude_papers)}")

    if start_from is not None:
        result_paths = [(path, paper_id) for path, paper_id in result_paths if paper_id >= start_from]
        print(f"Starting from paper {start_from}")

    print(f"Found {len(result_paths)} result paths to process")

    jobs_with_gpu = [(path, paper_id, i % num_gpus) for i, (path, paper_id) in enumerate(result_paths)]

    print(f"\nGPU assignment summary:")
    gpu_counts = {i: 0 for i in range(num_gpus)}
    for _, _, gpu_id in jobs_with_gpu:
        gpu_counts[gpu_id] += 1
    for gpu_id, count in sorted(gpu_counts.items()):
        print(f"  GPU {gpu_id}: {count} jobs queued")
    print()

    results = Parallel(n_jobs=n_jobs, verbose=verbose, backend='loky')(
        delayed(process_single_result)(path, paper_id, gpu_id)
        for path, paper_id, gpu_id in jobs_with_gpu
    )

    all_results = [r for r in results if r is not None]

    if all_results:
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv('./trial_results_se.csv', index=False)
        print(f"\nSuccessfully processed {len(all_results)} new replications")
        print(f"Results saved to ./trial_results_se.csv")
    else:
        print("\nNo new results were generated")

    return all_results

def replicate_full(X, y, z, metadata, gpu_id):
    """
    Replicate the full analysis for a single dataset, now including PPML.
    """
    import tensorflow as tf
    import numpy as np
    from loglinearcorrection.correction_estimator import DoublyRobustElasticityEstimator

    elasticity = metadata.get('elasticity', False)
    interest = metadata.get('interest', [0])
    weights = metadata.get('weights', None)
    if weights is not None:
        weights = np.array(float(weights))
    fe = [int(f) for f in metadata.get('fe', [])] if metadata.get('fe') is not None else None
    endog_x = metadata.get('endogenous_regressors', None)

    estimator = DoublyRobustElasticityEstimator(
        endog=y, exog=X, interest=interest, endog_x=endog_x,
        instruments=z, fe=fe, estimator_type='nn',
        elasticity=elasticity, weights=weights
    )

    fit = estimator.fit(method='ols', bootstrap=True, bootstrap_reps=100)
    print(fit.summary())

    semi_elasticity_point_estimate = fit.parametric_coef[fit.interest[0]]
    bootstrap_betas = fit.bootstrap_estimates_dict[fit.interest[0]][:, 0]
    semi_elasticity_bootstrap_se = fit.bootstrap_se_dict[fit.interest[0]][0]
    semi_elasticity_ci_05 = semi_elasticity_point_estimate - np.percentile(bootstrap_betas, 97.5)
    semi_elasticity_ci_95 = semi_elasticity_point_estimate - np.percentile(bootstrap_betas, 2.5)

    res_dict = {
        'beta': fit.parametric_coef[fit.interest[0]],
        'beta_se': fit.parametric_results.bse[fit.interest[0]],
        'average_estimate': fit.average_estimate(),
        'ae_bootstrap_se': fit.bootstrap_se_dict[fit.interest[0]][2],
        'conf_int_05': fit.average_estimate() - np.percentile(fit.bootstrap_estimates_dict[fit.interest[0]][:,2],97.5),
        'conf_int_95': fit.average_estimate() - np.percentile(fit.bootstrap_estimates_dict[fit.interest[0]][:,2],2.5),
        'estimate_at_average': fit.estimate_at_average(),
        'ea_bootstrap_se': fit.bootstrap_se_dict[fit.interest[0]][3],
        'ea_conf_05': fit.estimate_at_average()-np.percentile(fit.bootstrap_estimates_dict[fit.interest[0]][:,3],97.5),
        'ea_conf_95': fit.estimate_at_average()-np.percentile(fit.bootstrap_estimates_dict[fit.interest[0]][:,3],2.5),
        'correction': np.mean(fit.correction[fit.interest[0]]),
        'correction_se': fit.bootstrap_se_dict[fit.interest[0]][1],
        'correction_05': np.mean(fit.correction[fit.interest[0]]) - np.percentile(fit.bootstrap_estimates_dict[fit.interest[0]][:,1],97.5),
        'correction_95': np.mean(fit.correction[fit.interest[0]]) - np.percentile(fit.bootstrap_estimates_dict[fit.interest[0]][:,1],2.5),
        'semi_elasticity': semi_elasticity_point_estimate,
        'semi_elasticity_se': semi_elasticity_bootstrap_se,
        'semi_elasticity_ci_05': semi_elasticity_ci_05,
        'semi_elasticity_ci_95': semi_elasticity_ci_95,
    }

    # Add PPML estimation to the full run
    ppml_results = run_ppml_estimation(X, y, z, metadata)
    res_dict.update(ppml_results)

    print(res_dict)
    return res_dict

def aggregate_saved_results(DIR, output_file='./aggregated_results.csv', exclude_papers=None):
    """
    Create aggregated results CSV from previously saved pickle files.
    """
    if exclude_papers is None:
        exclude_papers = []

    print("=" * 60)
    print("Aggregating Results from Saved Pickle Files")
    print("=" * 60)

    all_results = []
    missing_results = []
    excluded_count = 0
    files = os.listdir(DIR)

    for paper_id in sorted(files):
        if paper_id.endswith('.ini'):
            continue
        if paper_id in exclude_papers:
            excluded_count += 1
            continue
        folder_path = os.path.join(DIR, paper_id)
        if not os.path.isdir(folder_path):
            continue
        
        subfolders = [f for f in os.listdir(folder_path) if not f.endswith('.ini') and os.path.isdir(os.path.join(folder_path, f))]

        for panel_id in subfolders:
            pickle_file = os.path.join(folder_path, panel_id, 'replication_results.pkl')
            if os.path.exists(pickle_file):
                try:
                    result = pd.read_pickle(pickle_file)
                    if isinstance(result, pd.Series):
                        result = result.to_dict()
                    result['paper'] = paper_id
                    result['panel'] = panel_id
                    all_results.append(result)
                    print(f"✓ Loaded: {paper_id}/{panel_id}")
                except Exception as e:
                    print(f"✗ Error loading {paper_id}/{panel_id}: {e}")
                    missing_results.append((paper_id, panel_id, f"Load error: {e}"))
            else:
                missing_results.append((paper_id, panel_id, "File not found"))

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values(['paper', 'panel']).reset_index(drop=True)
        results_df.to_csv(output_file, index=False)
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Successfully loaded: {len(all_results)} results")
        print(f"  Missing/failed: {len(missing_results)} results")
        print(f"  Excluded papers: {excluded_count}")
        print(f"  Output saved to: {output_file}")
        print("=" * 60)
        return results_df
    else:
        print("\n✗ No results found to aggregate!")
        return None

if __name__ == "__main__":
    run_replications(DIR, n_jobs=38, verbose=10, start_from=None, exclude_papers=['50','58','141','144'])

    df = aggregate_saved_results(
        DIR=DIR,
        output_file='./all_replication_results.csv',
        exclude_papers=['141']
    )

    if df is not None:
        print("\nQuick Statistics:")
        print(f"  Mean beta: {df['beta'].mean():.4f}")
        if 'ppml_beta' in df.columns:
            print(f"  Mean PPML beta: {df['ppml_beta'].mean():.4f}")
        print(f"  Mean correction: {df['correction'].mean():.4f}")
