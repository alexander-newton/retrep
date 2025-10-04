import os
import yaml
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import scipy.stats as st # Added for confidence intervals

print("=" * 60)
print("Multi-GPU Setup with Loky Backend (True Process Isolation)")
print("=" * 60)

# Load configuration
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)
DIR = config['outputdata']

def run_ppml_estimation(X, y, z, metadata):
    """
    Runs the PPML estimation for a single dataset to calculate semi-elasticity.
    This function is called within the isolated worker process.
    """
    # Import ppml module inside the worker to ensure it uses the correct environment
    from ppml import fit_ppml_or_ivppml

    interest_col = metadata.get('interest', [0])
    fe_cols = metadata.get('fe', None)
    fe_cols = [int(f) for f in fe_cols] if fe_cols is not None else None
    endog_cols = metadata.get('endogenous_regressors', None)

    try:
        instruments = z if z is not None else None

        # Call the provided PPML fitting function
        res_ppml, meta_ppml, interest_ppml = fit_ppml_or_ivppml(
            y=y,
            X=X,
            instruments=instruments,
            fe_cols=fe_cols,
            endog_cols=endog_cols,
            interest_var=interest_col,
            hdfe=True, # Use HDFE absorption as it's generally more robust
            return_meta=True
        )

        # Process results if the estimation was successful for the variable of interest
        if interest_ppml and 'beta' in interest_ppml and 'se' in interest_ppml and pd.notna(interest_ppml['beta']):
            beta = interest_ppml['beta']
            se = interest_ppml['se']

            # Calculate 95% Wald confidence interval
            z_score = st.norm.ppf(0.975)  # Approx 1.96
            conf_05 = beta - z_score * se
            conf_95 = beta + z_score * se

            return {
                'ppml_beta': beta,
                'ppml_se': se,
                'ppml_conf_05': conf_05,
                'ppml_conf_95': conf_95
            }
        else:
            print("PPML estimation did not return valid results for the interest variable.")
            return {'ppml_beta': np.nan, 'ppml_se': np.nan, 'ppml_conf_05': np.nan, 'ppml_conf_95': np.nan}

    except Exception as e:
        print(f"An error occurred during PPML estimation: {e}")
        import traceback
        traceback.print_exc()
        return {'ppml_beta': np.nan, 'ppml_se': np.nan, 'ppml_conf_05': np.nan, 'ppml_conf_95': np.nan}


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

        # Define the fields for both estimation types
        loglinear_fields = [
            'beta', 'beta_se', 'average_estimate', 'ae_bootstrap_se',
            'conf_int_05', 'conf_int_95', 'estimate_at_average',
            'ea_bootstrap_se', 'ea_conf_05', 'ea_conf_95',
            'correction', 'correction_se', 'correction_05', 'correction_95'
        ]
        ppml_fields = ['ppml_beta', 'ppml_se', 'ppml_conf_05', 'ppml_conf_95']

        new_path = os.path.join(result_path, 'replication_results.pkl')
        run_loglinear = True
        run_ppml = True
        existing_results = {}

        # Conditionally decide which estimations to run
        if os.path.exists(new_path):
            try:
                existing_results = pd.read_pickle(new_path)
                if isinstance(existing_results, pd.Series):
                    existing_results = existing_results.to_dict()

                loglinear_present = all(field in existing_results for field in loglinear_fields)
                ppml_present = all(field in existing_results for field in ppml_fields)

                if loglinear_present and ppml_present:
                    print(f"[GPU {gpu_id}] Skipping {paper_id}/{panel_id} - complete results already exist")
                    return None

                if loglinear_present:
                    # Original results exist, only run the new PPML part
                    run_loglinear = False
                    print(f"[GPU {gpu_id}] Found existing log-linear results for {paper_id}/{panel_id}. Will only run PPML.")
                else:
                    # If loglinear results are missing, re-run everything for consistency
                    print(f"[GPU {gpu_id}] Missing log-linear results for {paper_id}/{panel_id}. Re-running all estimations.")

            except Exception as e:
                print(f"[GPU {gpu_id}] Error reading existing results for {paper_id}/{panel_id}: {e}. Re-estimating from scratch.")
                if os.path.exists(new_path):
                    os.remove(new_path)
                existing_results = {}

        # Load data
        metadata_path = os.path.join(result_path, 'metadata.json')
        X = pd.read_parquet(os.path.join(result_path, 'X.parquet'))
        y = pd.read_parquet(os.path.join(result_path, 'y.parquet'))

        if os.path.exists(os.path.join(result_path, 'z.parquet')):
            z = pd.read_parquet(os.path.join(result_path, 'z.parquet'))
        else:
            z = None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"[GPU {gpu_id}] Metadata for {paper_id}/{panel_id}:", metadata)

        # Run estimations based on the flags set above
        final_results = existing_results.copy()

        if run_loglinear:
            print(f"[GPU {gpu_id}] Running log-linear correction for {paper_id}/{panel_id}")
            loglinear_res = replicate_full(X, y, z, metadata, gpu_id)
            final_results.update(loglinear_res)

        if run_ppml:
            print(f"[GPU {gpu_id}] Running PPML estimation for {paper_id}/{panel_id}")
            ppml_res = run_ppml_estimation(X, y, z, metadata)
            final_results.update(ppml_res)

        # Save the combined results
        pd.Series(final_results).to_pickle(new_path)

        # Add identifiers for aggregation
        final_results['paper'] = paper_id
        final_results['panel'] = panel_id
        final_results['gpu_used'] = gpu_id
        final_results['process_id'] = os.getpid()

        return final_results

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

    Parameters:
    -----------
    DIR : str
        Directory containing the data
    n_jobs : int, default=30
        Number of parallel jobs. Jobs will be distributed across available GPUs.
    verbose : int, default=10
        Verbosity level for joblib
    start_from : str, optional
        Paper ID to start from (e.g., '018'). Will skip all papers before this.
    """
    import tensorflow as tf

    # Get GPU count from main process
    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(gpus)

    print(f"\nConfiguration:")
    print(f"  Available GPUs: {num_gpus}")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"  Jobs per GPU: ~{n_jobs / num_gpus:.1f}")
    print(f"  TensorFlow threads per job: 2 intra + 2 inter = 4")
    print(f"  Total TensorFlow threads: {n_jobs} × 4 = {n_jobs * 4}")
    print(f"  Backend: loky (true process isolation)")
    print()

    # Collect all paths to process
    result_paths = collect_result_paths(DIR)


    # Filter out excluded papers
    if exclude_papers is None:
        exclude_papers = []

    if exclude_papers:
        result_paths = [(path, paper_id) for path, paper_id in result_paths
                        if paper_id not in exclude_papers]
        print(f"Excluded papers: {', '.join(exclude_papers)}")

    # Filter paths if start_from is specified
    if start_from is not None:
        result_paths = [(path, paper_id) for path, paper_id in result_paths
                        if paper_id >= start_from]
        print(f"Starting from paper {start_from}")

    print(f"Found {len(result_paths)} result paths to process")

    # Assign each job to a GPU in round-robin fashion
    jobs_with_gpu = [(path, paper_id, i % num_gpus)
                     for i, (path, paper_id) in enumerate(result_paths)]

    print(f"\nGPU assignment summary:")
    gpu_counts = {}
    for _, _, gpu_id in jobs_with_gpu:
        gpu_counts[gpu_id] = gpu_counts.get(gpu_id, 0) + 1
    for gpu_id, count in sorted(gpu_counts.items()):
        print(f"  GPU {gpu_id}: {count} jobs queued")
    print()

    # Process in parallel with loky backend
    results = Parallel(n_jobs=n_jobs, verbose=verbose, backend='loky')(
        delayed(process_single_result)(path, paper_id, gpu_id)
        for path, paper_id, gpu_id in jobs_with_gpu
    )

    # Filter out None results (failures or skipped)
    all_results = [r for r in results if r is not None]

    # Save aggregated results
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv('./trial_results_se.csv', index=False)
        print(f"\nSuccessfully processed {len(all_results)} replications")

        # Show GPU usage stats
        if 'gpu_used' in all_results_df.columns:
            print("\nActual GPU usage distribution:")
            print(all_results_df['gpu_used'].value_counts().sort_index())

        # Show process distribution
        if 'process_id' in all_results_df.columns:
            print(f"\nUnique processes used: {all_results_df['process_id'].nunique()}")

        print(f"Results saved to ./trial_results_se.csv")
    else:
        print("\nNo new results to save")

    return all_results

def replicate_full(X, y, z, metadata, gpu_id):
    """
    Replicate the full analysis for a single dataset.
    Each call runs in its own process with dedicated GPU and thread pool.
    """
    import tensorflow as tf
    import numpy as np
    from loglinearcorrection.correction_estimator import DoublyRobustElasticityEstimator

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
    }

    print(res_dict)
    return res_dict

def aggregate_saved_results(DIR, output_file='./aggregated_results.csv', exclude_papers=None):
    """
    Create aggregated results CSV from previously saved pickle files.

    Parameters:
    -----------
    DIR : str
        Directory containing the data (same as used in run_replications)
    output_file : str, default='./aggregated_results.csv'
        Path where aggregated results will be saved
    exclude_papers : list, optional
        List of paper IDs to exclude (e.g., ['141'])

    Returns:
    --------
    pd.DataFrame : Aggregated results
    """
    if exclude_papers is None:
        exclude_papers = []

    print("=" * 60)
    print("Aggregating Results from Saved Pickle Files")
    print("=" * 60)

    all_results = []
    missing_results = []
    excluded_count = 0

    # Walk through all paper folders
    files = os.listdir(DIR)

    for paper_id in sorted(files):
        if paper_id.endswith('.ini'):
            continue

        # Skip excluded papers
        if paper_id in exclude_papers:
            excluded_count += 1
            continue

        folder_path = os.path.join(DIR, paper_id)
        if not os.path.isdir(folder_path):
            continue

        # Find all result subfolders
        subfolders = [
            f for f in os.listdir(folder_path)
            if not f.endswith('.ini') and os.path.isdir(os.path.join(folder_path, f))
        ]

        for panel_id in subfolders:
            result_path = os.path.join(folder_path, panel_id)
            pickle_file = os.path.join(result_path, 'replication_results.pkl')

            if os.path.exists(pickle_file):
                try:
                    # Load the pickle file
                    result = pd.read_pickle(pickle_file)

                    # Convert Series to dict if needed
                    if isinstance(result, pd.Series):
                        result = result.to_dict()

                    # Ensure paper and panel IDs are present
                    if 'paper' not in result:
                        result['paper'] = paper_id
                    if 'panel' not in result:
                        result['panel'] = panel_id

                    all_results.append(result)
                    print(f"✓ Loaded: {paper_id}/{panel_id}")

                except Exception as e:
                    print(f"✗ Error loading {paper_id}/{panel_id}: {e}")
                    missing_results.append((paper_id, panel_id, f"Load error: {e}"))
            else:
                missing_results.append((paper_id, panel_id, "File not found"))

    # Create DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)

        # Sort by paper and panel
        results_df = results_df.sort_values(['paper', 'panel']).reset_index(drop=True)

        # Save to CSV
        results_df.to_csv(output_file, index=False)

        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Successfully loaded: {len(all_results)} results")
        print(f"  Missing/failed: {len(missing_results)} results")
        print(f"  Excluded papers: {excluded_count}")
        print(f"  Output saved to: {output_file}")
        print("=" * 60)

        # Show column info
        print("\nColumns in aggregated data:")
        for col in results_df.columns:
            print(f"  - {col}")

        # Show missing results if any
        if missing_results:
            print("\nMissing/Failed Results:")
            for paper_id, panel_id, reason in missing_results:
                print(f"  {paper_id}/{panel_id}: {reason}")

        return results_df
    else:
        print("\n✗ No results found to aggregate!")
        return None

if __name__ == "__main__":
    # Use 30 parallel jobs distributed across 8 GPUs
    # Each GPU handles ~3-4 jobs simultaneously
    #run_replications(DIR, n_jobs=38, verbose=10, start_from=None, exclude_papers=['50','58','141','144'])


    # Aggregate all results, excluding paper 141
    df = aggregate_saved_results(
        DIR=DIR,
        output_file='./all_replication_results.csv',
        exclude_papers=['141']
    )

    # Optional: Show some statistics
    if df is not None:
        print("\nQuick Statistics:")
        print(f"  Mean beta: {df['beta'].mean():.4f}")
        print(f"  Mean correction: {df['correction'].mean():.4f}")
        if 'ppml_beta' in df.columns:
            print(f"  Mean PPML beta (semi-elasticity): {df['ppml_beta'].mean():.4f}")

        if 'gpu_used' in df.columns:
            print("\nGPU distribution:")
            print(df['gpu_used'].value_counts().sort_index())
