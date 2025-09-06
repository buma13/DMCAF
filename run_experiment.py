import yaml
import os
import argparse
from dmcaf.dm_runner import DMRunner, DMRunner_Fundus
from dmcaf.evaluator import Evaluator, Evaluator_Fundus


def run_experiment(config_path, skip_dm_runner=False, skip_eval=False, custom_model=False):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check if data_dir exists
    data_dir = os.getenv('OUTPUT_DIRECTORY')
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"[ERROR] data_dir does not exist: {data_dir}, Is the env variable \"OUTPUT_DIRECTORY\" set correctly?")

    exp_id = config['experiment_id']
    conditioning_db_path = os.path.join(data_dir, config['conditioning_db'])
    output_db_path = os.path.join(data_dir, config['output_db'])
    metrics_db_path = os.path.join(data_dir, config['metrics_db'])

    condition_sets = config.get('conditioning', {}).get('condition_sets', [])

    # Stable diffusion models
    if not custom_model:
        # Step 1: Run DM on conditions
        if not skip_dm_runner:
            print("[Step 1] Running diffusion model...")
            dm_runner = DMRunner(
                conditioning_db_path=conditioning_db_path,
                output_db_path=output_db_path,
                output_dir=os.path.join(data_dir, exp_id)
            )
            model_configs = config['dm_runner']['models']
            visualizer_configs = config.get('visualizer', {})
            dm_runner.run_experiment(exp_id, model_configs, visualizer_configs, condition_sets=condition_sets)
        # Step 2: Evaluate outputs
        if not skip_eval:
            print("[Step 2] Evaluating outputs...")
            evaluator = Evaluator(
                conditioning_db_path=conditioning_db_path,
                output_db_path=output_db_path,
                metrics_db_path=metrics_db_path
            )
            metrics = config['evaluator']['metrics']
            evaluator.evaluate_outputs(exp_id, metrics)

    # Custom models
    else:
        print("[Step 1] Running diffusion model w/ custom model...")

        # Check if monai_home exists
        monai_home = os.getenv('MONAI_HOME')
        if not os.path.isdir(monai_home):
            raise FileNotFoundError(
                f"[ERROR] monai_home does not exist: {monai_home}, Is the env variable \"MONAI_HOME\" set correctly?")

        if not skip_dm_runner:
            dm_runner = DMRunner_Fundus(
                monai_home=monai_home,
                output_dir=os.path.join(data_dir, exp_id)
            )
            model_configs = config['dm_runner']['models']
            dm_runner.run_experiment(exp_id, model_configs, condition_sets=condition_sets)

        if not skip_eval:
            print("[Step 2] Evaluating outputs...")
            evaluator = Evaluator_Fundus(
                monai_home=monai_home,
                output_dir=os.path.join(data_dir, exp_id)
            )
            metrics = config['evaluator']['metrics']
            evaluator.evaluate_outputs(exp_id, metrics)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DMCAF experiment pipeline")
    parser.add_argument("config_path", type=str, help="Path to the experiment YAML config file")

    parser.add_argument("--skip_dm_runner", action="store_true", help="Skip diffusion model execution")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation step")
    parser.add_argument("--custom_model", action="store_true", help="Use a custom model for medical images")

    args = parser.parse_args()
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


    run_experiment(
        config_path=args.config_path,
        skip_dm_runner=args.skip_dm_runner,
        skip_eval=args.skip_eval,
        custom_model=args.custom_model
    )
