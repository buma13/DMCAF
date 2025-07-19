import yaml
import os
import argparse
from dmcaf.condition_generator import ConditionGenerator
from dmcaf.dm_runner import DMRunner
from dmcaf.evaluator import Evaluator


def run_experiment(config_path, skip_condition_gen=False, skip_dm_runner=False, skip_eval=False):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check if data_dir exists
    data_dir = config['data_dir']
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"[ERROR] data_dir does not exist: {data_dir}")

    exp_id = config['experiment_id']
    conditioning_db_path = os.path.join(config['data_dir'], config['conditioning_db'])
    output_db_path = os.path.join(config['data_dir'], config['output_db'])
    metrics_db_path = os.path.join(config['data_dir'], config['metrics_db'])

    # Step 1: Generate conditions
    if not skip_condition_gen:
        print("[Step 1] Generating conditions...")
        cond_gen = ConditionGenerator(conditioning_db_path=conditioning_db_path)
        cond_gen.generate_experiment(
            experiment_id=exp_id,
            n_text=config['condition_generator']['text_prompts'],
            n_seg=config['condition_generator']['segmentation_maps']
        )

    # Step 2: Run DM on conditions
    if not skip_dm_runner:
        print("[Step 2] Running diffusion model...")
        dm_runner = DMRunner(
            conditioning_db_path=conditioning_db_path,
            output_db_path=output_db_path,
            output_dir=os.path.join(config['data_dir'], exp_id)
        )
        model_configs = config['dm_runner']['models']
        dm_runner.run_experiment(exp_id, model_configs)

    # Step 3: Evaluate outputs
    if not skip_eval:
        print("[Step 3] Evaluating outputs...")
        evaluator = Evaluator(
            conditioning_db_path=conditioning_db_path,
            output_db_path=output_db_path,
            metrics_db_path=metrics_db_path
        )
        metrics = config['evaluator']['metrics']
        evaluator.evaluate_outputs(exp_id, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DMCAF experiment pipeline")
    parser.add_argument("config_path", type=str, help="Path to the experiment YAML config file")

    parser.add_argument("--skip_condition_gen", action="store_true", help="Skip condition generation")
    parser.add_argument("--skip_dm_runner", action="store_true", help="Skip diffusion model execution")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation step")

    args = parser.parse_args()
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


    run_experiment(
        config_path=args.config_path,
        skip_condition_gen=args.skip_condition_gen,
        skip_dm_runner=args.skip_dm_runner,
        skip_eval=args.skip_eval
    )
