import yaml
import os
import argparse
from dmcaf.condition_generator import ConditionGenerator


def run_condition_generation(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_dir = os.getenv('OUTPUT_DIRECTORY')
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"[ERROR] data_dir does not exist: {data_dir}, Is the env variable \"OUTPUT_DIRECTORY\" set correctly?")

    conditioning_db_path = os.path.join(data_dir, config['conditioning_db'])

    cond_gen = ConditionGenerator(conditioning_db_path=conditioning_db_path)
    cg_cfg = config.get('condition_generator', {})
    cond_gen.generate_experiment(
        experiment_id=config['condition_set_id'],
        n_text=cg_cfg.get('text_prompts', 0),
        n_compositional=cg_cfg.get('compositional_prompts', 0),
        n_seg=cg_cfg.get('segmentation_maps', 0),
        n_color=cg_cfg.get('color_prompts', 0)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate condition sets")
    parser.add_argument("config_path", type=str, help="Path to the condition set YAML config file")
    args = parser.parse_args()
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    run_condition_generation(args.config_path)
