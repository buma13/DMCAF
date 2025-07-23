import gradio as gr
import yaml
import os
import sqlite3
from run_experiment import run_experiment

config_path = "./config/experiments/experiment_000.yaml"

def load_yaml(path=config_path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(experiment_id, data_dir, text_prompts, segmentation_maps, models_yaml, metrics_yaml):
    config = {
        "experiment_id": experiment_id,
        "data_dir": data_dir,
        "conditioning_db": "conditioning.db",
        "output_db": "outputs.db",
        "metrics_db": "metrics.db",
        "condition_generator": {
            "text_prompts": int(text_prompts),
            "segmentation_maps": int(segmentation_maps)
        },
        "dm_runner": {
            "models": yaml.safe_load(models_yaml)
        },
        "evaluator": {
            "metrics": yaml.safe_load(metrics_yaml)
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return "✅ YAML updated!"

def run_pipeline(skip_cond, skip_dm, skip_eval):
    try:
        run_experiment(config_path, skip_cond, skip_dm, skip_eval)
        return "✅ Run completed!"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def show_results():
    try:
        cfg = load_yaml()
        exp_id = cfg["experiment_id"]
        db_path = os.path.join(cfg["data_dir"], cfg['output_db'])
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT prompt, image_path FROM dm_outputs WHERE experiment_id=?", (exp_id,))
        rows = cursor.fetchall()
        prompts, images = zip(*rows) if rows else ([], [])
        return list(images)
    except Exception as e:
        return [f"❌ Failed to load results: {str(e)}"]

# ---- Load values from config ----
cfg = load_yaml()
experiment_id_val = cfg.get("experiment_id", "")
data_dir_val = os.getenv('OUTPUT_DIRECTORY')
text_prompts_val = cfg.get("condition_generator", {}).get("text_prompts", 0)
seg_maps_val = cfg.get("condition_generator", {}).get("segmentation_maps", 0)
models_val = yaml.dump(cfg.get("dm_runner", {}).get("models", []))
metrics_val = yaml.dump(cfg.get("evaluator", {}).get("metrics", []))

# ---- Build UI ----
with gr.Blocks() as demo:
    gr.Markdown("## 🎛️ DMCAF Experiment GUI")

    with gr.Row():
        experiment_id = gr.Textbox(label="Experiment ID", value=experiment_id_val)
        data_dir = gr.Textbox(label="Data Directory", value=data_dir_val)

    with gr.Row():
        text_prompts = gr.Number(label="# Text Prompts", value=text_prompts_val)
        segmentation_maps = gr.Number(label="# Segmentation Maps", value=seg_maps_val)

    models_yaml = gr.Code(label="Models (YAML)", language="yaml", value=models_val)
    metrics_yaml = gr.Code(label="Evaluation Metrics (YAML)", language="yaml", value=metrics_val)

    save_btn = gr.Button("💾 Save YAML Config")
    status = gr.Textbox(label="YAML Save Status")

    save_btn.click(
        save_yaml,
        [experiment_id, data_dir, text_prompts, segmentation_maps, models_yaml, metrics_yaml],
        status
    )

    gr.Markdown("---")

    skip_cond = gr.Checkbox(label="Skip Condition Generation", value=False)
    skip_dm = gr.Checkbox(label="Skip DM Runner", value=False)
    skip_eval = gr.Checkbox(label="Skip Evaluation", value=False)
    run_btn = gr.Button("🚀 Run Experiment")
    run_status = gr.Textbox(label="Run Status")

    run_btn.click(run_pipeline, [skip_cond, skip_dm, skip_eval], run_status)

    gr.Markdown("---")
    show_btn = gr.Button("📸 Show Results")
    gallery = gr.Gallery(label="Generated Images")

    show_btn.click(show_results, [], gallery)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

demo.launch(share=True)
