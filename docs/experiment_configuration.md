# Enhanced DMCAF Experiment Configuration

This document describes the enhanced experiment configuration system that supports fine-grained control over condition selection and filtering.

## Overview

The new system supports two workflows:
1. **Condition Generation**: Create comprehensive condition sets once (thousands of prompts)
2. **Specialized Experiments**: Target specific subsets for focused analysis

## Configuration Format

### Basic Structure (Backwards Compatible)

```yaml
experiment_id: experiment_name
conditioning_db: conditioning.db
output_db: dm_output.db
metrics_db: metrics.db

conditioning:
  condition_sets:
    - condition_set_id: condition_set_000
      limit_number_of_conditions: 100
      types:
        - count_prompt  # Matches all count_prompt variants
        - color_prompt
        - compositional_prompt

dm_runner:
  models:
    - model_name: stable-diffusion-v1-5/stable-diffusion-v1-5
      guidance_scale: 7.5
      num_inference_steps: 50

evaluator:
  metrics:
    - YOLO Count
    - Segmentation Dice
```

### Advanced Configuration

```yaml
experiment_id: experiment_advanced
conditioning_db: conditioning.db
output_db: dm_output.db
metrics_db: metrics.db

conditioning:
  condition_sets:
    - condition_set_id: condition_set_000
      limit_number_of_conditions: 500
      
      # Simple type selection (backwards compatible)
      types:
        - color_prompt
        - compositional_prompt
        
      # Fine-grained variant control
      type_variants:
        count_prompt:
          variants: ["base", "numeral", "background"]  # Specific variants
          # variants: "all"  # Or include all available variants
          
      # Advanced filtering options
      filters:
        # Object-based filtering
        objects: ["cat", "dog", "car"]
        
        # Number-based filtering  
        number_range: [1, 2, 3, 4]
        
        # Background-based filtering
        backgrounds: ["unicolor background", "white background"]
        
        # Custom SQL conditions (advanced)
        custom_where: "number <= 3 AND object IN ('cat', 'dog')"

dm_runner:
  models:
    - model_name: stable-diffusion-v1-5/stable-diffusion-v1-5
      guidance_scale: 7.5
      num_inference_steps: 50

evaluator:
  metrics:
    - YOLO Count
    - Color Classification
    - YOLO Composition
    - Segmentation Dice
```

## Configuration Options

### Type Selection

#### 1. Simple Types (Backwards Compatible)
```yaml
types:
  - count_prompt    # Matches count_prompt, count_prompt_base, count_prompt_numeral, etc.
  - color_prompt    # Matches all color prompt variants
```

#### 2. Fine-grained Variant Control
```yaml
type_variants:
  count_prompt:
    variants: ["base", "numeral", "background"]  # Only these variants
  color_prompt:
    variants: "all"  # All available variants
```

### Filtering Options

#### 1. Object Filtering
```yaml
filters:
  objects: ["cat", "dog", "car", "bird"]
```

#### 2. Number Range Filtering
```yaml
filters:
  number_range: [1, 2, 3]  # Only conditions with these counts
```

#### 3. Background Filtering
```yaml
filters:
  backgrounds: ["unicolor background", "white background"]
```

#### 4. Custom SQL Filtering
```yaml
filters:
  custom_where: "number <= 3 AND confidence > 0.8"
```

## Example Specialized Experiments

### 1. Count Prompt Variants Analysis
```yaml
experiment_id: experiment_count_variants
# ... basic config ...
conditioning:
  condition_sets:
    - condition_set_id: condition_set_000
      limit_number_of_conditions: 300
      type_variants:
        count_prompt:
          variants: ["base", "numeral", "background"]
      filters:
        number_range: [1, 2, 3]
        objects: ["cat", "dog", "car", "bird"]
```

### 2. Color Analysis Only
```yaml
experiment_id: experiment_color_only
# ... basic config ...
conditioning:
  condition_sets:
    - condition_set_id: condition_set_000
      limit_number_of_conditions: 200
      types:
        - color_prompt
      filters:
        objects: ["cat", "car", "ball"]
```

### 3. Cross-Modal Analysis
```yaml
experiment_id: experiment_cross_modal
# ... basic config ...
conditioning:
  condition_sets:
    - condition_set_id: condition_set_000
      limit_number_of_conditions: 500
      types:
        - color_prompt
        - compositional_prompt
      type_variants:
        count_prompt:
          variants: ["base"]
      filters:
        objects: ["cat", "dog"]
        custom_where: "number <= 3"
```

### 4. Fundus Analysis Only
Run with config/experiments/experiment_fundus.yaml --custom_model
```yaml
experiment_id: experiment_fundus
conditioning:
  condition_sets:
    - condition_set_id: fundus
      limit_number_of_conditions: 512
      types:
        - segmentation_mask

dm_runner:
  models:
    - model_name: unet
      num_inference_steps: 1000
      seed: 42

evaluator:
  metrics:
    - DICE

```
Use the validation tool to check your configuration:

```bash
python validate_experiment_config.py config/experiments/experiment_object_count.yaml
```

### Sample Validation Output
```
============================================================
DMCAF EXPERIMENT CONFIG VALIDATION REPORT
============================================================
✅ Configuration is valid!

📊 STATISTICS:

  Condition Set: condition_set_000
    Total available conditions: 1000
    Would select: 150
    Available types: ['count_prompt', 'color_prompt', 'compositional_prompt']
    Type families:
      count_prompt: ['base', 'numeral', 'background']
      color_prompt: ['base']
      compositional_prompt: ['base']
```

## Troubleshooting

### Common Issues

1. **Type not found**: Use validation tool to see available types
2. **No conditions selected**: Check filters and type specifications
3. **Unexpected results**: Validate config and check database contents

### Best Practices

1. Always validate configs before running experiments
2. Use descriptive experiment IDs
3. Start with smaller limits for testing
4. Check available types in your condition sets
5. Use filters to focus on specific research questions
