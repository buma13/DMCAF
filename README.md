# TUM MLMI - DMCAF

![method](assets/architecture.png)

## Setup
To setup your environment to run DMCAF run:
```
source setup.sh
```
As environment variables are used to define output and model cashing paths, its required to source `setup.sh` in every
If your want to use different paths, we recommend to create a `custom_setup.sh` file and source this.

## Usage
Create condition sets first:
```
python run_condition_gen.py config/condition_sets/condition_set_000.yaml
```

Run an experiment that references one or more condition sets:
```
python run_experiment.py config/experiments/experiment_000.yaml
```

## Time Planing
```mermaid
gantt
    title Time-plan
    dateFormat  YYYY-MM-DD
    tickInterval 1week
    weekday monday

    section Concept Phase
        Lit. Rev. Diffusion Models (DMs) : done, c1, 2025-05-12, 3w
        Run DMs and Vis. Tools : done, c2, 2025-05-19, 17d
        Lit. Rev. Eval Frameworks: done, c3, 2025-05-26, 3w
        Prep. Slides: done, R1, 2025-06-05, 11d

    section Implementation
        MVP Implementation: i1, after m1, 2w
        Specific Metrics: i2, 2025-06-23, 2w
        Vis. Tools and Interact. GUI: i4, 2025-06-23, 3w
        Advanced Conditioning: i3,  2025-06-30, 2w
        Framework Refinement: i5, after i3, 4w

    section Experimentation
        Design & Run Exp. I: ex1, 2025-06-30 ,2w
        Design & Run Exp. II: ex2, 2025-07-07 ,3w
        Comprehensive Evaluation: wp6, 2025-07-14, 6w
        Run Final Experiments: ex3, after m2 ,2w

    section Reporting
        Prep. Final Slides: r1, 2025-07-21, 2w
        Finalize Report: r2, 2025-08-18, 2w

    section Milestones
        Mid-Presentation: milestone, m1, 2025-06-16, 0d
        Final Presentation: milestone, m2, 2025-08-04, 0d
        Final Submission: milestone, m3, 2025-09-08, 0d
```

## Tasks
- [ ] Make DMRUnner support other DMs (especially controlnet)
- [ ] Evaluation, add object count metric, other of the shelf metrics
- [ ] Segmentation mask condition generator for controlnet
- [ ] Visualization tool
- [ ] Evaluation: implement sores IS, CLIP
- [ ] More Parameters for DM runner (We have: guidance scale, inference steps, add: Scheduler, Seed, Number of images, Image Dimensions, Negative Prompt (ad to condition generator vs. fixed set e.g. blurry, unrealistic))

FILE STRUCTURE:
    - DMCAF/
        - assets/
        - config/
        - dmcaf/
        - medical_image/
    - models/
    - data/ (OUTPUT_DIRECTORY)