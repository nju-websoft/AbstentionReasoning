# ReHallucination Inference-Time Intervention

This repository contains the inference-time intervention code and released data for the paper **"Answering the Unanswerable"**. The code focuses on the inference pipeline: vanilla trajectory generation, probe-based intervention, and evaluation.

## Overview

The method is organized into three stages:

1. **Stage 1: Vanilla generation**
   Generate reasoning trajectories without intervention for answerable and unanswerable problems.

2. **Stage 2: Probe detection and intervention**
   Replay the vanilla trajectory, use a linear probe at `wait` positions to detect the first intervention point, insert the intervention prompt once, and continue generation.

3. **Stage 3: Evaluation**
   Evaluate unanswerable examples with Abstention and evaluate answerable examples with Answer Accuracy.

Precomputed Stage 1 and Stage 2 outputs are included under `data/out/`, so you can directly evaluate released results without rerunning generation.

## Repository Layout

```text
codes/
  main_vllm.py                  # Stage 1 vanilla generation
  a_inter.py                    # Stage 2 probe detection + intervention
  interven.py                   # Linear probe utilities
  stage_1_generate_vanilla.sh   # Stage 1 example script
  stage_2_run_intervention.sh   # Stage 2 example script
  stage_3_evaluate.py           # Stage 3 evaluator
  stage_3_evaluate.sh           # Stage 3 example script
  a_input_path.py               # Paths to released vanilla trajectories

data/
  sum/                          # SUM input data
  umwp/                         # UMWP input data
  out/stage1/                   # Released vanilla trajectories
  out/stage2/                   # Released intervention outputs

model/
  *_layer_result/layer_*.pt     # Released linear probe weights
```

## Setup

Install the Python dependencies needed by the stages you want to run.

```bash
pip install -r requirements.txt
```

The default Stage 3 evaluator uses a lightweight exact/numeric matcher and does not require a local judge model.

## Stage 1: Generate Vanilla Trajectories

Stage 1 generates answerable and unanswerable vanilla reasoning trajectories.

```bash
cd codes
bash stage_1_generate_vanilla.sh
```

Useful environment variables:

```bash
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
DATASET=SUM            # SUM or UMWP
RUN_SPLIT=both         # both, answer, or unanswer
```

Outputs are written to:

```text
data/out/stage1/{DATASET}/{MODEL_NAME}/{solve,unsolve}/
```

Released Stage 1 outputs are already included and referenced by `codes/a_input_path.py`.

## Stage 2: Run Intervention

Stage 2 first detects the intervention point with a linear probe, then performs one intervention by inserting the fixed intervention prompt and continuing generation.

Example:

```bash
cd codes
bash stage_2_run_intervention.sh
```

Default example configuration:

```bash
DATASET=SUM
MODEL_NAME=DeepSeek-R1-Distill-Llama-8B
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
PROBE_LAYER=22
THEROD=0.4
```

Outputs are written to:

```text
data/out/stage2/{DATASET}/{MODEL_NAME}/
  unsolve_inter_q_id_point.json
  solve_inter_q_id_point.json
  unsolve_intervention_result.jsonl
  solve_intervention_result.jsonl
```

Released Stage 2 intervention outputs are included for SUM and UMWP across the released models.

## Probe Layers

The released probe weights cover the layers used in our experiments:

| Dataset | Model | Probe Layer |
|---|---:|---:|
| SUM | DeepSeek-R1-Distill-Llama-8B | 22 |
| SUM | DeepSeek-R1-Distill-Qwen-7B | 17 |
| SUM | DeepSeek-R1-Distill-Qwen-14B | 30 |
| SUM | Qwen3-8B | 24 |
| SUM | Qwen3-14B | 26 |
| UMWP | DeepSeek-R1-Distill-Llama-8B | 17 |
| UMWP | DeepSeek-R1-Distill-Qwen-7B | 18 |
| UMWP | DeepSeek-R1-Distill-Qwen-14B | 30 |
| UMWP | Qwen3-8B | 20 |
| UMWP | Qwen3-14B | 24 |

## Stage 3: Evaluate Results

To evaluate released Stage 2 outputs:

```bash
cd codes
bash stage_3_evaluate.sh
```

By default, this evaluates:

```text
data/out/stage2/SUM/DeepSeek-R1-Distill-Llama-8B/
```

Change the target with environment variables:

```bash
DATASET=UMWP MODEL_NAME=Qwen3-8B bash stage_3_evaluate.sh
```

The evaluator reports:

- **Abstention** on unanswerable examples.
- **Answer Acc** on answerable examples.
- A JSON metrics file at `data/out/stage2/{DATASET}/{MODEL_NAME}/stage3_metrics.json`.

By default, answer equivalence uses a lightweight exact/numeric matcher:

```bash
ANSWER_JUDGE=exact bash stage_3_evaluate.sh
```

If you want to use an LLM judge for answer equivalence, configure your local judge service and pass the corresponding options to `stage_3_evaluate.py`.

## Notes
The Stage 2 implementation uses vanilla trajectories from Stage 1, which approximates online intervention while making experiments reproducible.
