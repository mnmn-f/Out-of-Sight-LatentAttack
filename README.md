# Out of Sight, Not Out of Mind: Unveiling Latent Attack in Latent-based Multi-Agent Systems

This repository contains the code for studying latent-space attacks in latent-based multi-agent systems. It implements the paper's main pipeline for constructing paired clean and attacked executions, extracting attack-associated latent directions, and injecting these directions back into clean runs.

The key setting is direction-only latent intervention. Direct text-space attacks are used to create reference trajectories, while the latent intervention stage modifies hidden states or KV-cache handoffs without reinserting the original adversarial text.

## What This Repository Contains

The repository is organized around a single experimental chain.

1. Run clean latent-based MAS executions.
2. Run matched direct-attack reference executions.
3. Save hidden-state and KV-cache traces at selected agents and handoffs.
4. Match clean-correct and attack-wrong examples.
5. Extract attack-associated latent directions with DiffMean, PCA, or RePS.
6. Inject the extracted directions into clean executions through node-level or edge-level latent carriers.

The released code is intentionally scoped to this chain. Legacy experiments, ad hoc analysis scripts, figure-generation assets, and unrelated attack families are excluded from the public release.

## Repository Structure

```text
Out-of-Sight-LatentAttack/
  latent_attack/
    run_experiment.py            # main experiment entrypoint
    config.py                    # command-line arguments and validation
    runtime/
      runner.py                  # execution loop for clean, attacked, trace, and injection runs
      results.py                 # partial JSONL, final JSON, and run summaries
    tasks/
      registry.py                # task names and loader dispatch
      loaders.py                 # dataset loader implementations
    vectors/
      extract.py                 # DiffMean/PCA extraction from saved traces
      reps.py                    # RePS direction training
      pairs.py                   # clean/attacked result matching and pair filters
      directions.py              # tensor pooling and direction estimators
      trace_store.py             # trace discovery and role/edge matching
  methods/
    latent_mas.py                # latent-based MAS execution, tracing, and injection
    cache_ops.py                 # KV-cache slicing and truncation helpers
  attacks.py                     # direct text-space reference attack utilities
  models.py                      # HuggingFace and optional vLLM model wrappers
  prompts.py                     # agent prompt construction
  utils.py                       # answer extraction and utility functions
  requirements.txt               # Python dependencies
```

## Installation

Create a fresh Python environment and install the dependencies.

```bash
conda create -n latent-attack python=3.11
conda activate latent-attack
pip install -r requirements.txt
```

The experiments require a GPU for practical runtimes. The default backend uses HuggingFace Transformers. The command-line interface also exposes optional vLLM arguments through `--use_vllm`, while hidden-state and KV-cache injection are intended for the standard HuggingFace backend.

## Supported Tasks

The main runner exposes the following task names.

```text
gsm8k, aime2024, aime2025, gpqa, arc_easy, arc_challenge,
openbookqa, mbppplus, humanevalplus, medqa
```

Some tasks are loaded through HuggingFace `datasets`. Others may require local files under `data/`. If a local dataset is needed, place it under the expected `data/` path before running the task.

## Main Entry Points

Use module entrypoints from the repository root.

```bash
python -m latent_attack.run_experiment --help
python -m latent_attack.vectors.extract --help
python -m latent_attack.vectors.reps --help
```

## Step 1. Clean Reference Runs

The clean run records the task output and, when requested, exports latent traces for selected agent roles.

```bash
python -m latent_attack.run_experiment \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --prompt sequential \
  --latent_steps 5 \
  --max_samples 100 \
  --output_path outputs/gsm8k_clean.json \
  --trace_export \
  --trace_save_hidden \
  --trace_save_kv
```

The output JSON contains a `summary`, resolved `args`, and per-sample `preds`. A streaming partial file is also written as `<output_path>.jsonl`, which can be reused with `--resume_partial`.

When trace export is enabled, the default trace directory is derived from the output path.

```text
outputs/gsm8k_clean_traces/
  sample_00000/
    planner_none__none__none__none.pt
    critic_none__none__none__none.pt
    refiner_none__none__none__none.pt
```

## Step 2. Direct-Attack Reference Runs

The attacked reference run should use the same model, task, prompt format, latent step count, sample order, and sample count as the clean run.

```bash
python -m latent_attack.run_experiment \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --prompt sequential \
  --latent_steps 5 \
  --max_samples 100 \
  --output_path outputs/gsm8k_attack_planner.json \
  --attack_surface role_prompt \
  --attack_type mi \
  --mi_roles planner \
  --trace_export \
  --trace_save_hidden \
  --trace_save_kv
```

The current command-line attack switch is `--attack_type mi`. In this workflow, the attacked run serves as the direct text-space reference used to induce paired latent trajectories. The latent intervention stage uses only the extracted latent directions.

Useful attack arguments are listed below.

| Argument | Description |
|---|---|
| `--attack_surface role_prompt` | Applies the direct reference attack through an agent role prompt. |
| `--attack_type mi` | Selects the implemented direct reference attack template. |
| `--mi_roles planner` | Comma-separated roles that receive the reference attack. |
| `--mi_reference_answer 0` | Optional reference answer used by the attack template. |

## Step 3. Extract DiffMean or PCA Directions

DiffMean and PCA use saved clean and attacked latent traces. The extraction step aligns traces at the same role or handoff site, filters matched examples, and writes a PyTorch direction payload.

```bash
python -m latent_attack.vectors.extract \
  --clean_trace_dir outputs/gsm8k_clean_traces \
  --attacked_trace_dir outputs/gsm8k_attack_planner_traces \
  --clean_results_json outputs/gsm8k_clean.json \
  --attacked_results_json outputs/gsm8k_attack_planner.json \
  --pair_filter clean_correct_attack_wrong \
  --roles planner \
  --vector_type both \
  --direction_method diffmean \
  --output_path outputs/gsm8k_planner_diffmean.pt
```

Common extraction arguments are:

| Argument | Values | Description |
|---|---|---|
| `--vector_type` | `hidden`, `kv`, `both` | Selects node hidden-state directions, edge KV directions, or both. |
| `--direction_method` | `mean`, `diffmean`, `pca` | Selects the geometric direction estimator. |
| `--roles` | `planner`, `planner,critic`, `all` | Selects source roles for extraction. |
| `--pair_filter` | see pair filters below | Selects which matched examples are used. |
| `--edge_map` | `sequential`, `none` | Maps source roles to sequential handoff edges. |
| `--kv_position` | `all`, `first`, `last`, integer | Selects positions in the saved KV handoff tensors. |

## Step 4. Train RePS Directions

RePS trains an intervention-oriented direction from paired clean and attacked outputs. It can train either hidden-state directions or KV-cache directions.

Hidden-state RePS:

```bash
python -m latent_attack.vectors.reps \
  --clean_results_json outputs/gsm8k_clean.json \
  --attacked_results_json outputs/gsm8k_attack_planner.json \
  --pair_filter clean_correct_attack_wrong \
  --target_role planner \
  --layer 18 \
  --vector_type hidden \
  --epochs 5 \
  --batch_size 1 \
  --alpha_values 1,2,4,6 \
  --output_path outputs/gsm8k_planner_reps_hidden_l18.pt
```

KV-cache RePS:

```bash
python -m latent_attack.vectors.reps \
  --clean_results_json outputs/gsm8k_clean.json \
  --attacked_results_json outputs/gsm8k_attack_planner.json \
  --pair_filter clean_correct_attack_wrong \
  --target_role planner \
  --layer 18 \
  --vector_type kv \
  --kv_mode kv_both \
  --epochs 5 \
  --batch_size 1 \
  --alpha_values 1,2,4,6 \
  --output_path outputs/gsm8k_planner_reps_kv_l18.pt
```

The default preference pair uses `clean_raw` as the chosen output and `attacked_raw` as the rejected output. These can be changed with `--chosen_source` and `--rejected_source`.

## Step 5. Direction-Only Latent Injection

After a direction file is produced, it can be injected into a clean execution. The injection does not add adversarial text to the prompt, messages, logits, model parameters, or final answer.

Hidden-state injection:

```bash
python -m latent_attack.run_experiment \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --prompt sequential \
  --latent_steps 5 \
  --max_samples 100 \
  --output_path outputs/gsm8k_inject_hidden.json \
  --state_injection \
  --state_injection_vector_path outputs/gsm8k_planner_diffmean.pt \
  --state_injection_role planner \
  --state_injection_layers last \
  --state_injection_alpha 2.0
```

KV-cache injection:

```bash
python -m latent_attack.run_experiment \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --prompt sequential \
  --latent_steps 5 \
  --max_samples 100 \
  --output_path outputs/gsm8k_inject_kv.json \
  --kv_injection \
  --kv_injection_vector_path outputs/gsm8k_planner_diffmean.pt \
  --kv_injection_role planner \
  --kv_injection_mode kv_both \
  --kv_injection_layers all \
  --kv_injection_position all \
  --kv_injection_alpha_k 1.0 \
  --kv_injection_alpha_v 1.0
```

For RePS payloads, use the corresponding RePS output file in `--state_injection_vector_path` or `--kv_injection_vector_path`.

## Pair Filters

Vector extraction and RePS share the same matched-pair filters.

| Filter | Description |
|---|---|
| `clean_correct_attack_wrong` | Clean execution is correct and attacked execution is wrong. |
| `clean_correct` | Clean execution is correct. |
| `attack_wrong` | Attacked execution is wrong. |
| `clean_wrong_attack_correct` | Clean execution is wrong and attacked execution is correct. |
| `all` | Uses all matched pairs. |

Matched result files should share the same task configuration, `start_index`, and sample order.

## Output Files

Each run with `--output_path` writes:

- `<output_path>` with summary metadata and predictions.
- `<output_path>.jsonl` with streaming partial predictions.
- `<output_stem>_traces/` when trace export is enabled.

Direction files are saved as `.pt` payloads. Hidden-state payloads contain per-role vectors. KV payloads contain per-role K and V vectors arranged by layer.

## Minimal End-to-End Example

The following commands run the full latent attack chain on a small GSM8K subset.

```bash
# 1. Clean traces
python -m latent_attack.run_experiment \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --max_samples 100 \
  --output_path outputs/clean.json \
  --trace_export \
  --trace_save_hidden \
  --trace_save_kv

# 2. Direct-attack reference traces
python -m latent_attack.run_experiment \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --max_samples 100 \
  --output_path outputs/attack.json \
  --attack_surface role_prompt \
  --attack_type mi \
  --mi_roles planner \
  --trace_export \
  --trace_save_hidden \
  --trace_save_kv

# 3. Direction extraction
python -m latent_attack.vectors.extract \
  --clean_trace_dir outputs/clean_traces \
  --attacked_trace_dir outputs/attack_traces \
  --clean_results_json outputs/clean.json \
  --attacked_results_json outputs/attack.json \
  --pair_filter clean_correct_attack_wrong \
  --roles planner \
  --vector_type both \
  --direction_method diffmean \
  --output_path outputs/vector.pt

# 4. Latent injection
python -m latent_attack.run_experiment \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --max_samples 100 \
  --output_path outputs/injected.json \
  --state_injection \
  --state_injection_vector_path outputs/vector.pt \
  --state_injection_role planner \
  --state_injection_alpha 2.0
```

## Experimental Notes

The paper uses Qwen3-4B as the main backbone for all agents and reports Llama-3.2-3B-Instruct as an additional backbone. Use `--model_name` to select a HuggingFace model identifier or a local model path.

The evaluated latent attack surface contains node-level hidden states and edge-level KV-cache handoffs. Node interventions use `--state_injection`; edge interventions use `--kv_injection` with `k_only`, `v_only`, or `kv_both` carriers.

## Troubleshooting

If a run is interrupted, rerun the same command with `--resume_partial` and the same `--output_path`.

If extraction reports zero matched pairs, check that the clean and attacked result files use the same task, sample count, `start_index`, and ordering. Also check whether the selected pair filter has enough matching examples.

If a trace directory cannot be found, check the output path stem. For `outputs/clean.json`, the default trace directory is `outputs/clean_traces`.

If a local dataset is missing, place the required files under `data/` or switch to a task loaded through HuggingFace `datasets`.

## Citation

The BibTeX entry will be added after the paper metadata is finalized.

## License

This repository follows the license included in [LICENSE](LICENSE).
