# Execution-Aware Visual Grounding for VoxPoser-Style Manipulation

This repository extends the open-source [VoxPoser](https://voxposer.github.io/) RLBench demo with a more explicit visual grounding and execution-repair pipeline. The original VoxPoser code provides the LMP-based voxel planning framework and uses RLBench oracle object masks in the demo setting. This project focuses on the perception and robustness layer around that planner: grounding task-level object names from multi-view RGB-D observations, ranking candidate 3D object evidence, verifying generated code, and retrying failed executions with compact feedback.

The project was developed for language-conditioned robotic manipulation experiments in [RLBench](https://sites.google.com/view/rlbench).

## What Is Added

- **VLM-guided phrase proposal**: converts task object names such as `rubbish`, `cap`, or `stand` into detector-friendly visual phrases.
- **Open-vocabulary visual grounding**: integrates GroundingDINO, SAM2, and RGB-D fusion to convert phrase masks into robot-usable 3D object evidence.
- **Phrase-level candidate scorer**: ranks fused 3D candidates using camera agreement, detector confidence, point count, object height, phrase rank, and cross-view consistency.
- **Planning verification**: repairs simple object-name aliases and checks generated LMP code before execution.
- **Execution-feedback self-repair**: after a failed rollout, summarizes the failure trace and asks the LMP to regenerate code for a retry.
- **Benchmarks and diagnostics**: separates perception quality, oracle planning, and full-stack execution; writes detailed JSON/CSV/HTML diagnostics for failure analysis.

## Relationship to VoxPoser

This repository is built on VoxPoser, and the core LMP and voxel-map planning abstraction are reused from the original project. The main extensions in this repo are the VLM-guided perception backend, candidate scoring, planning verifier, self-repair benchmark loop, and evaluation scripts.

Please cite the original VoxPoser paper if you use this code:

```bibtex
@article{huang2023voxposer,
  title={VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models},
  author={Huang, Wenlong and Wang, Chen and Zhang, Ruohan and Li, Yunzhu and Wu, Jiajun and Fei-Fei, Li},
  journal={arXiv preprint arXiv:2307.05973},
  year={2023}
}
```

## Results Snapshot

The main perception benchmark compares selected 3D object groundings against RLBench oracle object masks:

| Setting | Success | Success Rate | Mean Center Error |
| --- | ---: | ---: | ---: |
| Original selected perception | 42 / 57 | 73.7% | 5.4 cm |
| + Phrase-level scorer | 50 / 57 | 87.7% | 2.5 cm |

Full-stack execution remains harder because failures can come from perception, LMP planning, motion planning, contact, or RLBench success conditions. In the slide-summary runs, tasks such as `put_rubbish_in_bin`, `lamp_off`, and `push_button` are relatively reliable, while `open_wine_bottle` and `take_umbrella_out_of_umbrella_stand` expose small-part grounding and whole-arm feasibility limitations.

## Setup

This code is intended to run with a display or a working RLBench/CoppeliaSim headless setup.

1. Create an environment for VoxPoser/RLBench. The original VoxPoser demo uses Python 3.9:

```bash
conda create -n voxposer-env python=3.9
conda activate voxposer-env
```

2. Install PyRep and RLBench following the official RLBench instructions:

```text
https://github.com/stepjam/RLBench#install
```

3. Install the lightweight Python dependencies in this repository:

```bash
pip install -r requirements.txt
```

4. Install perception dependencies for GroundingDINO/SAM2 in your environment. The exact versions are platform-dependent, but the perception backend expects PyTorch, Transformers-compatible GroundingDINO/SAM2 packages, OpenCV/Pillow-style image utilities, and NumPy/Open3D.

5. Configure an OpenAI or OpenAI-compatible endpoint for LMP/VLM calls:

```bash
export OPENAI_API_KEY="your_key_here"
export OPENAI_BASE_URL="http://localhost:8317/v1"  # optional, for local OpenAI-compatible servers
```

Do not hard-code API keys in notebooks or source files.

## Running

All commands below assume the working directory is the repository root.

### Perception Benchmark

Run VLM-guided perception on a task and evaluate selected object groundings:

```bash
python src/perception/run_task_perception_benchmark.py \
  --tasks put_rubbish_in_bin \
  --episodes 3 \
  --backend-type inprocess \
  --use-openai-vlm \
  --vlm-base-url http://localhost:8317/v1 \
  --keep-going
```

Outputs are written under `src/perception_dumps/`, which is intentionally ignored by git.

### Offline Perception Replay

After collecting perception dumps, replay candidate scoring without rerunning the simulator:

```bash
python src/perception/replay_scorer_benchmark.py \
  --root src/perception_dumps/benchmark_live/YOUR_RUN \
  --out-dir src/perception_dumps/eval/replay_scorer
```

### Oracle Planning Benchmark

Use RLBench oracle masks to evaluate planning without perception errors:

```bash
python src/planning/run_oracle_planning_benchmark.py \
  --tasks push_button \
  --episodes 5 \
  --vision oracle \
  --self-repair-retries 2 \
  --keep-going
```

### Full-Stack Benchmark

Run the full perception-planning-execution loop:

```bash
python src/planning/run_oracle_planning_benchmark.py \
  --tasks take_umbrella_out_of_umbrella_stand \
  --episodes 1 \
  --vision perception \
  --backend-type inprocess \
  --use-openai-vlm \
  --vlm-base-url http://localhost:8317/v1 \
  --openai-base-url http://localhost:8317/v1 \
  --self-repair-retries 2 \
  --keep-going
```

Outputs are written under `src/planning_dumps/`, which is also ignored by git.

## Code Structure

Core VoxPoser components:

- `src/LMP.py`: Language Model Program implementation and generated-code execution.
- `src/interfaces.py`: LMP-facing APIs for voxel maps, object parsing, and execution.
- `src/planners.py`: Voxel-map trajectory planner.
- `src/controllers.py`: RLBench action controller.
- `src/prompts/rlbench/`: LMP prompts for planning, composing, parsing objects, and generating maps.
- `src/envs/rlbench_env.py`: RLBench wrapper used by VoxPoser and the new perception backend.

Project extensions:

- `src/perception/backend.py`: perception backend used by `rlbench_env.py`.
- `src/perception/scorer_v2.py`: phrase-level and per-view candidate scoring.
- `src/perception/run_task_perception_benchmark.py`: live perception benchmark over RLBench tasks.
- `src/perception/benchmark_perception.py`: object-level perception metric aggregation.
- `src/perception/replay_scorer_benchmark.py`: offline scorer replay benchmark.
- `src/perception/generate_failure_report.py`: browsable failure report generator.
- `src/planning_verifier.py`: lightweight verifier and object alias repair for generated code.
- `src/planning/run_oracle_planning_benchmark.py`: oracle planning and full-stack execution benchmark with self-repair retries.

Configuration:

- `src/configs/rlbench_config.yaml`: LMP, planner, controller, and visualization configuration.
- `src/envs/task_object_names.json`: task-to-object mapping used by the benchmark scripts.
- `.gitignore`: excludes generated perception/planning dumps, caches, logs, and local environment files.

## Generated Files

The following paths are generated during experiments and are not meant to be committed:

- `src/perception_dumps/`
- `src/planning_dumps/`
- `src/cache/`
- `*.log`
- `.env`

These directories can become very large because they store RGB images, masks, point clouds, diagnostics, generated code logs, and benchmark summaries.

## Known Limitations

- The candidate scorer is hand-designed rather than learned.
- Small-part grounding, such as bottle caps and umbrella handles, remains difficult.
- The planner mainly reasons about end-effector waypoints, not full-arm feasibility.
- Self-repair uses compact execution feedback for retries, but it is not a long-term memory system.
- Full-stack runs are expensive because they combine simulator execution, VLM calls, LMP calls, detection, segmentation, and RGB-D fusion.

## Acknowledgments

This project builds on:

- [VoxPoser](https://voxposer.github.io/) for LMP-based voxel-map planning.
- [RLBench](https://sites.google.com/view/rlbench) for the manipulation simulation environment.
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for open-vocabulary detection.
- [SAM2](https://github.com/facebookresearch/sam2) for segmentation.
- [Code as Policies](https://code-as-policies.github.io/) for the LMP-style robot programming idea used by VoxPoser.
