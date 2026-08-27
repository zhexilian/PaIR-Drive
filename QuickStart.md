# Pair-Drive

Pair-Drive Overview. We have reconstructed the navsim infra to faster caching and evaluating.

```text
camera + LiDAR + status
          |
          v
DiffusionDrive base trajectory (8 x 3)
          |
          v
Parallel-IR expands 15 trajectories
          |
          v
offline cache: 16 trajectories + features
          |
          v
RWM sim_logits.argmax selection
          |
          v
NAVSIM PDM Score
```

The learned-model chain is split into two stages. DiffusionDrive and Parallel-IR
run while building the offline candidate cache; the final RWM evaluation reads
that cache and does not regenerate trajectories online.

## What is included

- Inference-only DiffusionDrive, Parallel-IR, and RWM architectures.
- DiffusionDrive-based offline candidate generation and per-candidate PDM reward
  caching.
- RWM selection with base-model provenance validation and selected-path PDM
  evaluation.
- NAVSIM 2.2 scene loading, scenario building, metric caching, PDM planner,
  simulator, scorer, and traffic-policy infrastructure.


## 1. Environment

See [`docs/environment.md`](docs/environment.md) for the inspected package
versions and CUDA 11.7 installation notes.

## 2. External data and checkpoints

### Prepare the standard NAVSIM dataset layout:

```text
OPENSCENE_DATA_ROOT/
├── navsim_logs/test/*.pkl
├── sensor_blobs/test/...
└── maps
    ├──nuplan-maps-v1.0.json
    └── ...
```
### Download checkpoints

Download the three checkpoints from [Zhexilian/Pair-drive](https://huggingface.co/Zhexilian/Pair-drive) and place them as follows:

```text
checkpoints/
├── diffusiondrive_navsim_88p1_PDMS.pth
├── parallel_ir.ckpt
└── RWM.ckpt
```

## 3. Build the NAVSIM metric cache

Metric-cache construction uses NAVSIM logs and nuPlan maps. Camera/LiDAR blobs
are not read in this stage. `PDMClosedPlanner` caches the observation,
centerline, drivable map, traffic lights, human trajectory, ego state, and
metadata needed by later PDM scoring.

```bash
# Edit the USER CONFIG block at the top of the script once, then run:
bash slurm_build_metric_cache.sh
```

The script defaults to 16 local worker processes and limits OMP/MKL to one
thread per process. Adjust `PAIRDRIVE_WORKERS` to match the allocated CPU and
available memory, or set `PAIRDRIVE_FORCE_CACHE=true` to overwrite existing
files.

```text
metric_cache/
├── metadata/*.csv
└── <log>/<scene_type>/<token>/metric_cache.pkl
```

The Hydra entry point is also available:

```bash
python -m navsim.planning.script.run_metric_caching \
  train_test_split=navtest worker=sequential \
  metric_cache_path=/path/to/metric_cache
```

## 4. Build offline trajectory candidates

Edit the launcher's `USER CONFIG` block; no shell variables need to be exported
before running it.

### 4.1 DiffusionDrive + Parallel-IR

DiffusionDrive's `best_trajectory` is passed to Parallel-IR as
`expert_feature`. The base trajectory is appended after the 15 expanded paths.

```bash
bash slurm_build_ddrive_candidate_cache.sh
```

DiffusionDrive uses stochastic diffusion noise. The launcher defaults to seed
0; edit `PAIRDRIVE_SEED` in the script to control it.

Candidate generation keeps one DiffusionDrive/Parallel-IR model instance on
`PAIRDRIVE_DEVICE`, then distributes the 16 candidate trajectories of each
scene across 16 persistent CPU processes for NAVSIM PDM scoring. Adjust
`PAIRDRIVE_WORKERS` for the allocated CPU and memory. Set
`PAIRDRIVE_SCORING_WORKER=sequential` only when debugging.

### 4.2 Candidate-cache contract

Each file is written to
`<feature-cache>/<log>/<token>/pairdrive_feature.gz`. Existing DiffusionDrive/RWM
caches named `transfuser_feature.gz` remain readable as a legacy compatibility
fallback, but new caches no longer use that name.



## 5. Run RWM selection and PDM evaluation

Evaluate a DiffusionDrive cache:

```bash
bash slurm_PIR_dd_test_RWM.sh
```

The launcher passes `diffusiondrive` as the expected base model; a cache that
explicitly declares a different base model is rejected.  

**Cached PDM rewards are used only to report the score of the RWM-selected trajectory. RWM selection depends solely on predicted sim_logits and never accesses ground-truth rewards.**

Evaluation runs in two stages. `torchrun` first splits tokens across 4
independent CPU ranks, and each rank performs batched RWM selection. Selected
trajectories are then sent through NAVSIM/nuPlan's `WorkerPool` for parallel PDM
simulation and scoring. After all ranks finish, rank 0 gathers the results and
writes two files:

- `rwm_pdm_<timestamp>.csv`: the complete per-scenario result in the original
  token order.
- `rwm_pdm_<timestamp>_summary.csv`: scenario counts, success rate, and the mean
  of each PDM metric, `EPDMS`, and final `PDMS`, computed over valid scenarios.

## License

Vendored NAVSIM-derived files retain the upstream Apache-2.0 license. See
`LICENSE` and `NOTICE`.
