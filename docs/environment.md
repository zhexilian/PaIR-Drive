# Reproducing `pairdrive`

The repository environment is a reconstruction of the inference, cache-building,
and evaluation dependencies. The versions below
were inspected from that environment and pinned in `pyproject.toml`:

| Component | Version |
| --- | --- |
| Python | 3.9.23 |
| pip | 23.3.1 |
| PyTorch | 2.0.1 (the source environment reports `2.0.1+cu117`) |
| torchvision | 0.15.2 (source reports `0.15.2+cu117`) |
| timm | 1.0.20 |
| diffusers | 0.35.2 |
| OpenCV | 4.9.0.80 |
| Hydra | 1.2.0 |
| OmegaConf | 2.3.0 |
| NumPy | 1.26.4 |
| pandas | 2.3.3 |
| SciPy | 1.13.1 |
| Shapely | 2.0.7 |
| GeoPandas | 1.0.1 |
| Pyogrio | 0.11.1 |
| Rasterio | 1.3.11 |
| Matplotlib | 3.9.4 |
| psutil | 7.1.0 |
| aioboto3 / boto3 | 15.2.0 / 1.40.18 |
| retry | 0.9.2 |
| pytest | 8.4.2 |
| nuPlan devkit | 1.2.0, commit `ce3c323af01c0d7ec5672f7832ef53f9c679aab0` |

## Create the environment

Run from the repository root:

```bash
conda env create -f environment.yml
conda activate pairdrive
export PAIRDRIVE_ENV_NAME=pairdrive
```

To update an environment that was created while some dependencies were
missed, run from the repository root:

```bash
conda activate pairdrive
python -m pip install -e .
```

If the configured internal PyPI mirror fails with an SSL/TLS error, retry that
one installation against public PyPI:

```bash
python -m pip install --index-url https://pypi.org/simple -e .
```

## CUDA note

The captured source environment uses the PyTorch 2.0.1 CUDA 11.7 wheels. On a
CUDA machine, verify the driver can run CUDA 11.7 binaries:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The supplied evaluation launchers explicitly use `--device cpu`; CUDA is not
required for RWM trajectory selection or NAVSIM PDM scoring. They also set
`OMP_NUM_THREADS` and `MKL_NUM_THREADS` per rank to avoid CPU oversubscription.

## Verify installation

```bash
python -m pairdrive.evaluation.run_rwm_pdm_score --help
python -m pairdrive.evaluation.build_candidate_cache --help
python -m navsim.planning.script.run_metric_caching train_test_split=navtest --help
python -c "import torch, timm, diffusers, navsim, pairdrive; print(torch.__version__, timm.__version__, diffusers.__version__)"
```