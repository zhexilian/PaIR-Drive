import logging

from hydra.utils import instantiate
from nuplan.planning.script.builders.utils.utils_type import is_target_type, validate_type
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from omegaconf import DictConfig
from typing import Optional
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def _infer_worker_count(worker, cfg: DictConfig) -> Optional[int]:
    # 1) 常见属性名
    for attr in ("num_workers", "n_workers", "max_workers", "threads", "n_jobs"):
        if hasattr(worker, attr):
            v = getattr(worker, attr)
            if isinstance(v, int):
                return v
            if isinstance(v, (list, tuple, set)):
                return len(v)

    # 2) 常见的 executor 包装（如 ThreadPoolExecutor）
    exec_ = getattr(worker, "executor", None)
    if exec_ is not None:
        for attr in ("_max_workers", "max_workers"):
            if hasattr(exec_, attr):
                v = getattr(exec_, attr)
                if isinstance(v, int):
                    return v

    # 3) 回退到配置字段
    try:
        for k in ("num_workers", "n_workers", "max_workers", "threads", "n_jobs"):
            if k in cfg.worker and isinstance(cfg.worker[k], int):
                return cfg.worker[k]
    except Exception:
        pass

    # 4) 类型特判
    try:
        if isinstance(worker, Sequential):
            return 1
    except NameError:
        pass

    return None

def build_worker(cfg: DictConfig) -> WorkerPool:
    """
    Builds the worker.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of WorkerPool.
    """
    logger.info("Building WorkerPool...")
    worker: WorkerPool = (
        instantiate(cfg.worker)
        if (is_target_type(cfg.worker, SingleMachineParallelExecutor) or is_target_type(cfg.worker, Sequential))
        else instantiate(cfg.worker, output_dir=cfg.output_dir)
    )
    validate_type(worker, WorkerPool)
    logger.info("Building WorkerPool...DONE!")
    return worker
