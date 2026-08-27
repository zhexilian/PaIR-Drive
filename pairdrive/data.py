"""Readers for the precomputed inputs consumed by RWM evaluation."""

import csv
import gzip
import lzma
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from navsim.planning.metric_caching.metric_cache import MetricCache


class MetricCacheLoader:
    """Load NAVSIM metric-cache pickles using their metadata manifest."""

    def __init__(self, cache_root: Path):
        self.cache_root = cache_root.expanduser().resolve()
        metadata_dir = self.cache_root / "metadata"
        manifests = sorted(metadata_dir.glob("*.csv"))
        if not manifests:
            raise FileNotFoundError(f"No metric-cache metadata CSV found under {metadata_dir}")

        self.metric_cache_paths: Dict[str, Path] = {}
        for manifest in manifests:
            with manifest.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    raw_path = Path(row["file_name"])
                    path = self._portable_cache_path(raw_path)
                    self.metric_cache_paths[path.parent.name] = path

    def _portable_cache_path(self, path: Path) -> Path:
        if path.is_file():
            return path
        # NAVSIM metadata commonly stores an absolute path.  Preserve the
        # log/scene-type/token/file suffix when the cache was relocated.
        if len(path.parts) >= 4:
            relocated = self.cache_root.joinpath(*path.parts[-4:])
            if relocated.is_file():
                return relocated
        return path

    @property
    def tokens(self) -> List[str]:
        return list(self.metric_cache_paths)

    def get_from_token(self, token: str) -> MetricCache:
        path = self.metric_cache_paths[token]
        if not path.is_file():
            raise FileNotFoundError(f"Metric cache for token {token} does not exist: {path}")
        with lzma.open(path, "rb") as stream:
            return pickle.load(stream)


class FeatureCacheLoader:
    """Load cached RWM input tensors indexed by scene token."""

    def __init__(self, cache_root: Path):
        self.cache_root = cache_root.expanduser().resolve()
        if not self.cache_root.is_dir():
            raise FileNotFoundError(f"Feature-cache directory does not exist: {self.cache_root}")
        self.feature_paths: Dict[str, Path] = {}
        # Load the historical filename first so the canonical Pair-Drive cache
        # wins when both exist for the same token.
        for filename in ("transfuser_feature.gz", "pairdrive_feature.gz"):
            paths = self.cache_root.glob(f"*/*/{filename}")
            self.feature_paths.update({path.parent.name: path for path in paths})
        if not self.feature_paths:
            raise FileNotFoundError(
                "No <log>/<token>/(pairdrive_feature.gz|transfuser_feature.gz) "
                f"files found under {self.cache_root}"
            )

    @property
    def tokens(self) -> List[str]:
        return list(self.feature_paths)

    def load(self, token: str) -> Dict:
        with gzip.open(self.feature_paths[token], "rb") as stream:
            return pickle.load(stream)


def load_tokens_csv(path: Optional[Path], token_column: str = "token") -> Optional[List[str]]:
    """Read an optional ordered token subset from a CSV file."""

    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Token CSV does not exist: {path}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or token_column not in reader.fieldnames:
            raise ValueError(f"CSV {path} does not contain a '{token_column}' column")
        return list(dict.fromkeys(row[token_column] for row in reader if row[token_column]))
