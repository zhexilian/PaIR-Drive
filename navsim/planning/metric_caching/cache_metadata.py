"""Small portable manifest types for NAVSIM metric caches."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class CacheMetadataEntry:
    file_name: Path


@dataclass
class CacheResult:
    failures: int
    successes: int
    cache_metadata: List[Optional[CacheMetadataEntry]]


def save_cache_metadata(
    entries: List[CacheMetadataEntry], cache_path: Path, node_id: int
) -> Path:
    metadata_dir = cache_path / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_path = metadata_dir / f"cache_metadata_node_{node_id}.csv"
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file_name"])
        writer.writeheader()
        writer.writerows({"file_name": str(entry.file_name)} for entry in entries)
    return output_path

