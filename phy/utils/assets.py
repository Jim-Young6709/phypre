"""THOR USD discovery and metadata loading.""" # TODO: this is THOR specific

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .transforms import wxyz_to_matrix

ROOT = Path(__file__).resolve().parents[3] # TODO: this is very much overfit to the current folder structure
DEFAULT_USD_ROOT = (
    ROOT
    / "molmospaces"
    / "molmo_spaces_isaac"
    / "assets"
    / "usd"
)
METADATA_PATHS = (
    ROOT
    / "molmospaces"
    / "molmo_spaces_isaac"
    / "src"
    / "molmo_spaces_isaac"
    / "resources"
    / "usd_assets_metadata.json",
)


@dataclass(frozen=True) # frozen to prevent accidental mutation of the asset data
class ThorAsset:
    asset_id: str
    usd_path: Path


def natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    """Convert text into alternating lowercase-text and integer sort parts.

    Each part is tagged with ``1`` for text or ``0`` for a number so Python can
    compare every tuple safely. For example, ``"Apple_10"`` becomes
    ``((1, "apple_"), (0, 10), (1, ""))``. This places ``Apple_2`` before
    ``Apple_10``, unlike normal string sorting.
    """
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", value)
    )


def _asset_id(path: Path) -> str:
    """Extract an asset ID from a USD filename."""
    return path.stem.removesuffix("_mesh").removesuffix("_prim")


def _variant_rank(path: Path) -> int:
    """
    Rank USD variants:
    1. mesh (most accurate collision geometry)
    2. prim (use primitives such as cube etc. to approximate collision geometry)
    3. base usda file (fallback option)
    """
    if path.stem.endswith("_mesh"):
        return 0
    if path.stem.endswith("_prim"):
        return 1
    return 2


def discover_thor_assets() -> list[ThorAsset]:
    """Discover one preferred USD variant per THOR asset in the default root."""
    object_root = DEFAULT_USD_ROOT / "objects" / "thor"
    if not object_root.is_dir():
        return []

    assets: dict[str, Path] = {}
    for pattern in ("**/*.usda", "**/*.usd", "**/*.usdc"):
        for path in sorted(object_root.glob(pattern)):
            if path.name.startswith("."):
                continue
            asset_id = _asset_id(path)
            current = assets.get(asset_id)
            if current is None or _variant_rank(path) < _variant_rank(current):
                assets[asset_id] = path.absolute()
    return [
        ThorAsset(asset_id, assets[asset_id])
        for asset_id in sorted(assets, key=natural_key)
    ]


def load_usd_asset_metadata() -> dict[str, Any]:
    """Load and merge available USD asset metadata files."""
    metadata: dict[str, Any] = {}
    for path in METADATA_PATHS:
        if path.is_file():
            with path.open("r", encoding="utf-8") as file:
                metadata.update(json.load(file))
    return metadata


def object_height(
    asset_id: str,
    metadata: dict[str, Any],
    default_height: float,
    asset_root_orientation: tuple[float, float, float, float],
) -> float:
    """Compute an asset's world-space height from its oriented bounding box."""
    bbox_size = (metadata.get(asset_id) or {}).get("bbox_size")
    if not isinstance(bbox_size, list) or len(bbox_size) < 3:
        return default_height
    local_extent = np.abs(np.asarray(bbox_size[:3], dtype=np.float64))
    return float((np.abs(wxyz_to_matrix(asset_root_orientation)) @ local_extent)[2])
