"""
--------------------------------------------
NOT THOROUGHLY REVIEWED Codex generated scripts.
--------------------------------------------

THOR asset discovery, metadata, selection, and grasp loading.
""" # TODO: asset loading is THOR specific, grasp loading is DROID set (include all THOR grasps) specific

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np
import torch

from .transforms import wxyz_to_matrix

ROOT = (
    Path(__file__).resolve().parents[3]
)  # TODO: this is very much overfit to the current folder structure
DEFAULT_USD_ROOT = ROOT / "molmospaces" / "molmo_spaces_isaac" / "assets" / "usd"
METADATA_PATHS = (
    ROOT
    / "molmospaces"
    / "molmo_spaces_isaac"
    / "src"
    / "molmo_spaces_isaac"
    / "resources"
    / "usd_assets_metadata.json",
)
GRASP_ROOT = Path.home() / ".cache" / "molmo-spaces-resources"


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
    asset_root_orientation: torch.Tensor,
) -> float:
    """Compute an asset's world-space height from its oriented bounding box."""
    bbox_size = (metadata.get(asset_id) or {}).get("bbox_size")
    if not isinstance(bbox_size, list) or len(bbox_size) < 3:
        return default_height
    local_extent = torch.as_tensor(
        bbox_size[:3],
        dtype=asset_root_orientation.dtype,
        device=asset_root_orientation.device,
    ).abs()
    return (wxyz_to_matrix(asset_root_orientation).abs() @ local_extent)[2].item()


def load_asset_grasps(asset_id: str, num_grasps: int = 0) -> torch.Tensor | None:
    """Load object-relative grasp transforms for one asset."""
    pattern = f"*/{asset_id}/{asset_id}_grasps_filtered.npz"
    paths = sorted((GRASP_ROOT / "grasps" / "droid").glob(pattern))
    if not paths:
        return None

    try:
        with np.load(paths[-1]) as data:
            transforms = np.asarray(data["transforms"], dtype=np.float32)
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
            raise ValueError(f"expected (N, 4, 4), got {transforms.shape}")
    except (BadZipFile, EOFError, KeyError, OSError, ValueError) as error:
        print(f"[WARN] Could not load grasps for {asset_id}: {error}")
        return None

    if num_grasps > 0:
        transforms = transforms[:num_grasps]
    return torch.from_numpy(transforms) if len(transforms) else None


def select_env_assets(
    assets: list[ThorAsset],
    num_envs: int,
    start_object_idx: int,
    num_grasps: int,
    require_grasps: bool,
) -> tuple[list[ThorAsset], dict[str, torch.Tensor]]:
    """Choose an asset per environment and optionally require grasp data."""

    # re-ordering assets list according to the start index
    start = start_object_idx % len(assets)
    ordered = [assets[(start + index) % len(assets)] for index in range(len(assets))]

    if not require_grasps:
        return [ordered[index % len(ordered)] for index in range(num_envs)], {}

    grasps: dict[str, torch.Tensor] = {}
    valid: list[ThorAsset] = []
    for asset in ordered:
        grasp_poses = load_asset_grasps(asset.asset_id, num_grasps)
        if grasp_poses is not None:
            grasps[asset.asset_id] = grasp_poses
            valid.append(asset)
        if len(valid) == num_envs:
            break

    if not valid:
        raise RuntimeError(f"No THOR assets with grasps found in {GRASP_ROOT}.")
    if len(valid) < num_envs:
        print(f"[WARN] Reusing {len(valid)} graspable assets across {num_envs} envs.")
    return [valid[index % len(valid)] for index in range(num_envs)], grasps
