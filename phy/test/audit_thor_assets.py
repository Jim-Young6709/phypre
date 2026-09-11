"""
--------------------------------------------
Pure Codex generated scripts with no review.
--------------------------------------------

Audit THOR USD variants, metadata, bounding boxes, and grasps.

Run from the ``phypre`` directory:

    python -m phy.test.audit_thor_assets

The JSON report is written to ``outputs/thor_asset_audit.json`` by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from phy.utils.assets import (
    DEFAULT_USD_ROOT_ALL,
    GRASP_ROOT,
    METADATA_PATHS,
    ThorAsset,
    discover_thor_assets,
    load_asset_grasps,
    load_usd_asset_metadata,
)

OBJECT_ROOT = DEFAULT_USD_ROOT_ALL / "objects" / "thor"
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[2] / "outputs" / "thor_asset_audit.json"
)
USD_PATTERNS = ("**/*.usda", "**/*.usd", "**/*.usdc")
ISSUE_LABELS = (
    "missing_mesh_usda",
    "missing_metadata",
    "invalid_bbox_size",
    "no_valid_grasp",
)


def _asset_id(path: Path) -> str:
    """Extract an asset ID using the same naming rule as asset discovery."""
    return path.stem.removesuffix("_mesh").removesuffix("_prim")


def discover_all_variants(object_root: Path) -> dict[str, list[Path]]:
    """Collect every supported USD variant, grouped by logical asset ID."""
    if not object_root.is_dir():
        raise FileNotFoundError(f"THOR asset directory does not exist: {object_root}")

    variants: dict[str, list[Path]] = {}
    for pattern in USD_PATTERNS:
        for path in sorted(object_root.glob(pattern)):
            if path.name.startswith("."):
                continue
            variants.setdefault(_asset_id(path), []).append(path.absolute())
    if not variants:
        raise RuntimeError(f"No THOR USD assets found in: {object_root}")
    return variants


def audit_asset(
    asset: ThorAsset,
    variants: list[Path],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return one JSON-serializable audit record for an asset."""
    metadata_entry = metadata.get(asset.asset_id)
    has_metadata = metadata_entry is not None
    bbox_size = (
        metadata_entry.get("bbox_size")
        if isinstance(metadata_entry, dict)
        else None
    )
    has_valid_bbox_size = isinstance(bbox_size, list) and len(bbox_size) >= 3
    has_mesh_usda = any(
        path.name == f"{asset.asset_id}_mesh.usda" for path in variants
    )

    grasp_poses = load_asset_grasps(asset.asset_id)
    has_valid_grasp = grasp_poses is not None

    issues: list[str] = []
    if not has_mesh_usda:
        issues.append("missing_mesh_usda")
    if not has_metadata:
        issues.append("missing_metadata")
    if not has_valid_bbox_size:
        issues.append("invalid_bbox_size")
    if not has_valid_grasp:
        issues.append("no_valid_grasp")

    return {
        "asset_id": asset.asset_id,
        "preferred_usd_path": asset.usd_path.as_posix(),
        "usd_variants": [path.as_posix() for path in variants],
        "has_mesh_usda": has_mesh_usda,
        "has_metadata": has_metadata,
        "bbox_size": bbox_size,
        "has_valid_bbox_size": has_valid_bbox_size,
        "has_valid_grasp": has_valid_grasp,
        "grasp_count": len(grasp_poses) if grasp_poses is not None else 0,
        "issues": issues,
    }


def build_report() -> dict[str, Any]:
    """Audit every discovered THOR asset and build the complete report."""
    assets = discover_thor_assets()
    if not assets:
        raise FileNotFoundError(f"No THOR assets discovered in: {OBJECT_ROOT}")

    variants_by_asset = discover_all_variants(OBJECT_ROOT)
    metadata = load_usd_asset_metadata()
    records: list[dict[str, Any]] = []

    for index, asset in enumerate(assets, start=1):
        records.append(
            audit_asset(
                asset,
                variants_by_asset.get(asset.asset_id, []),
                metadata,
            )
        )
        if index % 100 == 0 or index == len(assets):
            print(f"[INFO] Audited {index}/{len(assets)} THOR assets.")

    labels = {
        issue: [
            record["asset_id"] for record in records if issue in record["issues"]
        ]
        for issue in ISSUE_LABELS
    }
    return {
        "settings": {
            "asset_root": OBJECT_ROOT.absolute().as_posix(),
            "metadata_paths": [path.absolute().as_posix() for path in METADATA_PATHS],
            "grasp_root": GRASP_ROOT.as_posix(),
        },
        "summary": {
            "asset_count": len(records),
            "usd_file_count": sum(
                len(paths) for paths in variants_by_asset.values()
            ),
            **{f"{issue}_count": len(asset_ids) for issue, asset_ids in labels.items()},
        },
        "labels": labels,
        "assets": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"JSON report path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report()

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)
        file.write("\n")

    print(f"[INFO] Wrote THOR asset audit report to {output_path.absolute()}.")
    for issue in ISSUE_LABELS:
        print(f"[INFO] {issue}: {report['summary'][f'{issue}_count']}")


if __name__ == "__main__":
    main()
