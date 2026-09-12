"""Copy graspable, single-body THOR assets into a compatible USD root.

Run from ``phypre`` with the Isaac Lab Python environment:

    python -m phy.processing.filter_rigid_objects

The default output is ``set_object/usd/rigid`` at the workspace root,
which is also the default object set for datagen.
The output preserves ``thor/<variant>/<variant>.usda`` and textures,
copying the same preferred variant selected by the loader (mesh, prim, base).
Asset IDs stay unchanged, so the existing grasp cache and metadata still apply.
The output directory must be empty to prevent stale assets surviving a rerun.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from pxr import Tf, Usd, UsdPhysics

from phy.utils.assets import (
    DEFAULT_USD_ROOT_ALL,
    DEFAULT_USD_ROOT_GRASP,
    discover_thor_assets,
    load_asset_grasps,
)


def rigid_body_issue(usd_path: Path) -> str | None:
    """Check the default-prim subtree that UsdFileCfg will reference."""
    try:
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None or stage.GetCompositionErrors():
            return "invalid_usd"
        root = stage.GetDefaultPrim()
        if not root:
            return "missing_default_prim"
        # Match RigidObject's traversal: active prims, without instance proxies.
        prims = list(Usd.PrimRange(root))
        bodies = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(bodies) != 1:
            return f"rigid_body_count_{len(bodies)}"
        if not UsdPhysics.RigidBodyAPI(bodies[0]).GetRigidBodyEnabledAttr().Get():
            return "disabled_rigid_body"
        for prim in prims:
            # An articulation is enabled unless explicitly disabled; the
            # PhysX schema may not be registered in standalone USD Python.
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and (
                prim.GetAttribute("physxArticulation:articulationEnabled").Get()
                is not False
            ):
                return "enabled_articulation"
    except (Tf.ErrorException, OSError, RuntimeError) as error:
        return f"usd_error: {error}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd-root", type=Path, default=DEFAULT_USD_ROOT_ALL)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_USD_ROOT_GRASP,
    )
    args = parser.parse_args()
    source_root = args.usd_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    source_objects = (
        source_root / "objects" if (source_root / "objects").is_dir() else source_root
    )
    source_objects /= "thor"
    if (
        output_root == source_root
        or output_root in source_root.parents
        or output_root.is_relative_to(source_objects)
    ):
        parser.error("Output must not contain the source root or be inside its THOR objects.")
    if output_root.exists() and (
        not output_root.is_dir() or any(output_root.iterdir())
    ):
        parser.error(f"Output directory must be empty: {output_root}")
    assets = discover_thor_assets(source_root)
    if not assets:
        parser.error(f"No THOR USD assets found under {source_objects}")

    accepted = []
    rejected = []
    output_root.mkdir(parents=True, exist_ok=True)
    for index, asset in enumerate(assets, start=1):
        grasps = load_asset_grasps(asset.asset_id)
        if grasps is None or not np.isfinite(grasps.numpy()).all():
            reason = "no_valid_grasps"
        else:
            reason = rigid_body_issue(asset.usd_path)
        if reason is not None:
            rejected.append({"asset_id": asset.asset_id, "reason": reason})
        else:
            relative_path = asset.usd_path.relative_to(source_root)
            # Keep each complete variant directory so relative texture and
            # other resource paths resolve exactly as in the original set.
            shutil.copytree(asset.usd_path.parent, (output_root / relative_path).parent)
            accepted.append(
                {
                    "asset_id": asset.asset_id,
                    "usd_path": relative_path.as_posix(),
                    "grasp_count": len(grasps),
                }
            )
        if index % 100 == 0 or index == len(assets):
            print(
                f"[INFO] Checked {index}/{len(assets)} assets; kept {len(accepted)}.",
                flush=True,
            )

    summary = {
        "total": len(assets),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_reasons": dict(Counter(record["reason"] for record in rejected)),
    }
    report_path = output_root / "filter_report.json"
    report_path.write_text(
        json.dumps(
            {
                "source_root": str(source_root),
                "output_root": str(output_root),
                "summary": summary,
                "accepted": accepted,
                "rejected": rejected,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] {json.dumps(summary)}")
    print(f"[INFO] Saved filtered object set and report to {output_root}")


if __name__ == "__main__":
    main()
