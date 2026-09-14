"""View an unchanged THOR MJCF object and raw pickup grasps in MuJoCo.

Run from ``phypre`` in an environment with a working ``mujoco.viewer``:

    python -m phy.test.visualize_mujoco_object_grasps --asset-id Apple_1

N/P: next/previous object. J: enter an asset ID in the terminal. Close the
window to quit. Switching objects opens a new viewer window.

Grasps are drawn directly in model world coordinates, without an axis
conversion, centering, scale change, body-pose multiplication, or TCP offset.
Only display colors and camera settings change; physics is never stepped.
"""

from __future__ import annotations

import argparse
import queue
import time
from pathlib import Path

import mujoco
import numpy as np

from phy.utils.assets import ROOT, load_asset_grasps, natural_key

DEFAULT_MJCF_ROOT = ROOT / "set_object" / "mjcf" / "thor"


def load_object(path: Path, max_grasps: int):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Match the plain orange object style without modifying its geometry/pose.
    visual = model.geom_group != 4
    model.geom_matid[visual] = -1
    model.geom_rgba[visual] = [240 / 255, 135 / 255, 45 / 255, 1]
    grasps = load_asset_grasps(path.stem)
    total = 0 if grasps is None else len(grasps)
    if grasps is None:
        poses = np.empty((0, 4, 4))
    else:
        poses = grasps.numpy()
        if max_grasps:
            poses = poses[:max_grasps]
    return model, data, poses, total


def draw_grasps(scene: mujoco.MjvScene, poses: np.ndarray) -> None:
    """Draw the same 8 cm wide, 4 cm long outlines at the raw saved poses."""
    if 3 * len(poses) > scene.maxgeom:
        raise ValueError(
            f"Viewer capacity exceeded; use --max-grasps {scene.maxgeom // 3} or less."
        )
    points = np.array([[0, -0.04, 0], [0, 0.04, 0], [0, -0.04, 0.04], [0, 0.04, 0.04]])
    scene.ngeom = 0
    for pose in poses:
        # These are the saved rotation/translation, with no additional pose.
        endpoints = points @ pose[:3, :3].T + pose[:3, 3]
        for start, end in ((0, 1), (0, 2), (1, 3)):
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).ravel(),
                np.array([0, 1, 0, 1], dtype=np.float32),
            )
            mujoco.mjv_connector(
                geom, mujoco.mjtGeom.mjGEOM_LINE, 3.0, endpoints[start], endpoints[end]
            )
            scene.ngeom += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf-root", type=Path, default=DEFAULT_MJCF_ROOT)
    parser.add_argument("--asset-id", default="Apple_1")
    parser.add_argument(
        "--max-grasps",
        type=int,
        default=64,
        help="0 shows all grasps, within viewer capacity.",
    )
    args = parser.parse_args()
    if args.max_grasps < 0:
        parser.error("--max-grasps must be nonnegative")
    assets = {
        path.stem: path
        for path in sorted(args.mjcf_root.expanduser().rglob("*.xml"))
        if path.stem == path.parent.name and path.stat().st_size
    }
    if args.asset_id not in assets:
        parser.error(f"Asset {args.asset_id!r} not found under {args.mjcf_root}")
    asset_ids = sorted(assets, key=natural_key)
    index = asset_ids.index(args.asset_id)

    try:
        import mujoco.viewer as mjviewer
    except ImportError as error:
        parser.exit(1, f"The installed MuJoCo native viewer cannot load: {error}\n")

    commands: queue.Queue[int] = queue.Queue()

    def on_key(key: int) -> None:
        if key in (ord("N"), ord("P"), ord("J")):
            commands.put(key)

    while True:
        asset_id = asset_ids[index]
        model, data, poses, total = load_object(assets[asset_id], args.max_grasps)
        print(
            f"[INFO] {asset_id}: object {index + 1}/{len(asset_ids)}; "
            f"showing {len(poses)}/{total} grasps. N/P: browse; J: jump via terminal.",
            flush=True,
        )
        command = None
        with mjviewer.launch_passive(model, data, key_callback=on_key) as viewer:
            with viewer.lock():
                viewer.opt.geomgroup[4] = 0
                viewer.opt.sitegroup[:] = 0
                viewer.cam.lookat[:] = model.stat.center
                viewer.cam.distance = max(2 * model.stat.extent, 0.2)
                draw_grasps(viewer.user_scn, poses)
            while viewer.is_running():
                viewer.sync()
                try:
                    command = commands.get_nowait()
                    break
                except queue.Empty:
                    time.sleep(1 / 60)
        if command is None:
            return
        if command == ord("J"):
            selected = input("Asset ID: ").strip()
            if selected not in assets:
                print(f"Unknown asset: {selected!r}; reopening {asset_id}.", flush=True)
            else:
                index = asset_ids.index(selected)
        else:
            index = (index + (1 if command == ord("N") else -1)) % len(asset_ids)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        pass
