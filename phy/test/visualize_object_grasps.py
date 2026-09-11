"""Browse filtered objects and their stored pickup grasps in Viser.

Run from ``phypre`` in the ``phy`` environment:

    python -m phy.test.visualize_object_grasps

Open http://localhost:8080 and use Previous Object / Next Object.
Objects use a plain mesh material; no Isaac Sim or physics is started.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import numpy as np
import viser

from phy.test.test_molmospaces_viser import extract_usd_mesh, grounded_object_mesh
from phy.utils.assets import (
    DEFAULT_USD_ROOT_GRASP,
    discover_thor_assets,
    load_asset_grasps,
)


def grasp_segments(transforms: np.ndarray) -> np.ndarray:
    """Use the same three-line geometry as visualize_grasp_poses (meters)."""
    points = np.array(
        [[0, -0.04, 0], [0, 0.04, 0], [0, -0.04, 0.04], [0, 0.04, 0.04]],
        dtype=np.float32,
    )
    points_w = points @ transforms[:, :3, :3].transpose(0, 2, 1)
    points_w += transforms[:, None, :3, 3]
    return points_w[:, [[0, 1], [0, 2], [1, 3]]].reshape(-1, 2, 3)


class ObjectGraspViewer:
    def __init__(self, server: viser.ViserServer, usd_root: Path, max_grasps: int = 0):
        self.server = server
        self.max_grasps = max_grasps
        self.assets = discover_thor_assets(usd_root)
        if not self.assets:
            raise RuntimeError(f"No THOR objects found under {usd_root}")
        self.index = 0
        self.lock = threading.Lock()
        self.center = np.zeros(3)
        self.distance = 1.0
        server.scene.set_up_direction("+z")
        self.status = server.gui.add_markdown("")
        previous = server.gui.add_button("Previous Object")
        next_object = server.gui.add_button("Next Object")

        @previous.on_click
        def _previous(_event):
            self.show_object(-1)

        @next_object.on_click
        def _next(_event):
            self.show_object(1)

        @server.on_client_connect
        def _connect(client):
            self.set_camera(client)

        self.show_object()

    def set_camera(self, client: viser.ClientHandle) -> None:
        with client.atomic():
            client.camera.position = self.center + self.distance * np.array([1, -1, 0.7])
            client.camera.look_at = self.center

    def show_object(self, step: int = 0) -> None:
        with self.lock:
            index = (self.index + step) % len(self.assets)
            asset = self.assets[index]
            raw_mesh = extract_usd_mesh(str(asset.usd_path), False, sys.maxsize)
            mesh, world_from_object = grounded_object_mesh(
                raw_mesh, (0.0, 0.0, 0.0), "y_up_to_z_up"
            )
            grasps = load_asset_grasps(asset.asset_id, num_grasps=0)
            total_count = 0 if grasps is None else len(grasps)
            if grasps is not None and self.max_grasps > 0:
                grasps = grasps[: self.max_grasps]
            count = 0 if grasps is None else len(grasps)
            bounds_min, bounds_max = mesh.bounds
            center = (bounds_min + bounds_max) * 0.5
            mesh.vertices[:] -= center
            world_from_object[:3, 3] -= center
            self.distance = max(float(np.linalg.norm(bounds_max - bounds_min)), 0.2)
            with self.server.atomic():
                self.server.scene.reset()
                self.server.scene.add_mesh_simple(
                    "/object",
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    color=(240, 135, 45),
                    side="double",
                )
                if grasps is not None:
                    segments = grasp_segments(world_from_object @ grasps.numpy())
                    for line_id, endpoints in enumerate(segments):
                        self.server.scene.add_spline_catmull_rom(
                            f"/grasps/{line_id}",
                            positions=endpoints,
                            segments=1,
                            color=(0, 255, 0),
                            line_width=3.0,
                        )
                self.index = index
                self.status.content = (
                    f"**{asset.asset_id}**  \n"
                    f"Object {index + 1}/{len(self.assets)} · Showing {count}/{total_count} grasps"
                )
                for client in self.server.get_clients().values():
                    self.set_camera(client)
            print(
                f"[INFO] {asset.asset_id}: showing {count}/{total_count} grasps.",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd-root", type=Path, default=DEFAULT_USD_ROOT_GRASP)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--max-grasps",
        type=int,
        default=0,
        help="Maximum grasps per object; 0 loads all (default).",
    )
    args = parser.parse_args()
    if args.max_grasps < 0:
        parser.error("--max-grasps must be nonnegative")
    server = viser.ViserServer(port=args.port)
    try:
        ObjectGraspViewer(server, args.usd_root, args.max_grasps)
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
