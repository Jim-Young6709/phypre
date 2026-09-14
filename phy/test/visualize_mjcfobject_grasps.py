"""Browse THOR MJCF objects and their stored pickup grasps in Viser.

Run from ``phypre`` in the ``phy_mujoco`` environment:

    python -m phy.test.visualize_mjcfobject_grasps --max-grasps 64

Open http://localhost:8080. Use Previous Object / Next Object, or select an
Asset and click Show Asset. Objects use a plain mesh material; no simulation
or MuJoCo renderer is started. A maximum of 0 displays all stored grasps.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import mujoco
import numpy as np
import viser

from phy.utils.assets import ROOT, load_asset_grasps, natural_key

DEFAULT_MJCF_ROOT = ROOT / "set_object" / "mjcf" / "thor"
# Standalone THOR objects and their object-relative grasps are authored Y-up.
Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def discover_mjcf_assets(root: Path) -> dict[str, Path]:
    """Select the base XML per object, excluding _mesh/_prim/_old variants."""
    assets = {}
    for path in sorted(root.expanduser().rglob("*.xml")):
        if path.stem == path.parent.name and path.stat().st_size:
            assets[path.stem] = path
    return dict(sorted(assets.items(), key=lambda item: natural_key(item[0])))


def load_mjcf_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centered visual geometry and its matching object-to-view pose."""
    print(f"Loading object: {path}", flush=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    vertices, faces = [], []
    vertex_offset = 0
    for geom_id in range(model.ngeom):
        # MolmoSpaces puts collision geometry in group 4; visuals are meshes.
        if (
            model.geom_group[geom_id] == 4
            or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH
        ):
            continue
        mesh_id = model.geom_dataid[geom_id]
        vertex_start = model.mesh_vertadr[mesh_id]
        face_start = model.mesh_faceadr[mesh_id]
        points = model.mesh_vert[
            vertex_start : vertex_start + model.mesh_vertnum[mesh_id]
        ]
        triangles = model.mesh_face[
            face_start : face_start + model.mesh_facenum[mesh_id]
        ]
        # Compiled vertices include mesh scale; geom_xmat/geom_xpos account for
        # MuJoCo's mesh recentering as well as the complete body transform chain.
        points = (
            points @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
        )
        vertices.append(points @ Y_UP_TO_Z_UP.T)
        faces.append(triangles + vertex_offset)
        vertex_offset += len(points)
    if not vertices:
        raise ValueError(f"No visual meshes in {path.name}")
    vertices = np.concatenate(vertices)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, path.stem)
    if body_id < 0:
        raise ValueError(f"Object body {path.stem!r} not found in {path.name}")
    world_from_object = np.eye(4)
    world_from_object[:3, :3] = Y_UP_TO_Z_UP @ data.xmat[body_id].reshape(3, 3)
    world_from_object[:3, 3] = Y_UP_TO_Z_UP @ data.xpos[body_id] - center
    return vertices - center, np.concatenate(faces), world_from_object


def grasp_segments(transforms: np.ndarray) -> np.ndarray:
    """Match the three-line gripper outlines in visualize_object_grasps."""
    points = np.array(
        [[0, -0.04, 0], [0, 0.04, 0], [0, -0.04, 0.04], [0, 0.04, 0.04]],
        dtype=np.float32,
    )
    points_w = points @ transforms[:, :3, :3].transpose(0, 2, 1)
    points_w += transforms[:, None, :3, 3]
    return points_w[:, [[0, 1], [0, 2], [1, 3]]].reshape(-1, 2, 3)


class MjcfObjectGraspViewer:
    def __init__(self, server: viser.ViserServer, mjcf_root: Path, max_grasps: int = 0):
        self.server = server
        self.max_grasps = max_grasps
        self.assets = discover_mjcf_assets(mjcf_root)
        if not self.assets:
            raise RuntimeError(f"No THOR object MJCF files found under {mjcf_root}")
        self.asset_ids = list(self.assets)
        self.index = 0
        self.lock = threading.Lock()
        self.distance = 1.0
        server.scene.set_up_direction("+z")
        self.status = server.gui.add_markdown("")
        previous = server.gui.add_button("Previous Object")
        next_object = server.gui.add_button("Next Object")
        self.asset_selector = server.gui.add_dropdown("Asset", options=self.asset_ids)
        jump = server.gui.add_button("Show Asset")

        @previous.on_click
        def _previous(_event):
            self.show_object(step=-1)

        @next_object.on_click
        def _next(_event):
            self.show_object(step=1)

        @jump.on_click
        def _jump(_event):
            self.show_object(asset_id=self.asset_selector.value)

        @server.on_client_connect
        def _connect(client):
            self.set_camera(client)

        self.show_object()

    def set_camera(self, client: viser.ClientHandle) -> None:
        with client.atomic():
            client.camera.position = self.distance * np.array([1, -1, 0.7])
            client.camera.look_at = np.zeros(3)

    def show_object(self, step: int = 0, asset_id: str | None = None) -> None:
        with self.lock:
            index = (
                self.asset_ids.index(asset_id)
                if asset_id is not None
                else (self.index + step) % len(self.asset_ids)
            )
            asset_id = self.asset_ids[index]
            try:
                vertices, faces, world_from_object = load_mjcf_mesh(
                    self.assets[asset_id]
                )
                grasps = load_asset_grasps(asset_id, num_grasps=0)
            except (ValueError, OSError) as error:
                self.index = index
                self.asset_selector.value = asset_id
                self.server.scene.reset()
                self.status.content = (
                    f"**{asset_id}**  \nCould not load object: {error}"
                )
                print(f"[ERROR] {asset_id}: {error}", flush=True)
                return
            total_count = 0 if grasps is None else len(grasps)
            if grasps is not None and self.max_grasps > 0:
                grasps = grasps[: self.max_grasps]
            count = 0 if grasps is None else len(grasps)
            self.distance = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 0.2)
            with self.server.atomic():
                self.server.scene.reset()
                self.server.scene.add_mesh_simple(
                    "/object",
                    vertices=vertices,
                    faces=faces,
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
                self.asset_selector.value = asset_id
                self.status.content = (
                    f"**{asset_id}**  \n"
                    f"Object {index + 1}/{len(self.assets)} · Showing {count}/{total_count} grasps"
                )
                for client in self.server.get_clients().values():
                    self.set_camera(client)
            print(
                f"[INFO] {asset_id}: showing {count}/{total_count} grasps.", flush=True
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf-root", type=Path, default=DEFAULT_MJCF_ROOT)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--max-grasps",
        type=int,
        default=0,
        help="Maximum grasps per object; 0 loads all.",
    )
    args = parser.parse_args()
    if args.max_grasps < 0:
        parser.error("--max-grasps must be nonnegative")
    server = viser.ViserServer(port=args.port)
    try:
        MjcfObjectGraspViewer(server, args.mjcf_root, args.max_grasps)
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
