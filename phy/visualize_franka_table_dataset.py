"""Replay Franka table HDF5 demonstrations in Viser.

From ``phypre``:

    python -m phy.visualize_franka_table_dataset

Open http://localhost:8080. Playback starts automatically. Use the demo
buttons or selector to switch trajectories, and drag Frame to seek.
Playback Speed scales replay from 1x to 10x.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import trimesh
import viser
import yourdfpy
from viser.extras import ViserUrdf
from viser.transforms import SO3

from phy.test.test_molmospaces_viser import extract_usd_mesh

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "phypre/outputs/frankatable/frankatable_datagen.hdf5"
DEFAULT_URDF = (
    PROJECT_ROOT
    / "curobo/src/curobo/content/assets/robot/franka_description/franka_panda_no_gripper.urdf"
)
ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))


class RobotiqGripper:
    """Isaac's Robotiq 2F-85 visual meshes, mounted at the saved panda_hand pose."""

    # Body, parent, pivot in meters, recorded joint column, rotation about X.
    # These frames and axes come from the Isaac USD's physics joints.
    LINKS = (
        ("base_link", None, (0, 0, 0), None, 0),
        ("left_outer_knuckle", "base_link", (0, -0.0306, 0.05466), 7, -1),
        ("right_outer_knuckle", "base_link", (0, 0.0306, 0.05466), 8, 1),
        ("left_outer_finger", "left_outer_knuckle", (0, 0, 0), None, 0),
        ("right_outer_finger", "right_outer_knuckle", (0, 0, 0), None, 0),
        ("left_inner_finger", "left_outer_finger", (0, -0.06776, 0.09809), 9, -1),
        ("right_inner_finger", "right_outer_finger", (0, 0.06776, 0.09809), 10, -1),
        ("left_inner_knuckle", "left_inner_finger", (0, -0.04986, 0.1046), 11, 1),
        ("right_inner_knuckle", "right_inner_finger", (0, 0.04986, 0.1046), 12, -1),
    )

    def __init__(self, server: viser.ViserServer):
        self.handles = {}
        paths = {}
        asset = Path(__file__).with_name("assets") / "robotiq_2f_85.npz"
        with np.load(asset) as meshes:
            for name, parent, _pivot, _column, _sign in self.LINKS:
                path = f"{paths[parent] if parent else '/eef/gripper'}/{name}"
                paths[name] = path
                self.handles[name] = server.scene.add_frame(path, show_axes=False)
                server.scene.add_mesh_trimesh(
                    f"{path}/visual",
                    trimesh.Trimesh(
                        vertices=meshes[f"{name}_vertices"],
                        faces=meshes[f"{name}_faces"],
                        vertex_colors=meshes[f"{name}_colors"],
                        process=False,
                    ),
                )

    def update(self, joint_pos: np.ndarray) -> None:
        if len(joint_pos) < 13:
            # Older Panda recordings provide only an opening width: approximate
            # that width with the Robotiq mechanism. New recordings store all DOFs.
            opening = (
                np.clip(joint_pos[7:9].sum(), 0, 0.085)
                if len(joint_pos) >= 9
                else 0.085
            )
            angle = 0.8203 * (1 - opening / 0.085)
            joint_pos = np.r_[joint_pos[:7], angle * np.array([1, 1, -1, 1, -1, -1])]
        for name, _parent, pivot, column, sign in self.LINKS:
            if column is not None:
                rotation = SO3.from_x_radians(sign * joint_pos[column])
                self.handles[name].wxyz = rotation.wxyz
                pivot = np.asarray(pivot)
                self.handles[name].position = pivot - rotation @ pivot


class DatasetViewer:
    def __init__(
        self,
        server: viser.ViserServer,
        dataset: Path,
        urdf_path: Path,
        first_demo: int | None,
    ):
        self.server = server
        self.dataset = dataset
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.setting_frame = False
        self.playing = True
        self.frame_index = 0
        self.demo_index = 0
        self.object_handle = None
        self.scene_center = np.zeros(3)

        with h5py.File(dataset, "r") as file:
            self.demo_names = sorted(
                (name for name in file["data"] if name.startswith("demo_")),
                key=lambda name: int(name.removeprefix("demo_")),
            )
            self.phase_names = {
                int(value): key
                for key, value in json.loads(file.attrs["phase_ids"]).items()
            }
            self.control_dt = float(file.attrs["control_dt"])
        if not self.demo_names:
            raise ValueError(f"No demonstrations found in {dataset}")

        self.urdf = yourdfpy.URDF.load(str(urdf_path))
        if set(self.urdf.actuated_joint_names) != set(ARM_JOINTS):
            raise ValueError(
                "The robot URDF must have exactly the seven panda arm joints"
            )
        for mesh_name in tuple(self.urdf.scene.geometry):
            node = mesh_name
            while node != self.urdf.scene.graph.base_frame:
                if node in {"panda_hand", "panda_leftfinger", "panda_rightfinger"}:
                    self.urdf.scene.delete_geometry(mesh_name)
                    break
                node = self.urdf.scene.graph.transforms.parents[node]
        self.arm_indices = [
            ARM_JOINTS.index(name) for name in self.urdf.actuated_joint_names
        ]
        server.scene.set_up_direction("+z")
        server.scene.add_grid("/grid", width=3.0, height=3.0, cell_size=0.25)
        self.robot_frame = server.scene.add_frame("/robot", show_axes=False)
        self.robot = ViserUrdf(server, self.urdf, root_node_name="/robot")
        self.eef_frame = server.scene.add_frame("/eef", show_axes=False)
        self.gripper = RobotiqGripper(server)
        self.table = server.scene.add_box(
            "/table", color=(140, 140, 132), dimensions=(0.8, 0.8, 0.05)
        )

        self.status = server.gui.add_markdown("")
        previous = server.gui.add_button("Previous Demo")
        next_demo = server.gui.add_button("Next Demo")
        self.demo_selector = server.gui.add_dropdown("Demo", options=self.demo_names)
        show_demo = server.gui.add_button("Show Demo")
        self.play_button = server.gui.add_button("Pause")
        restart = server.gui.add_button("Restart")
        self.speed_slider = server.gui.add_slider(
            "Playback Speed (×)", min=1.0, max=10.0, step=0.1, initial_value=1.0
        )
        self.frame_slider = server.gui.add_slider(
            "Frame", min=0, max=1, step=1, initial_value=0
        )

        @previous.on_click
        def _previous(_event):
            self.show_demo((self.demo_index - 1) % len(self.demo_names))

        @next_demo.on_click
        def _next(_event):
            self.show_demo((self.demo_index + 1) % len(self.demo_names))

        @show_demo.on_click
        def _show(_event):
            self.show_demo(self.demo_names.index(self.demo_selector.value))

        @self.play_button.on_click
        def _play(_event):
            with self.lock:
                self.playing = not self.playing
                self.last_tick = time.perf_counter()
                self.play_button.label = "Pause" if self.playing else "Play"
                self.update_status()

        @self.speed_slider.on_update
        def _speed(_event):
            with self.lock:
                self.last_tick = time.perf_counter()
                self.update_status()

        @restart.on_click
        def _restart(_event):
            with self.lock:
                self.show_frame(0)
                self.playing = True
                self.play_button.label = "Pause"
                self.update_status()

        @self.frame_slider.on_update
        def _seek(_event):
            if not self.setting_frame:
                with self.lock:
                    self.playing = False
                    self.play_button.label = "Play"
                    self.show_frame(int(self.frame_slider.value))

        @server.on_client_connect
        def _connect(client):
            client.camera.position = self.scene_center + np.array([1.5, -1.5, 1.0])
            client.camera.look_at = self.scene_center

        initial_name = (
            self.demo_names[0] if first_demo is None else f"demo_{first_demo}"
        )
        if initial_name not in self.demo_names:
            raise ValueError(f"{initial_name} is not in {dataset}")
        self.show_demo(self.demo_names.index(initial_name))
        self.thread = threading.Thread(target=self.play_loop, daemon=True)
        self.thread.start()

    def show_demo(self, index: int) -> None:
        with self.lock:
            name = self.demo_names[index]
            with h5py.File(self.dataset, "r") as file:
                group = file["data"][name]
                joint_pos = group["obs/joint_pos"][:]
                eef_pose = group["obs/eef_pose"][:]
                object_pose = group["obs/object_pose"][:]
                phase = group["phase"][:]
                asset_id = str(group.attrs["asset_id"])
                usd_path = Path(group.attrs["usd_path"])
            mesh = extract_usd_mesh(str(usd_path), False, sys.maxsize)

            self.urdf.update_cfg(dict(zip(ARM_JOINTS, joint_pos[0, :7])))
            hand_in_base = self.urdf.get_transform("panda_hand", "base_link")[:3, 3]
            base_position = eef_pose[0, :3] - hand_in_base
            self.robot_frame.position = base_position
            self.scene_center = object_pose[0, :3].copy()
            self.table.position = (self.scene_center[0], self.scene_center[1], 0.0)
            if self.object_handle is not None:
                self.object_handle.remove()
            self.object_handle = self.server.scene.add_mesh_simple(
                "/object",
                vertices=mesh.vertices,
                faces=mesh.faces,
                color=(240, 135, 45),
                side="double",
            )
            self.joint_pos = joint_pos
            self.eef_pose = eef_pose
            self.object_pose = object_pose
            self.phase = phase
            self.asset_id = asset_id
            self.demo_index = index
            self.demo_selector.value = name
            self.frame_slider.max = len(phase) - 1
            self.playing = True
            self.play_button.label = "Pause"
            self.show_frame(0)
            for client in self.server.get_clients().values():
                client.camera.position = self.scene_center + np.array([1.5, -1.5, 1.0])
                client.camera.look_at = self.scene_center

    def show_frame(self, frame: int) -> None:
        with self.lock:
            self.frame_index = frame
            self.playback_position = float(frame)
            self.last_tick = time.perf_counter()
            self.robot.update_cfg(self.joint_pos[frame, self.arm_indices])
            self.eef_frame.position = self.eef_pose[frame, :3]
            self.eef_frame.wxyz = self.eef_pose[frame, 3:7]
            self.gripper.update(self.joint_pos[frame])
            self.object_handle.position = self.object_pose[frame, :3]
            self.object_handle.wxyz = self.object_pose[frame, 3:7]
            self.setting_frame = True
            try:
                self.frame_slider.value = frame
            finally:
                self.setting_frame = False
            self.update_status()

    def update_status(self) -> None:
        name = self.demo_names[self.demo_index]
        phase = self.phase_names.get(int(self.phase[self.frame_index]), "unknown")
        self.status.content = (
            f"**{name} · {self.asset_id}**  \n"
            f"Frame {self.frame_index + 1}/{len(self.phase)} · "
            f"{self.frame_index * self.control_dt:.2f} s · {phase} · "
            f"{'Playing' if self.playing else 'Paused'} · {self.speed_slider.value:g}×"
        )

    def play_loop(self) -> None:
        # Advance by elapsed time so high speeds can skip display frames while
        # the robot, gripper, and object stay synchronized to the same sample.
        while not self.stop_event.wait(min(self.control_dt, 1 / 60)):
            with self.lock:
                now = time.perf_counter()
                elapsed = now - self.last_tick
                self.last_tick = now
                if self.playing:
                    position = (
                        self.playback_position
                        + elapsed * self.speed_slider.value / self.control_dt
                    ) % len(self.phase)
                    if int(position) != self.frame_index:
                        self.show_frame(int(position))
                    self.playback_position = position
                    self.last_tick = now

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--demo", type=int, help="Initial demo number (default: first available)"
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = viser.ViserServer(port=args.port)
    viewer = None
    try:
        viewer = DatasetViewer(
            server, args.dataset.expanduser(), args.urdf.expanduser(), args.demo
        )
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        server.stop()


if __name__ == "__main__":
    main()
