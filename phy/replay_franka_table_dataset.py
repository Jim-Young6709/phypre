"""Replay saved Franka table joint targets in Isaac Lab and record MP4 videos.

From ``phypre``:

    python -m phy.replay_franka_table_dataset --headless --num-videos 10

Each video uses the saved object asset and state. Older datasets without the
pre-action state start from their first recorded observation instead.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1] / "outputs/frankatable/frankatable_datagen.hdf5"
)
INITIAL_KEYS = (
    "initial_joint_pos",
    "initial_joint_vel",
    "initial_robot_root_pose",
    "initial_object_state",
)


@dataclass
class Demo:
    name: str
    asset_id: str
    usd_path: Path
    actions: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    object_state: np.ndarray
    reference_pose: np.ndarray
    has_initial_state: bool


def load_demos(path: Path, count: int) -> tuple[list[Demo], float, int]:
    with h5py.File(path, "r") as file:
        names = sorted(
            (name for name in file["data"] if re.fullmatch(r"demo_\d+", name)),
            key=lambda name: int(name.removeprefix("demo_")),
        )[:count]
        if not names:
            raise ValueError(f"No demonstrations found in {path}")
        sim_dt = float(file.attrs["sim_dt"])
        control_dt = float(file.attrs["control_dt"])
        decimation = round(control_dt / sim_dt)
        if decimation < 1 or not np.isclose(decimation * sim_dt, control_dt):
            raise ValueError("Dataset control_dt must be a multiple of sim_dt")

        demos = []
        for name in names:
            group = file["data"][name]
            has_initial = all(key in group for key in INITIAL_KEYS)
            if has_initial:
                joint_pos = group["initial_joint_pos"][:]
                joint_vel = group["initial_joint_vel"][:]
                object_state = group["initial_object_state"][:]
                reference_pose = group["initial_robot_root_pose"][:]
            else:
                joint_pos = group["obs/joint_pos"][0]
                joint_vel = group["obs/joint_vel"][0]
                object_state = np.concatenate(
                    (group["obs/object_pose"][0], np.zeros(6, dtype=np.float32))
                )
                reference_pose = group["obs/eef_pose"][0]
            demos.append(
                Demo(
                    name=name,
                    asset_id=str(group.attrs["asset_id"]),
                    usd_path=Path(group.attrs["usd_path"]),
                    actions=group["actions"][:],
                    joint_pos=joint_pos,
                    joint_vel=joint_vel,
                    object_state=object_state,
                    reference_pose=reference_pose,
                    has_initial_state=has_initial,
                )
            )
    return demos, sim_dt, decimation


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--num-videos", type=int, default=10)
    parser.add_argument("--output-dir", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_videos <= 0:
        parser.error("--num-videos must be positive")

    dataset = args.dataset.expanduser().resolve()
    demos, sim_dt, decimation = load_demos(dataset, args.num_videos)
    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else dataset.parent
        / "replay_videos"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    if len(demos) < args.num_videos:
        print(
            f"[INFO] Dataset has {len(demos)} demos; recording all of them.", flush=True
        )
    if any(not demo.has_initial_state for demo in demos):
        print(
            "[WARNING] Some demos lack pre-action state; replay starts at their first recorded frame.",
            flush=True,
        )

    args.enable_cameras = True
    simulation_app = AppLauncher(args).app
    env = None
    recorder = None
    try:
        from phy.cfg.franka_base_env_cfg import make_franka_robot_cfg
        from phy.cfg.franka_table_env_cfg import FrankaTableEnvCfg
        from phy.franka_table import FrankaTableEnv
        from phy.utils.assets import ThorAsset
        from phy.utils.recording import DebugVideoRecorder

        assets = [ThorAsset(demo.asset_id, demo.usd_path) for demo in demos]

        class ReplayEnv(FrankaTableEnv):
            def _get_assets(self, _cfg):
                return assets

            def _pre_physics_step(self, targets: torch.Tensor) -> None:
                self.robot_dof_targets[:, self.arm_dof_indices] = targets[:, :7]
                self.robot_dof_targets[:, self.gripper_dof_indices] = targets[:, 7:8]

        joint_count = len(demos[0].joint_pos)
        if any(len(demo.joint_pos) != joint_count for demo in demos):
            raise ValueError("All selected demos must use the same robot")
        if joint_count not in (9, 13):
            raise ValueError(f"Unsupported robot joint count: {joint_count}")
        cfg = FrankaTableEnvCfg()
        cfg.seed = 0
        cfg.scene.num_envs = len(demos)
        cfg.start_object_idx = 0
        cfg.require_grasps = False
        cfg.use_robotiq_gripper = joint_count == 13
        cfg.robot_cfg = make_franka_robot_cfg(cfg.use_robotiq_gripper)
        cfg.enable_recording_camera = True
        cfg.sim.device = args.device
        cfg.sim.dt = sim_dt
        cfg.decimation = decimation
        cfg.sim.render_interval = decimation
        control_dt = sim_dt * decimation
        cfg.episode_length_s = max(
            cfg.episode_length_s,
            (max(len(demo.actions) for demo in demos) + 2) * control_dt,
        )

        env = ReplayEnv(cfg)
        env.reset()
        device = env.device
        joint_pos = torch.as_tensor(
            np.stack([d.joint_pos for d in demos]), device=device
        )
        joint_vel = torch.as_tensor(
            np.stack([d.joint_vel for d in demos]), device=device
        )
        object_state = torch.as_tensor(
            np.stack([d.object_state for d in demos]), device=device
        ).clone()
        env.set_robot_joint_state(joint_pos, joint_vel, forward_sim=True)
        env.scene.update(dt=0.0)

        eef_body_id = env.robot.find_bodies(cfg.eef_body_name)[0][0]
        for index, demo in enumerate(demos):
            reference = torch.as_tensor(demo.reference_pose, device=device)
            if demo.has_initial_state:
                offset = env.robot.data.root_pose_w[index, :3] - reference[:3]
            else:
                offset = (
                    env.robot.data.body_pose_w[index, eef_body_id, :3] - reference[:3]
                )
            object_state[index, :3] += offset
        env.object.write_root_state_to_sim(object_state)
        env.robot_dof_targets.copy_(joint_pos)
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.forward()

        names = [
            f"{demo.name}_{re.sub('[^A-Za-z0-9_.-]+', '_', demo.asset_id)}"
            for demo in demos
        ]
        recorder = DebugVideoRecorder(
            True, output_dir, max(1, round(1 / (2 * control_dt))), 2, len(demos)
        )
        recorder.start()
        env.sim.render()
        env._recording_camera.update(0.0, force_recompute=True)
        recorder.capture(
            env._recording_camera.data.output["rgb"].detach().cpu().numpy(),
            0,
            names,
        )

        max_steps = max(
            len(demo.actions) - int(not demo.has_initial_state) for demo in demos
        )
        for step in range(max_steps):
            targets = env.robot_dof_targets[:, env.logical_dof_indices].clone()
            active = []
            for index, demo in enumerate(demos):
                action_index = step + int(not demo.has_initial_state)
                if action_index < len(demo.actions):
                    targets[index] = torch.as_tensor(
                        demo.actions[action_index], device=device
                    )
                    active.append(index)
            env.step(targets)
            if active and (step + 1) % recorder.every == 0:
                rgb = env._recording_camera.data.output["rgb"].detach().cpu().numpy()
                recorder.capture(
                    rgb[active], step + 1, [names[index] for index in active]
                )

        print(f"[INFO] Recorded {len(demos)} replay videos in {output_dir}", flush=True)
    finally:
        if recorder is not None:
            recorder.close()
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
