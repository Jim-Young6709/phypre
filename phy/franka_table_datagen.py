"""Generate Franka table grasp-lift trajectories with MolmoSpaces THOR objects.

Example:
    python -m phy.franka_table_datagen --headless \
        env.scene.num_envs=4 env.num_trajectories=4

For video, also pass ``--enable_cameras env.enable_recording_camera=true``.


TODO:
1. data filter detection mechanism
2. if IK fail just filter out the demo for that env
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch

from phy.utils.recording import DebugVideoRecorder, next_demo_index
from phy.utils.transforms import offset_and_perturb_transforms, poses_from_transforms

if TYPE_CHECKING:
    from phy.cfg.franka_table_datagen_cfg import FrankaTableDatagenCfg

TASK_NAME = "Phy-Franka-Table-Datagen-v0"
PHASE_IDS = {"reach": 0, "close": 1, "lift": 2}
TrajectoryBuffer = dict[str, list[np.ndarray | int]]


class FrankaTableDatagen:
    """Generate one batch with shared environment, trajectory, and recording state."""

    def __init__(
        self,
        cfg: FrankaTableDatagenCfg,
        h5_file: h5py.File,
        first_demo_id: int,
        active_count: int,
        generator: torch.Generator,
    ) -> None:
        from phy.franka_table import FrankaTableEnv

        self.cfg = cfg
        self.h5_file = h5_file
        self.first_demo_id = first_demo_id
        self.generator = generator

        env_cfg = cfg.copy()
        env_cfg.start_object_idx += first_demo_id
        env_cfg.scene.num_envs = active_count
        env_cfg.use_eef_control = True
        num_control_steps = sum(
            max(0, steps)
            for steps in (
                cfg.init_ik_steps,
                cfg.reach_steps,
                cfg.close_steps,
                cfg.lift_steps,
            )
        )
        env_cfg.episode_length_s = max(
            env_cfg.episode_length_s,
            (num_control_steps + 2) * env_cfg.sim.dt * env_cfg.decimation,
        )
        self.env = FrankaTableEnv(env_cfg)
        self.recorder = DebugVideoRecorder(
            cfg.enable_recording_camera,
            Path(cfg.video_dir).expanduser(),
            cfg.video_fps,
            cfg.video_every,
            cfg.record_video_envs,
        )

    def run(self) -> None:
        """Initialize, collect, and save one batch, then release its resources."""
        try:
            self.env.sim.set_camera_view(eye=[2.2, -2.2, 1.6], target=[0.55, 0.0, 0.45])
            self.env.reset()
            self.select_grasp_poses()
            self.initialize_pregrasp()
            self.collect_trajectories()
            for env_id in range(self.env.num_envs):
                self.write_demo(env_id)
            self.h5_file.flush()
            print(
                f"[INFO] Wrote demos {self.first_demo_id}..{self.first_demo_id + self.env.num_envs - 1}"
            )
        finally:
            self.recorder.close()
            self.env.close()

    def select_grasp_poses(self) -> None:
        """Select and store grasp poses, perturbed pregrasps, and gripper targets."""
        env = self.env
        cfg = self.cfg
        selected_grasps = [
            env.select_grasp(
                env_id,
                cfg.grasp_index,
            )
            for env_id in range(env.num_envs)
        ]
        grasp_transforms = torch.stack([transform for transform, _ in selected_grasps])
        self.grasp_indices = [index for _, index in selected_grasps]
        pregrasp_transforms = offset_and_perturb_transforms(
            grasp_transforms,
            cfg.pregrasp_offset,
            cfg.pregrasp_axis,
            cfg.pos_noise_std,
            cfg.rot_noise_std,
            self.generator,
        )
        self.grasp_pose_w = poses_from_transforms(grasp_transforms)
        self.pregrasp_pose_w = poses_from_transforms(pregrasp_transforms)
        self.open_width = torch.full_like(self.grasp_pose_w[:, :1], cfg.open_width)
        self.closed_width = torch.full_like(self.grasp_pose_w[:, :1], cfg.closed_width)

    def initialize_pregrasp(self) -> None:
        """Set the initial IK joint state and let the robot settle at the pregrasp."""
        env = self.env
        pregrasp_arm_targets, success = env.franka_ik(self.pregrasp_pose_w)
        if not torch.all(success):
            failed_env_ids = torch.nonzero(~success, as_tuple=False).flatten().tolist()
            raise RuntimeError(f"CuRobo IK failed for env IDs {failed_env_ids}.")

        joint_pos = env.robot.data.joint_pos.clone()
        joint_pos[:, env.arm_dof_indices] = pregrasp_arm_targets
        joint_pos[:, env.gripper_dof_indices] = self.open_width
        joint_pos = torch.clamp(
            joint_pos, env.robot_dof_lower_limits, env.robot_dof_upper_limits
        )
        env.robot_dof_targets.copy_(joint_pos)
        env._apply_action()
        env.scene.write_data_to_sim()
        env.set_robot_joint_state(joint_pos, forward_sim=True)

        # give it a few steps to stablize IK physics
        for _ in range(self.cfg.init_ik_steps):
            env.step(env.make_eef_w_actions(self.pregrasp_pose_w, self.open_width))

    def collect_trajectories(self) -> None:
        """Record the reach, close, and lift phases for the batch."""
        env = self.env
        cfg = self.cfg
        self.buffer: TrajectoryBuffer = defaultdict(list)
        self.video_names = []
        for env_id, asset in enumerate(env.selected_assets):
            asset_name = re.sub("[^A-Za-z0-9_.-]+", "_", asset.asset_id).strip("_")
            self.video_names.append(
                f"demo_{self.first_demo_id + env_id:06d}_env_{env_id}_{asset_name or 'unnamed'}"
            )
        self.recording_camera = env._recording_camera
        self.recorder.start()

        for step in range(cfg.reach_steps):
            alpha = (step + 1) / cfg.reach_steps
            target_pose_w = self.grasp_pose_w.clone()
            target_pose_w[:, :3] = (1.0 - alpha) * self.pregrasp_pose_w[
                :, :3
            ] + alpha * self.grasp_pose_w[:, :3]
            self.execute_step(target_pose_w, self.open_width, PHASE_IDS["reach"])

        for step in range(cfg.close_steps):
            alpha = (step + 1) / cfg.close_steps
            gripper_width = (1.0 - alpha) * self.open_width + alpha * self.closed_width
            self.execute_step(self.grasp_pose_w, gripper_width, PHASE_IDS["close"])

        for step in range(cfg.lift_steps):
            alpha = (step + 1) / cfg.lift_steps
            target_pose_w = self.grasp_pose_w.clone()
            target_pose_w[:, 2] += cfg.lift_height * alpha
            self.execute_step(target_pose_w, self.closed_width, PHASE_IDS["lift"])

    def execute_step(
        self, target_pose_w: torch.Tensor, gripper_width: torch.Tensor, phase_id: int
    ) -> None:
        """Advance one control step and capture its state and optional video frame."""
        env = self.env
        env.step(env.make_eef_w_actions(target_pose_w, gripper_width))
        step_index = len(self.buffer["phase"])
        self.record_step(target_pose_w, phase_id)
        if self.recording_camera is not None and step_index % self.recorder.every == 0:
            rgb = self.recording_camera.data.output["rgb"].detach().cpu().numpy()
            self.recorder.capture(rgb, step_index, self.video_names)

    def record_step(self, target_pose_w: torch.Tensor, phase_id: int) -> None:
        """Snapshot one control step into the trajectory buffer."""
        env = self.env
        values = {
            "actions": env.robot_dof_targets[:, env.logical_dof_indices],
            "obs/joint_pos": env.robot.data.joint_pos,
            "obs/joint_vel": env.robot.data.joint_vel,
            "obs/eef_pose": env.robot.data.body_pose_w[:, env._eef_body_id],
            "obs/object_pose": torch.stack(
                [obj.data.root_pose_w[0] for obj in env.objects]
            ),
            "target_eef_pose": target_pose_w,
            "gripper_width": env.robot_dof_targets[:, env.gripper_dof_index, None],
        }
        for key, value in values.items():
            self.buffer[key].append(value.detach().cpu().numpy().astype(np.float32))
        self.buffer["phase"].append(phase_id)

    def write_demo(self, env_id: int) -> None:
        """Write one environment's trajectory and asset metadata to HDF5."""
        env = self.env
        group = self.h5_file.require_group("data").create_group(
            f"demo_{self.first_demo_id + env_id}"
        )
        for key in (
            "actions",
            "target_eef_pose",
            "gripper_width",
            "phase",
            "obs/joint_pos",
            "obs/joint_vel",
            "obs/eef_pose",
            "obs/object_pose",
        ):
            values = np.asarray(
                self.buffer[key], dtype=np.int32 if key == "phase" else np.float32
            )
            if key != "phase" and len(values):
                values = values[:, env_id]
            group.create_dataset(key, data=values, compression="gzip")
        for key, pose in (
            ("grasp_pose", self.grasp_pose_w[env_id]),
            ("pregrasp_pose", self.pregrasp_pose_w[env_id]),
            ("object_pose", env.objects[env_id].data.root_pose_w[0]),
        ):
            group.create_dataset(
                key, data=pose.detach().cpu().numpy().astype(np.float32)
            )

        asset = env.selected_assets[env_id]
        group.attrs.update(
            env_id=env_id,
            asset_id=asset.asset_id,
            object_prim_path=f"{env.scene.env_prim_paths[env_id]}/Object",
            usd_path=asset.usd_path.as_posix(),
            asset_axis_convention=env.cfg.asset_axis_convention,
            grasp_index=self.grasp_indices[env_id],
            action_type="joint_position_targets_with_symmetric_gripper_width",
        )


def write_root_attrs(h5_file: h5py.File, cfg: FrankaTableDatagenCfg) -> None:
    """Store generation settings shared by every demo in the file."""
    h5_file.attrs.update(
        generator="phy.franka_table_datagen",
        phase_ids=json.dumps(PHASE_IDS),
        sim_dt=cfg.sim.dt,
        control_dt=cfg.sim.dt * cfg.decimation,
        reach_steps=cfg.reach_steps,
        close_steps=cfg.close_steps,
        lift_steps=cfg.lift_steps,
        lift_height=cfg.lift_height,
        pregrasp_offset=cfg.pregrasp_offset,
        pregrasp_axis=cfg.pregrasp_axis,
        asset_axis_convention=cfg.asset_axis_convention,
    )


def generate(cfg: FrankaTableDatagenCfg) -> None:
    num_envs = cfg.scene.num_envs
    if num_envs <= 0:
        raise ValueError(f"env.scene.num_envs must be positive, got {num_envs}.")
    if cfg.num_trajectories <= 0:
        raise ValueError(
            f"env.num_trajectories must be positive, got {cfg.num_trajectories}."
        )

    output_hdf5 = Path(cfg.output_hdf5).expanduser()
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    seed = cfg.seed or 0
    generator = torch.Generator(device=cfg.sim.device).manual_seed(seed)
    torch.manual_seed(seed)

    with h5py.File(output_hdf5, "a" if cfg.append else "w") as h5_file:
        write_root_attrs(h5_file, cfg)
        next_demo_id = next_demo_index(h5_file.require_group("data"))
        remaining = cfg.num_trajectories
        while remaining:
            active_count = min(num_envs, remaining)
            FrankaTableDatagen(
                cfg, h5_file, next_demo_id, active_count, generator
            ).run()
            next_demo_id += active_count
            remaining -= active_count

    print(f"[INFO] Finished {cfg.num_trajectories} trajectory demos -> {output_hdf5}")


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *hydra_args]
    simulation_app = AppLauncher(args).app

    try:
        from isaaclab_tasks.utils.hydra import hydra_task_config

        @hydra_task_config(TASK_NAME, None)
        def run(cfg: FrankaTableDatagenCfg, _agent_cfg: None) -> None:
            cfg.sim.device = args.device
            generate(cfg)

        run()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
