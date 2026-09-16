"""Generate Franka table grasp-lift trajectories with MolmoSpaces THOR objects.

Example:
    python -m phy.franka_table_datagen --headless \
        env.scene.num_envs=4 env.num_trajectories=4

For video, also pass ``--enable_cameras env.enable_recording_camera=true``.
For grasp/pregrasp GUI markers, pass ``env.debug_grasp_vis=true`` without ``--headless``.
Per-batch object outcomes are written beside the HDF5 file under ``<stem>_logs/<run>/``.


TODO:
1. if IK fail just filter out the demo for that env
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from collections import defaultdict
from datetime import UTC, datetime
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
        max_demos: int,
        generator: torch.Generator,
    ) -> None:
        from isaaclab.sim import create_new_stage

        from phy.franka_table import FrankaTableEnv
        # cfg overrides
        if cfg.use_robotiq_gripper:
            cfg.open_width = 0.0
            cfg.closed_width = 0.8203
            cfg.gripper_ik_offset = 0.155
        else:
            cfg.open_width = 0.04
            cfg.closed_width = 0.0
            cfg.gripper_ik_offset = 0.1025

        self.cfg = cfg
        self.h5_file = h5_file
        self.first_demo_id = first_demo_id
        self.max_demos = max_demos
        self.generator = generator
        self._grasp_debug_draw = None

        env_cfg = cfg.copy()
        env_cfg.start_object_idx += first_demo_id
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
        # env.close() leaves the stage and timeline alive between batches.
        create_new_stage()
        self.env = FrankaTableEnv(env_cfg)
        self.recorder = DebugVideoRecorder(
            cfg.enable_recording_camera,
            Path(cfg.video_dir).expanduser(),
            cfg.video_fps,
            cfg.video_every,
            cfg.record_video_envs,
        )

    def run(self, report_path: Path) -> int:
        """Save successful demos from one batch and return the number saved."""
        try:
            self.env.sim.set_camera_view(eye=[2.2, -2.2, 1.6], target=[0.55, 0.0, 0.45])
            self.env.reset()
            self.select_grasp_poses()
            self.initialize_pregrasp()
            initial_object_heights = torch.stack(
                [obj.data.root_pose_w[0, 2] for obj in self.env.objects]
            )
            self.collect_trajectories()
            final_object_heights = torch.stack(
                [obj.data.root_pose_w[0, 2] for obj in self.env.objects]
            )
            successful_env_ids = torch.nonzero(
                final_object_heights >= initial_object_heights + 0.05,
                as_tuple=False,
            ).flatten().tolist()
            num_successes = len(successful_env_ids)
            saved_env_ids = successful_env_ids[: self.max_demos]
            for demo_offset, env_id in enumerate(saved_env_ids):
                self.write_demo(env_id, self.first_demo_id + demo_offset)
            num_saved = len(saved_env_ids)
            self.h5_file.flush()
            self.write_batch_report(
                report_path,
                final_object_heights - initial_object_heights,
                successful_env_ids,
                saved_env_ids,
            )
            print(
                f"[INFO] Batch success rate: {num_successes}/{self.env.num_envs} "
                f"({num_successes / self.env.num_envs:.1%}); wrote {num_saved} demos"
            )
            return num_saved
        finally:
            if self._grasp_debug_draw is not None:
                self._grasp_debug_draw.clear_lines()
            self.recorder.close()
            self.env.close()

    def write_batch_report(
        self,
        path: Path,
        height_gains: torch.Tensor,
        successful_env_ids: list[int],
        saved_env_ids: list[int],
    ) -> None:
        """Report lift outcomes and observed initialization issues for every object.

        init_pose_collision means the target pregrasp world-frame z is below 0.05 m.
        Initialization issues are diagnostics, not proof of why the lift failed.
        """
        demo_ids = {
            env_id: self.first_demo_id + offset
            for offset, env_id in enumerate(saved_env_ids)
        }
        num_other_failures = 0
        with path.open("w", encoding="utf-8") as report:
            for env_id, (asset, height_gain, pregrasp_z) in enumerate(
                zip(self.env.selected_assets, height_gains.tolist(), self.pregrasp_pose_w[:, 2].tolist())
            ):
                success = env_id in successful_env_ids
                reasons = []
                if not success:
                    if not self.ik_success[env_id]:
                        reasons.append("ik")
                    if self.init_pose_collision[env_id]:
                        reasons.append("init_pose_collision")
                    if not reasons:
                        reasons.append("other_lift_below_threshold")
                        num_other_failures += 1
                report.write(
                    f"{asset.asset_id}\t{'SUCCESS' if success else 'FAIL'}"
                    f"\treason={','.join(reasons) or '-'}\tenv_id={env_id}"
                    f"\tik={'PASS' if self.ik_success[env_id] else 'FAIL'}"
                    f"\tpregrasp_z_m={pregrasp_z:.6f}\tlift_m={height_gain:.6f}"
                    f"\tsaved_demo={demo_ids.get(env_id, '-')}\n"
                )
        print(f"[INFO] Batch object report -> {path}")
        print(
            f"[INFO] Batch other failure rate: {num_other_failures}/{self.env.num_envs} "
            f"({num_other_failures / self.env.num_envs:.1%})"
        )

    def select_grasp_poses(self) -> None:
        """Try grasps sequentially until pregrasp height and IK pass, or grasps run out."""
        env = self.env
        cfg = self.cfg
        grasp_counts = [
            len(env.object_grasps[asset.asset_id]) for asset in env.selected_assets
        ]
        attempts = [0] * env.num_envs
        self.grasp_indices = [0] * env.num_envs
        self.ik_success = [False] * env.num_envs
        self.init_pose_collision = [False] * env.num_envs
        grasp_transforms = env.robot.data.joint_pos.new_empty((env.num_envs, 4, 4))
        pregrasp_transforms = torch.empty_like(grasp_transforms)
        self.pregrasp_arm_targets = env.robot.data.joint_pos[:, env.arm_dof_indices].clone()
        pending = list(range(env.num_envs))
        while pending:
            height_retries = pending
            while height_retries:
                selected_grasps = [
                    env.select_grasp(env_id, cfg.grasp_index + attempts[env_id])
                    for env_id in height_retries
                ]
                # Convert fingertip-midpoint grasps to panda_hand IK targets.
                candidates = offset_and_perturb_transforms(
                    torch.stack([transform for transform, _ in selected_grasps]),
                    cfg.gripper_ik_offset, "-z", 0.0, 0.0, self.generator,
                )
                pregrasps = offset_and_perturb_transforms(
                    candidates, cfg.pregrasp_offset, cfg.pregrasp_axis,
                    cfg.pos_noise_std, cfg.rot_noise_std, self.generator,
                )
                grasp_transforms[height_retries] = candidates
                pregrasp_transforms[height_retries] = pregrasps
                for env_id, (_, index), below_table in zip(
                    height_retries, selected_grasps, (pregrasps[:, 2, 3] < 0.05).tolist()
                ):
                    attempts[env_id] += 1
                    self.grasp_indices[env_id] = index
                    self.init_pose_collision[env_id] = below_table
                height_retries = [
                    env_id for env_id in height_retries
                    if self.init_pose_collision[env_id] and attempts[env_id] < grasp_counts[env_id]
                ]

            self.pregrasp_pose_w = poses_from_transforms(pregrasp_transforms)
            # Keep full environment order for IK's root transforms and CUDA graphs.
            # Exhausted low pregrasps still need joint targets for normal rollout.
            arm_targets, success = env.franka_ik(self.pregrasp_pose_w)
            self.pregrasp_arm_targets[pending] = arm_targets[pending]
            for env_id, ik_success in zip(pending, success[pending].tolist()):
                self.ik_success[env_id] = ik_success
            pending = [
                env_id for env_id in pending
                if not self.ik_success[env_id] and attempts[env_id] < grasp_counts[env_id]
            ]

        failed_env_ids = [
            env_id for env_id in range(env.num_envs)
            if not self.ik_success[env_id] or self.init_pose_collision[env_id]
        ]
        if failed_env_ids:
            print(f"[WARNING] Prefilter failed for env IDs {failed_env_ids}")

        self.grasp_pose_w = poses_from_transforms(grasp_transforms)
        self.open_width = torch.full_like(self.grasp_pose_w[:, :1], cfg.open_width)
        self.closed_width = torch.full_like(self.grasp_pose_w[:, :1], cfg.closed_width)
        if cfg.debug_grasp_vis and env.sim.has_gui():
            self.visualize_grasp_poses(grasp_transforms, pregrasp_transforms)

    def visualize_grasp_poses(
        self, grasp_transforms: torch.Tensor, pregrasp_transforms: torch.Tensor
    ) -> None:
        """Draw persistent three-line grippers at the selected world-frame poses."""
        from isaacsim.util.debug_draw import _debug_draw

        self._grasp_debug_draw = _debug_draw.acquire_debug_draw_interface()
        self._grasp_debug_draw.clear_lines()
        # Crossbar at the hand origin along Y, with short fingers extending in +Z.
        half_width = self.cfg.open_width
        points = grasp_transforms.new_tensor(
            [
                [0.0, -half_width, 0.0],
                [0.0, half_width, 0.0],
                [0.0, -half_width, 0.04],
                [0.0, half_width, 0.04],
            ]
        )
        for transforms, color in (
            (grasp_transforms, (0.0, 1.0, 0.0, 1.0)),
            (pregrasp_transforms, (0.0, 1.0, 1.0, 1.0)),
        ):
            points_w = points @ transforms[:, :3, :3].transpose(-1, -2)
            points_w += transforms[:, None, :3, 3]
            starts = points_w[:, [0, 0, 1]].reshape(-1, 3).cpu().tolist()
            ends = points_w[:, [1, 2, 3]].reshape(-1, 3).cpu().tolist()
            self._grasp_debug_draw.draw_lines(
                starts, ends, [color] * len(starts), [3.0] * len(starts)
            )

    def initialize_pregrasp(self) -> None:
        """Set the initial IK joint state and let the robot settle at the pregrasp."""
        env = self.env
        num_below_table = sum(self.init_pose_collision)
        num_ik_failed = env.num_envs - sum(self.ik_success)
        print(
            f"[INFO] Batch pregrasps below z=0.05 m: {num_below_table}/{env.num_envs} "
            f"({num_below_table / env.num_envs:.1%}); "
            f"env IDs {[i for i, below in enumerate(self.init_pose_collision) if below]}\n"
            f"[INFO] Batch pregrasp IK failure rate: {num_ik_failed}/{env.num_envs} "
            f"({num_ik_failed / env.num_envs:.1%}); "
            f"env IDs {[i for i, success in enumerate(self.ik_success) if not success]}"
        )
        joint_pos = env.robot.data.joint_pos.clone()
        joint_pos[:, env.arm_dof_indices] = self.pregrasp_arm_targets
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
        if self.cfg.print_ctrl_err:
            self.print_control_errors("pregrasp", self.pregrasp_pose_w, self.open_width)

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
        if cfg.reach_steps > 0 and cfg.print_ctrl_err:
            self.print_control_errors("reach", target_pose_w, self.open_width)

        for step in range(cfg.close_steps):
            self.execute_step(self.grasp_pose_w, self.closed_width, PHASE_IDS["close"])
        if cfg.close_steps > 0 and cfg.print_ctrl_err:
            self.print_control_errors("close", self.grasp_pose_w, self.closed_width)

        for step in range(cfg.lift_steps):
            alpha = (step + 1) / cfg.lift_steps
            target_pose_w = self.grasp_pose_w.clone()
            target_pose_w[:, 2] += cfg.lift_height * alpha
            self.execute_step(target_pose_w, self.closed_width, PHASE_IDS["lift"])
        if cfg.lift_steps > 0 and cfg.print_ctrl_err:
            self.print_control_errors("lift", target_pose_w, self.closed_width)

    def print_control_errors(
        self, phase: str, target_pose_w: torch.Tensor, gripper_width: torch.Tensor
    ) -> None:
        """Print hand-pose and per-finger-width errors in environment-index order."""
        from isaaclab.utils.math import quat_error_magnitude

        actual_pose_w = self.env.robot.data.body_pose_w[:, self.env._eef_body_id]
        position_error = torch.linalg.vector_norm(
            target_pose_w[:, :3] - actual_pose_w[:, :3], dim=-1
        ).tolist()
        rotation_error = torch.rad2deg(
            quat_error_magnitude(target_pose_w[:, 3:7], actual_pose_w[:, 3:7])
        ).tolist()
        gripper_error = torch.abs(
            gripper_width[:, 0]
            - self.env.robot.data.joint_pos[:, self.env.gripper_dof_index]
        ).tolist()
        print(
            f"[INFO] {phase} control errors (env-index order): "
            f"\nposition (m)=[{', '.join(f'{error:.4f}' for error in position_error)}]"
            f"\nrotation (deg)=[{', '.join(f'{error:.4f}' for error in rotation_error)}]"
            f"\ngripper (m, per-finger)=[{', '.join(f'{error:.4f}' for error in gripper_error)}]"
        )

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

    def write_demo(self, env_id: int, demo_id: int) -> None:
        """Write one environment's trajectory and asset metadata to HDF5."""
        env = self.env
        group = self.h5_file.require_group("data").create_group(
            f"demo_{demo_id}"
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
    log_dir = (
        output_hdf5.parent / f"{output_hdf5.stem}_logs"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    log_dir.mkdir(parents=True)
    seed = cfg.seed or 0
    generator = torch.Generator(device=cfg.sim.device).manual_seed(seed)
    torch.manual_seed(seed)

    with h5py.File(output_hdf5, "a" if cfg.append else "w") as h5_file:
        write_root_attrs(h5_file, cfg)
        next_demo_id = next_demo_index(h5_file.require_group("data"))
        remaining = cfg.num_trajectories
        batch_index = 0
        while remaining:
            num_saved = FrankaTableDatagen(
                cfg, h5_file, next_demo_id, remaining, generator
            ).run(log_dir / f"batch_{batch_index:06d}.txt")
            next_demo_id += num_saved
            remaining -= num_saved
            batch_index += 1

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
    except Exception:
        # Kit's fast shutdown can terminate before Python prints the traceback.
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
