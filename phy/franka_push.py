"""Non-prehensile pushing; all actor geometry and actions use the EEF frame."""

from __future__ import annotations

import torch
from isaaclab.utils.math import (
    create_rotation_matrix_from_view,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_mul,
    quat_unique,
    subtract_frame_transforms,
)
from pxr import Usd, UsdGeom

from phy.cfg.franka_push_env_cfg import FrankaPushEnvCfg
from phy.franka_table import FrankaTableEnv


class FrankaPushEnv(FrankaTableEnv):
    cfg: FrankaPushEnvCfg

    def __init__(self, cfg: FrankaPushEnvCfg, render_mode=None, **kwargs):
        self._video_logger = None
        super().__init__(cfg, render_mode, **kwargs)
        bounds = torch.tensor(
            self._object_bounds, device=self.device, dtype=torch.float32
        )
        self.object_center = bounds.mean(dim=1)
        self.object_size = bounds[:, 1] - bounds[:, 0]
        self.table_top = cfg.table_center[2] + cfg.table_size[2] / 2
        self.goal_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.action_frame_quat = self.robot.data.body_pose_w[
            :, self._eef_body_id, 3:7
        ].clone()
        self.action_rate = torch.zeros(self.num_envs, device=self.device)
        self.success_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.failure = torch.zeros_like(self.success)
        self.episode_reward_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in ("reach", "position", "orientation", "action_rate")
        }
        limits = (
            self.robot_dof_upper_limits
            if cfg.use_robotiq_gripper
            else self.robot_dof_lower_limits
        )
        self.robot.data.default_joint_pos[:, self.gripper_dof_indices] = limits[
            :, self.gripper_dof_indices
        ]
        self._settle_objects()
        if cfg.enable_recording_camera:
            from phy.rl.video import PushVideoLogger

            self._video_logger = PushVideoLogger(self)

    def step(self, actions):
        result = super().step(actions)
        if self._video_logger is not None:
            self._video_logger.step()
        return result

    def reset(self, seed=None, options=None):
        observations, extras = super().reset(seed=seed, options=options)
        if self.cfg.randomize_initial_progress:
            self.episode_length_buf.random_(self.max_episode_length)
            observations = self._get_observations()
        return observations, extras

    def close(self):
        if self._video_logger is not None:
            self._video_logger.close()
        super().close()

    def _configure_gym_env_spaces(self):
        self.cfg.action_space = 6  # the task supplies the controller's gripper action
        super()._configure_gym_env_spaces()

    def _get_assets(self, cfg):
        assets = super()._get_assets(cfg)
        # Fit even when a tall asset tips onto a different face during settling.
        sizes = torch.tensor(
            [
                self.asset_metadata.get(asset.asset_id, {}).get(
                    "bbox_size", [cfg.default_object_height] * 3
                )
                for asset in assets
            ]
        )
        diameter = sizes.norm(dim=-1)
        width = min(
            cfg.workspace_x[1] - cfg.workspace_x[0],
            cfg.workspace_y[1] - cfg.workspace_y[0],
        )
        return [
            asset
            for asset, fits in zip(assets, diameter < width - 2 * cfg.table_margin)
            if fits
        ]

    def _setup_task_scene(self):
        if self.cfg.enable_recording_camera:
            camera_ids = "|".join(
                str(i) for i in range(min(self.cfg.video_envs, self.num_envs))
            )
            self.cfg.recording_camera.prim_path = (
                f"/World/envs/env_({camera_ids})/RecordCamera"
            )
            target = torch.tensor(
                [
                    [
                        *self.cfg.table_center[:2],
                        self.cfg.table_center[2] + self.cfg.table_size[2] / 2,
                    ]
                ]
            )
            eye = target + 1.0
            rotation = create_rotation_matrix_from_view(eye, target, "Z", device="cpu")
            # Author the pose before Fabric initializes the camera transforms.
            self.cfg.recording_camera.offset.pos = tuple(eye[0].tolist())
            self.cfg.recording_camera.offset.rot = tuple(
                quat_from_matrix(rotation)[0].tolist()
            )
            self.cfg.recording_camera.offset.convention = "opengl"
        super()._setup_task_scene()
        if self._recording_camera is not None:
            # Recording cameras have their own indices, separate from the N RL environments.
            self.scene.sensors.pop("recording_camera")
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), ["default", "render", "proxy"]
        )
        self._object_bounds = []
        for env_path in self.scene.env_prim_paths:
            geometry = self.scene.stage.GetPrimAtPath(f"{env_path}/Object/Geometry")
            body = next(iter(geometry.GetChildren()))
            bounds = cache.ComputeUntransformedBound(body).ComputeAlignedRange()
            self._object_bounds.append((tuple(bounds.GetMin()), tuple(bounds.GetMax())))

    def _settle_objects(self):
        """Cache resting poses once; episode resets never step other environments."""
        super()._reset_idx(None)
        quat = self.object_orientation_offset.to(self.device).expand(self.num_envs, -1)
        center = quat_apply(quat, self.object_center)
        half_size = (
            matrix_from_quat(quat).abs() @ (self.object_size / 2).unsqueeze(-1)
        ).squeeze(-1)
        pose = self.object.data.root_pose_w.clone()
        pose[:, :3] = self.scene.env_origins
        pose[:, :2] += (
            torch.tensor(self.cfg.table_center[:2], device=self.device) - center[:, :2]
        )
        pose[:, 2] += (
            self.table_top
            - center[:, 2]
            + half_size[:, 2]
            + self.cfg.object_table_clearance
        )
        pose[:, 3:7] = quat
        self.object.write_root_pose_to_sim(pose)
        self.object.write_root_velocity_to_sim(
            torch.zeros((self.num_envs, 6), device=self.device)
        )
        for _ in range(self.cfg.settle_steps):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
        self.support_quat = self.object.data.root_quat_w.clone()
        self.support_height = (
            self.object.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        )

    def _sample_pose(self, env_ids):
        n = len(env_ids)
        yaw = torch.empty(n, device=self.device).uniform_(*self.cfg.yaw_range)
        zeros = torch.zeros_like(yaw)
        quat = quat_mul(
            quat_from_euler_xyz(zeros, zeros, yaw), self.support_quat[env_ids]
        )
        center = quat_apply(quat, self.object_center[env_ids])
        half_size = (
            matrix_from_quat(quat).abs() @ (self.object_size[env_ids] / 2).unsqueeze(-1)
        ).squeeze(-1)
        lower = torch.tensor(
            [self.cfg.workspace_x[0], self.cfg.workspace_y[0]], device=self.device
        )
        upper = torch.tensor(
            [self.cfg.workspace_x[1], self.cfg.workspace_y[1]], device=self.device
        )
        low = lower + half_size[:, :2] + self.cfg.table_margin
        high = upper - half_size[:, :2] - self.cfg.table_margin
        pos = self.scene.env_origins[env_ids].clone()
        pos[:, :2] += (
            low + torch.rand((n, 2), device=self.device) * (high - low) - center[:, :2]
        )
        pos[:, 2] += self.support_height[env_ids]
        return torch.cat((pos, quat), dim=-1)

    def _pre_physics_step(self, actions):
        eef_quat = self.robot.data.body_quat_w[:, self._eef_body_id]
        previous = self._actions_in_frame(eef_quat)
        close = torch.full_like(
            actions[:, :1], 1.0 if self.cfg.use_robotiq_gripper else -1.0
        )
        super()._pre_physics_step(torch.cat((actions, close), dim=-1))
        self.action_rate = (self.actions[:, :6] - previous).square().sum(dim=-1)
        self.action_frame_quat = eef_quat.clone()

    def _actions_in_frame(self, quat):
        return torch.cat(
            [
                quat_apply_inverse(quat, quat_apply(self.action_frame_quat, part))
                for part in self.actions[:, :6].split(3, dim=-1)
            ],
            dim=-1,
        )

    def _get_observations(self):
        eef = self.robot.data.body_pose_w[:, self._eef_body_id]
        obj = self.object.data.root_pose_w
        object_pos, object_quat = subtract_frame_transforms(
            eef[:, :3], eef[:, 3:7], obj[:, :3], obj[:, 3:7]
        )
        goal_pos, goal_quat = subtract_frame_transforms(
            eef[:, :3], eef[:, 3:7], self.goal_pose[:, :3], self.goal_pose[:, 3:7]
        )
        size_eef = (
            matrix_from_quat(object_quat).abs() @ self.object_size.unsqueeze(-1)
        ).squeeze(-1)
        actor = torch.cat(
            (
                object_pos,
                quat_unique(object_quat),
                goal_pos,
                quat_unique(goal_quat),
                size_eef,
                self._actions_in_frame(eef[:, 3:7]),
            ),
            dim=-1,
        )

        joints = self.arm_dof_indices
        lower, upper = (
            self.robot_dof_lower_limits[:, joints],
            self.robot_dof_upper_limits[:, joints],
        )
        joint_pos = (
            2 * (self.robot.data.joint_pos[:, joints] - lower) / (upper - lower) - 1
        )
        root = self.robot.data.root_pose_w
        eef_pos_b, eef_quat_b = subtract_frame_transforms(
            root[:, :3], root[:, 3:7], eef[:, :3], eef[:, 3:7]
        )
        critic = torch.cat(
            (
                actor,
                joint_pos,
                self.robot.data.joint_vel[:, joints] * self.cfg.dof_velocity_scale,
                eef_pos_b,
                quat_unique(eef_quat_b),
                self.object.data.root_vel_w,
                (1 - self.episode_length_buf / self.max_episode_length).unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": actor, "critic": critic}

    def _get_dones(self):
        obj = self.object.data.root_pose_w
        self.position_error = (obj[:, :3] - self.goal_pose[:, :3]).norm(dim=-1)
        self.orientation_error = quat_error_magnitude(
            obj[:, 3:7], self.goal_pose[:, 3:7]
        )
        reached = (
            (self.position_error < self.cfg.position_tolerance)
            & (self.orientation_error < self.cfg.orientation_tolerance)
            & (
                self.object.data.root_lin_vel_w.norm(dim=-1)
                < self.cfg.linear_speed_tolerance
            )
            & (
                self.object.data.root_ang_vel_w.norm(dim=-1)
                < self.cfg.angular_speed_tolerance
            )
        )
        self.success_steps = torch.where(reached, self.success_steps + 1, 0)
        center = (
            obj[:, :3]
            + quat_apply(obj[:, 3:7], self.object_center)
            - self.scene.env_origins
        )
        table_xy = torch.tensor(self.cfg.table_center[:2], device=self.device)
        table_half = torch.tensor(self.cfg.table_size[:2], device=self.device) / 2
        self.failure = ((center[:, :2] - table_xy).abs() > table_half).any(dim=-1) | (
            center[:, 2] < self.table_top
        )
        self.success = (
            self.success_steps >= self.cfg.success_hold_steps
        ) & ~self.failure
        return (
            self.success | self.failure,
            self.episode_length_buf >= self.max_episode_length - 1,
        )

    def _get_rewards(self):
        self.extras.pop("log", None)
        obj = self.object.data.root_pose_w
        eef = self.robot.data.body_pose_w[:, self._eef_body_id]
        tcp = eef[:, :3] + quat_apply(
            eef[:, 3:7],
            eef.new_tensor(self.cfg.tcp_offset).expand(self.num_envs, -1),
        )
        center = obj[:, :3] + quat_apply(obj[:, 3:7], self.object_center)
        reach_distance = (tcp - center).norm(dim=-1)
        position = torch.exp(-self.position_error / 0.10)
        rewards = {
            "reach": self.cfg.reach_reward_scale * torch.exp(-reach_distance / 0.10),
            "position": self.cfg.position_reward_scale * position,
            "orientation": self.cfg.orientation_reward_scale
            * position
            * torch.exp(-self.orientation_error / 0.5),
            "action_rate": -self.cfg.action_rate_penalty_scale * self.action_rate,
        }
        for name, reward in rewards.items():
            self.episode_reward_sums[name] += reward
        return sum(rewards.values())

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        finished = env_ids[self.episode_length_buf[env_ids] > 0]
        self.extras.pop("log", None)
        if len(finished):
            # Vectors let rl_games average over episodes, not uneven reset batches.
            self.extras["log"] = {
                "success_rate": self.success[finished].float(),
                "failure_rate": self.failure[finished].float(),
                "Reward/total": sum(self.episode_reward_sums.values())[finished],
                **{
                    f"Reward/{name}": reward[finished]
                    for name, reward in self.episode_reward_sums.items()
                },
            }
        for reward in self.episode_reward_sums.values():
            reward[env_ids] = 0
        super()._reset_idx(env_ids)
        self.object.write_root_pose_to_sim(self._sample_pose(env_ids), env_ids=env_ids)
        self.object.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
        )
        self.goal_pose[env_ids] = self._sample_pose(env_ids)
        self.action_rate[env_ids] = 0
        self.success_steps[env_ids] = 0
        self.success[env_ids] = False
        self.failure[env_ids] = False
