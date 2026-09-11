"""Minimal Franka DirectRLEnv base for Isaac Lab tasks."""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_from_angle_axis,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
)

from phy.cfg.franka_base_env_cfg import FrankaBaseEnvCfg
from phy.utils.eef_ctrl import compute_dof_pos_delta


class FrankaBaseEnv(DirectRLEnv):
    """Base Franka env with joint or differential-IK end-effector control."""

    cfg: FrankaBaseEnvCfg

    def __init__(self, cfg: FrankaBaseEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.use_eef_control:
            cfg.action_space = 7
            cfg.observation_space = 2 * (len(cfg.arm_joint_names) + 1) + 7

        super().__init__(cfg, render_mode, **kwargs) # will call setup scene

        self.dt = self.cfg.sim.dt * self.cfg.decimation # effective control timestep
        self.num_robot_dofs = self.robot.num_joints
        self.arm_dof_indices = self._joint_indices(self.cfg.arm_joint_names)
        self.gripper_dof_indices = self._joint_indices(self.cfg.gripper_joint_names)
        self.gripper_dof_index = self.gripper_dof_indices[0]
        self.canonical_arm_joint_pos = torch.tensor(
            self.cfg.canonical_arm_joint_pos, dtype=torch.float32, device=self.device
        ) # dim (7,)
        self.ik_regularization_config = self.canonical_arm_joint_pos.expand(
            self.num_envs, -1
        ).clone()
        self.logical_dof_indices = self.arm_dof_indices + [self.gripper_dof_index]
        self.actuated_dof_indices = self.arm_dof_indices + self.gripper_dof_indices

        self.num_arm_actions = len(self.arm_dof_indices)
        self.num_gripper_actions = 1
        self.num_action_joints = (
            6 if self.cfg.use_eef_control else self.num_arm_actions
        ) + self.num_gripper_actions
        if isinstance(self.cfg.action_space, int) and self.cfg.action_space != self.num_action_joints:
            raise ValueError(
                f"action_space={self.cfg.action_space} must match {self.num_action_joints} control actions."
            )

        joint_pos_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.robot_dof_lower_limits = joint_pos_limits[..., 0]
        self.robot_dof_upper_limits = joint_pos_limits[..., 1]

        if self.cfg.use_eef_control:
            eef_body_ids, _ = self.robot.find_bodies(self.cfg.eef_body_name)
            if not eef_body_ids:
                raise ValueError(
                    f"Body {self.cfg.eef_body_name!r} not found: {self.robot.body_names}"
                )
            self._eef_body_id = eef_body_ids[0]
            self._eef_jacobi_idx = (
                self._eef_body_id - 1 if self.robot.is_fixed_base else self._eef_body_id
            )
            jacobian_dof_offset = 0 if self.robot.is_fixed_base else 6
            self._eef_jacobian_dof_indices = [
                index + jacobian_dof_offset for index in self.arm_dof_indices
            ]

        self.actions = torch.zeros((self.num_envs, self.num_action_joints), device=self.device) # TODO: check if this is normalized actions & if this is delta actions
        self.robot_dof_targets = self.robot.data.default_joint_pos.clone()
        self.curobo_ik_solver = IKSolver(
            IKSolverConfig.load_from_robot_config(
                "franka.yml",
                WorldConfig(),
                tensor_args=TensorDeviceType(device=torch.device(self.device)),
                num_seeds=32,
                collision_checker_type=CollisionCheckerType.PRIMITIVE,
                collision_cache={"obb": 100},
            )
        )

    def _joint_indices(self, joint_names: Sequence[str]) -> list[int]:
        """Return articulation indices in the requested joint-name order."""
        joint_indices = []
        for joint_name in joint_names:
            try:
                joint_indices.append(self.robot.joint_names.index(joint_name))
            except ValueError as exc:
                raise ValueError(f"Joint {joint_name!r} not found in Franka joints: {self.robot.joint_names}") from exc
        return joint_indices

    def set_robot_joint_state(
        self,
        joint_state: torch.Tensor,
        joint_vel: torch.Tensor | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Set the arm and gripper joint state for selected environments.

        Args:
            joint_state: Joint positions shaped ``[B, num_robot_dofs]`` in the
                articulation's joint order.
            joint_vel: Optional joint velocities with the same shape as
                ``joint_state``. Velocities default to zero.
            env_ids: The ``B`` environment indices to update. Defaults to every
                environment.
        """
        if joint_vel is None:
            joint_vel = torch.zeros_like(joint_state)
        self.robot.write_joint_state_to_sim(joint_state, joint_vel, env_ids=env_ids)

    def franka_ik(
        self,
        eef_pose_w: torch.Tensor,
        null_space_target: torch.Tensor | None = None,
        obstacles: WorldConfig | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve collision-aware IK for a batch of end-effector poses.

        Args:
            eef_pose_w: World-frame poses shaped ``[B, 7]`` and ordered as
                ``[x, y, z, qw, qx, qy, qz]``.
            null_space_target: Arm configurations shaped ``[B, 7]`` and ordered
                according to ``cfg.arm_joint_names``.
            obstacles: Optional shared robot-base-frame world containing up to 100
                cuboids. A provided world replaces the current world; ``None`` clears it.

                Example::

                    from curobo.geom.types import Cuboid, WorldConfig
                    obstacles = WorldConfig(cuboid=[Cuboid(name="table", dims=[0.8, 0.8, 0.05],
                        pose=[0.55, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0])])

        Returns:
            Arm targets shaped ``[B, 7]`` in ``cfg.arm_joint_names`` order and the
            CuRobo success mask for each batch element.
        """
        batch_size = eef_pose_w.shape[0]
        eef_pose_w = eef_pose_w.to(device=self.device, dtype=torch.float32)
        if null_space_target is None:
            null_space_target = self.ik_regularization_config[:batch_size]
        else:
            null_space_target = null_space_target.to(device=self.device, dtype=torch.float32)
        root_pose_w = self.robot.data.root_pose_w[:batch_size]
        eef_pos_b, eef_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            eef_pose_w[:, :3],
            eef_pose_w[:, 3:7],
        )

        null_space_target = JointState.from_position(
            null_space_target, joint_names=self.cfg.arm_joint_names
        ).get_ordered_joint_state(self.curobo_ik_solver.joint_names).position

        # ik solver need to have a fixed batch size (set to num_envs here), because by default use_cuda_graph=True and is necessary to boost ik speed
        if batch_size < self.num_envs:
            padding = self.num_envs - batch_size
            eef_pos_b = torch.cat((eef_pos_b, eef_pos_b[:1].expand(padding, -1)))
            eef_quat_b = torch.cat((eef_quat_b, eef_quat_b[:1].expand(padding, -1)))
            null_space_target = torch.cat(
                (null_space_target, null_space_target[:1].expand(padding, -1))
            )

        # Note under current implementation the obstacle config is not batched, this assume the same set of obstacles for every envs
        # TODO: have a proper batched collision free IK
        if obstacles is None or not obstacles.cuboid:
            self.curobo_ik_solver.world_coll_checker.clear_cache()
        else:
            self.curobo_ik_solver.update_world(obstacles)

        goal = Pose(position=eef_pos_b, quaternion=eef_quat_b)
        result = self.curobo_ik_solver.solve_batch(
            goal_pose=goal, retract_config=null_space_target
        )
        arm_targets = JointState.from_position(
            result.solution[:batch_size, 0], joint_names=self.curobo_ik_solver.joint_names
        ).get_ordered_joint_state(self.cfg.arm_joint_names).position
        return arm_targets, result.success[:batch_size, 0]

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path=self.cfg.ground_prim_path, cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=True) # each clone will be an independent copy, under current setting only robot is copied across all envs
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=self.cfg.light_intensity, color=self.cfg.light_color)
        light_cfg.func(self.cfg.light_prim_path, light_cfg)

        self._setup_task_scene()

    def _setup_task_scene(self) -> None:
        """Hook for child environments to add task-specific assets after env cloning."""

    def make_eef_w_actions(
        self, target_pose_w: torch.Tensor, gripper_width: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert a world-frame target pose to scaled local XYZ/axis-angle and gripper deltas.
        So the executed actions through step will be exactly the same amount of delta in world frame.

        But returned actions are clamped by ``_pre_physics_step`` during ``step``. so the actual executed delta in world frame may be smaller than the commanded delta.
        """
        current_pose_w = self.robot.data.body_pose_w[:, self._eef_body_id]
        position_delta_local, rotation_delta_local = subtract_frame_transforms(
            current_pose_w[:, :3],
            current_pose_w[:, 3:7],
            target_pose_w[:, :3],
            target_pose_w[:, 3:7],
        )
        current_width = self.robot_dof_targets[:, self.gripper_dof_index, None]
        return torch.cat(
            (
                position_delta_local / (self.cfg.eef_position_action_scale * self.dt),
                axis_angle_from_quat(rotation_delta_local)
                / (self.cfg.eef_rotation_action_scale * self.dt),
                (gripper_width - current_width) / (self.cfg.gripper_action_scale * self.dt),
            ),
            dim=-1,
        )

    def _compute_eef_arm_targets(self, eef_actions: torch.Tensor) -> torch.Tensor:
        """eef actions are interpreted as in eef local frame, and converted to arm joint targets by differential IK."""
        eef_pose_w = self.robot.data.body_pose_w[:, self._eef_body_id]
        position_delta_local = (
            eef_actions[:, :3] * self.cfg.eef_position_action_scale * self.dt
        )
        rotation_delta_local = (
            eef_actions[:, 3:6] * self.cfg.eef_rotation_action_scale * self.dt
        )
        target_eef_pos = eef_pose_w[:, :3] + quat_apply(
            eef_pose_w[:, 3:7], position_delta_local
        )
        angle = torch.linalg.vector_norm(rotation_delta_local, dim=-1)
        axis = rotation_delta_local / angle.unsqueeze(-1).clamp_min(1.0e-8)
        rotation_delta_quat_local = quat_from_angle_axis(angle, axis)
        target_eef_quat = quat_mul(
            eef_pose_w[:, 3:7], rotation_delta_quat_local
        )
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self._eef_jacobi_idx, :, self._eef_jacobian_dof_indices
        ]
        joint_pos = self.robot.data.joint_pos[:, self.arm_dof_indices]
        delta_joint_pos = compute_dof_pos_delta(
            arm_dof_pos=joint_pos,
            current_eef_pos=eef_pose_w[:, :3],
            current_eef_quat=eef_pose_w[:, 3:7],
            jacobian=jacobian,
            ctrl_target_eef_pos=target_eef_pos,
            ctrl_target_eef_quat=target_eef_quat,
            ik_nullspace_target=self.ik_regularization_config,
            ik_nullspace_gain=self.cfg.eef_ik_nullspace_gain,
            damping=self.cfg.eef_ik_damping,
        )
        return joint_pos + delta_joint_pos

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Convert normalized actions into arm and gripper joint-position targets.

        In joint mode, the first seven actions are normalized joint deltas.
        In EEF mode, the first six actions are local-frame XYZ
        and axis-angle deltas for ``panda_hand``, converted to arm targets by
        differential IK.
        The final action updates the gripper target in either mode.
        All actions and resulting joint targets are clamped to their limits.
        """
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # compute arm joint targets
        if self.cfg.use_eef_control:
            arm_targets = self._compute_eef_arm_targets(self.actions[:, :6])
        else:
            joint_pos = self.robot.data.joint_pos[:, self.arm_dof_indices]
            joint_range = (
                self.robot_dof_upper_limits[:, self.arm_dof_indices]
                - self.robot_dof_lower_limits[:, self.arm_dof_indices]
            )
            arm_targets = joint_pos + (
                self.actions[:, : self.num_arm_actions]
                * self.cfg.franka_joint_action_scale
                * self.dt
                * joint_range
            )

        # compute gripper joint targets
        gripper_targets = self.robot_dof_targets[
            :, self.gripper_dof_index : self.gripper_dof_index + 1
        ]
        gripper_targets = gripper_targets + (
            self.actions[:, -1:]
            * self.cfg.gripper_action_scale
            * self.dt
        )
        gripper_targets = torch.clamp(
            gripper_targets,
            self.robot_dof_lower_limits[
                :, self.gripper_dof_index : self.gripper_dof_index + 1
            ],
            self.robot_dof_upper_limits[
                :, self.gripper_dof_index : self.gripper_dof_index + 1
            ],
        )

        self.robot_dof_targets[:, self.arm_dof_indices] = torch.clamp(
            arm_targets,
            self.robot_dof_lower_limits[:, self.arm_dof_indices],
            self.robot_dof_upper_limits[:, self.arm_dof_indices],
        )
        self.robot_dof_targets[:, self.gripper_dof_indices] = gripper_targets.expand(
            -1, len(self.gripper_dof_indices)
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(
            self.robot_dof_targets[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices,
        )

    def _get_observations(self) -> dict:
        dof_pos = self.robot.data.joint_pos[:, self.logical_dof_indices]
        dof_pos_normalized = (
            2.0
            * (dof_pos - self.robot_dof_lower_limits[:, self.logical_dof_indices])
            / (
                self.robot_dof_upper_limits[:, self.logical_dof_indices]
                - self.robot_dof_lower_limits[:, self.logical_dof_indices]
            )
            - 1.0
        )
        dof_vel = self.robot.data.joint_vel[:, self.logical_dof_indices]
        obs = torch.cat((dof_pos_normalized, dof_vel * self.cfg.dof_velocity_scale, self.actions), dim=-1)
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self.arm_dof_indices] = self.canonical_arm_joint_pos
        joint_pos += sample_uniform(
            -self.cfg.joint_reset_noise,
            self.cfg.joint_reset_noise,
            (len(env_ids), self.num_robot_dofs),
            self.device,
        )
        joint_pos = torch.clamp(
            joint_pos,
            self.robot_dof_lower_limits[env_ids],
            self.robot_dof_upper_limits[env_ids],
        )
        gripper_pos = joint_pos[:, self.gripper_dof_index : self.gripper_dof_index + 1]
        joint_pos[:, self.gripper_dof_indices] = gripper_pos.expand(-1, len(self.gripper_dof_indices))
        joint_vel = torch.zeros_like(joint_pos)

        self.actions[env_ids] = 0.0
        self.robot_dof_targets[env_ids] = joint_pos

        self.robot.set_joint_position_target(
            joint_pos[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices,
            env_ids=env_ids,
        )
        self.robot.set_joint_velocity_target(
            joint_vel[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices,
            env_ids=env_ids,
        )
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
