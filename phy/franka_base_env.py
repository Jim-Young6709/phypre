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
from isaaclab.utils.math import sample_uniform, subtract_frame_transforms

from phy.cfg.franka_base_env_cfg import FrankaBaseEnvCfg


class FrankaBaseEnv(DirectRLEnv):
    """Base Franka env with only robot control, reset, and placeholder RL signals."""

    cfg: FrankaBaseEnvCfg

    def __init__(self, cfg: FrankaBaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs) # will call setup scene

        self.dt = self.cfg.sim.dt * self.cfg.decimation # effective control timestep
        self.num_robot_dofs = self.robot.num_joints
        self.arm_dof_indices = self._joint_indices(self.cfg.arm_joint_names)
        self.gripper_dof_indices = self._joint_indices(self.cfg.gripper_joint_names)
        self.gripper_dof_index = self.gripper_dof_indices[0]
        self.canonical_arm_joint_pos = torch.tensor(
            self.cfg.canonical_arm_joint_pos, dtype=torch.float32, device=self.device
        ) # dim (7,)
        self.logical_dof_indices = self.arm_dof_indices + [self.gripper_dof_index]
        self.actuated_dof_indices = self.arm_dof_indices + self.gripper_dof_indices

        self.num_arm_actions = len(self.arm_dof_indices)
        self.num_gripper_actions = 1
        self.num_action_joints = self.num_arm_actions + self.num_gripper_actions
        if isinstance(self.cfg.action_space, int) and self.cfg.action_space != self.num_action_joints:
            raise ValueError(
                f"action_space={self.cfg.action_space} must match {self.num_action_joints} control actions."
            )

        joint_pos_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.robot_dof_lower_limits = joint_pos_limits[..., 0]
        self.robot_dof_upper_limits = joint_pos_limits[..., 1]

        # arm speed scale set to 1.0, gripper speed scale set to 0.1
        self.robot_dof_speed_scales = torch.ones(self.num_action_joints, device=self.device)
        self.robot_dof_speed_scales[-1] = 0.1

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
            null_space_target = self.canonical_arm_joint_pos.expand(batch_size, -1)
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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Actions are interpreted as normalized joint target-velocity commands, which is dequivalent to delta joint position in this context"""
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets[:, self.logical_dof_indices]
        targets = targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        targets = torch.clamp(
            targets,
            self.robot_dof_lower_limits[:, self.logical_dof_indices],
            self.robot_dof_upper_limits[:, self.logical_dof_indices],
        )
        self.robot_dof_targets[:, self.arm_dof_indices] = targets[:, : self.num_arm_actions]
        self.robot_dof_targets[:, self.gripper_dof_indices] = targets[:, -1:].expand(
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
