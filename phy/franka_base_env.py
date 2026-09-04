"""Minimal Franka DirectRLEnv base for Isaac Lab tasks."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

try:
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_ROBOTIQ_GRIPPER_CFG
except ImportError:  # pragma: no cover - older Isaac Lab exports this at package root.
    from isaaclab_assets import FRANKA_PANDA_CFG, FRANKA_ROBOTIQ_GRIPPER_CFG  # type: ignore


def make_franka_robot_cfg(use_robotiq_gripper: bool = False) -> ArticulationCfg:
    """Build a Franka Panda config with DEXTRAH-style physics and arm gains."""
    asset_cfg = FRANKA_ROBOTIQ_GRIPPER_CFG if use_robotiq_gripper else FRANKA_PANDA_CFG
    gripper_joint_expr = "finger_joint" if use_robotiq_gripper else "panda_finger_joint.*"
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=asset_cfg.spawn.usd_path,
            variants=dict(asset_cfg.spawn.variants) if asset_cfg.spawn.variants else None,
            activate_contact_sensors=False, # this doesn't turn off collsion, just the contact sensor data
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0), # wxyz quaternion
            joint_pos=dict(asset_cfg.init_state.joint_pos),
        ),
        actuators={
            "panda_actuators": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-7]", gripper_joint_expr],
                effort_limit_sim={
                    "panda_joint[1-7]": 300.0, # 100
                    gripper_joint_expr: 200.0, # 1650
                },
                stiffness={
                    "panda_joint[1-4]": 300.0,
                    "panda_joint5": 100.0,
                    "panda_joint6": 50.0,
                    "panda_joint7": 25.0,
                    gripper_joint_expr: 2e3,
                },
                damping={
                    "panda_joint[1-4]": 45.0,
                    "panda_joint5": 20.0,
                    "panda_joint6": 15.0,
                    "panda_joint7": 15.0,
                    gripper_joint_expr: 1e2,
                },
            ),
        },
        soft_joint_pos_limit_factor=asset_cfg.soft_joint_pos_limit_factor,
    )


@configclass
class FrankaBaseEnvCfg(DirectRLEnvCfg):
    """Base configuration shared by Franka Isaac Lab direct environments."""

    sim_dt = 1 / 120.0 # PhysX runs at 120Hz
    decimation = 2 # every env/policy step spans two physics steps, so policy runs at 60Hz
    episode_length_s = 10.0 # this gives 10x60Hz = 600 policy control steps per episode
    num_sim_steps_to_render = 2 # every 2 physics steps, render a frame, so render runs at 60Hz which is consistent with control frequency

    action_space = 8 # 7 DoF arm + 1 scalar gripper action
    observation_space = 24
    state_space = 0 # critic specific state space, for privileged information

    # Select the Franka asset variant with a Robotiq 2F-85 gripper. Both
    # variants expose one scalar gripper action, so action_space remains 8.
    use_robotiq_gripper = False

    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=num_sim_steps_to_render,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_patch_count=4 * 5 * 2**15,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=2.0,
        replicate_physics=False, # set to False allow loading different assets per env
    )

    robot_cfg: ArticulationCfg = make_franka_robot_cfg(use_robotiq_gripper=use_robotiq_gripper)
    arm_joint_names = [
        "panda_joint2",
        "panda_joint1",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    gripper_joint_names = ["finger_joint"] if use_robotiq_gripper else [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]

    action_scale = 7.5
    dof_velocity_scale = 0.1 # rescale joint velocity so the variable is on the same order of magnitude as the normalized joint position [-1, 1]
    joint_reset_noise = 0.125

    ground_prim_path = "/World/ground"
    light_prim_path = "/World/Light"
    light_intensity = 1000.0
    light_color = (0.75, 0.75, 0.75)


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

    def _joint_indices(self, joint_names: Sequence[str]) -> list[int]:
        """Return articulation indices in the requested joint-name order."""
        joint_indices = []
        for joint_name in joint_names:
            try:
                joint_indices.append(self.robot.joint_names.index(joint_name))
            except ValueError as exc:
                raise ValueError(f"Joint {joint_name!r} not found in Franka joints: {self.robot.joint_names}") from exc
        return joint_indices

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path=self.cfg.ground_prim_path, cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=True)
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

        joint_pos = self.robot.data.default_joint_pos[env_ids] + sample_uniform(
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
