"""Configuration for the base Franka environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import (
    FRANKA_PANDA_CFG,
    FRANKA_ROBOTIQ_GRIPPER_CFG,
)


def make_franka_robot_cfg(use_robotiq_gripper: bool = False) -> ArticulationCfg:
    """Build a Franka Panda configuration with DEXTRAH-style gains."""
    asset_cfg = FRANKA_ROBOTIQ_GRIPPER_CFG if use_robotiq_gripper else FRANKA_PANDA_CFG
    gripper_joint_expr = (
        "finger_joint" if use_robotiq_gripper else "panda_finger_joint.*"
    )
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=asset_cfg.spawn.usd_path,
            variants=dict(asset_cfg.spawn.variants)
            if asset_cfg.spawn.variants
            else None,
            activate_contact_sensors=False,
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
                    "panda_joint6": 100.0,
                    "panda_joint7": 100.0,
                    gripper_joint_expr: 2e3,
                },
                damping={
                    "panda_joint[1-4]": 45.0,
                    "panda_joint5": 20.0,
                    "panda_joint6": 20.0,
                    "panda_joint7": 20.0,
                    gripper_joint_expr: 1e2,
                },
            ),
        },
        soft_joint_pos_limit_factor=asset_cfg.soft_joint_pos_limit_factor,
    )

@configclass
class FrankaBaseEnvCfg(DirectRLEnvCfg):
    """Base configuration shared by Franka Isaac Lab environments."""

    sim_dt = 1 / 120.0 # PhysX runs at 120Hz
    decimation = 2 # every env/policy step spans two physics steps, so policy runs at 60Hz
    episode_length_s = 10.0 # this gives 10x60Hz = 600 policy control steps per episode
    num_sim_steps_to_render = 2 # every 2 physics steps, render a frame, so render runs at 60Hz which is consistent with control frequency

    # EEF control uses local-frame delta XYZ, delta axis-angle, and one gripper action.
    use_eef_control = False
    eef_body_name = "panda_hand"
    eef_position_action_scale = 0.6
    eef_rotation_action_scale = 0.6
    eef_ik_damping = 0.1
    eef_ik_nullspace_gain = 0.05

    franka_joint_action_scale = 0.6
    gripper_action_scale =0.1
    dof_velocity_scale = 0.1 # rescale joint velocity so the variable is on the same order of magnitude as the normalized joint position [-1, 1]
    joint_reset_noise = 0.125

    action_space = 7 if use_eef_control else 8  # joint control: 7 arm joints + 1 scalar gripper action ; eef_control: 3 delta XYZ + 3 delta axis-angle + 1 scalar gripper action
    observation_space = 23 if use_eef_control else 24  # joint state, joint velocity, and previous action
    state_space = 0 # critic specific state space, for privileged information

    # Select the Franka asset variant with a Robotiq 2F-85 gripper.
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
        num_envs=4,
        env_spacing=2.0,
        replicate_physics=False, # set to False allow loading different assets per env
    )

    robot_cfg: ArticulationCfg = make_franka_robot_cfg(
        use_robotiq_gripper=use_robotiq_gripper
    )
    arm_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    canonical_arm_joint_pos = (0.0, 0.0, 0.0, -2.35619, 0.0, 2.35619, 0.78540)
    gripper_joint_names = (
        ["finger_joint"]
        if use_robotiq_gripper
        else ["panda_finger_joint1", "panda_finger_joint2"]
    )

    ground_prim_path = "/World/ground"
    light_prim_path = "/World/Light"
    light_intensity = 1000.0
    light_color = (0.75, 0.75, 0.75)
