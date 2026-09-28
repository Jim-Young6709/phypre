"""Tabletop pushing with an EEF-only actor and privileged PPO critic."""

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .franka_table_env_cfg import FrankaTableEnvCfg


@configclass
class FrankaPushEnvCfg(FrankaTableEnvCfg):
    use_eef_control = True
    require_grasps = False
    enable_recording_camera = False
    video_interval = 12000  # policy steps between clips
    video_length = 0  # zero uses the full episode duration
    video_envs = 4
    video_every = 2  # capture at 30 fps with the default 60 Hz controller
    video_dir: str | None = None  # defaults to the training run's videos/push directory
    observation_space = 23
    state_space = 51
    scene = InteractiveSceneCfg(num_envs=1024, env_spacing=2.0, replicate_physics=False)

    episode_length_s = 10.0
    randomize_initial_progress = False  # enabled by the training launcher
    joint_reset_noise = 0.0
    eef_position_action_scale = 0.25  # m/s at unit action
    eef_rotation_action_scale = 1.0  # rad/s at unit action
    tcp_offset = (
        0.0,
        0.0,
        0.155,
    )  # fingertip midpoint relative to panda_hand, matching datagen

    # Coordinates relative to each environment origin; bounds apply to the whole object.
    workspace_x = (0.30, 0.75)
    workspace_y = (-0.25, 0.25)
    table_margin = 0.01
    yaw_range = (-3.14159265, 3.14159265)
    settle_steps = 120  # once at startup to cache each asset's resting pose

    position_tolerance = 0.02
    orientation_tolerance = 0.15
    linear_speed_tolerance = 0.03
    angular_speed_tolerance = 0.15
    success_hold_steps = 10

    reach_reward_scale = 1.0
    position_reward_scale = 3.0
    orientation_reward_scale = 1.0
    action_rate_penalty_scale = 0.01

    def __post_init__(self):
        super().__post_init__()
        # Keep the training-only PhysX reservation and recording buffers modest.
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**20
        self.recording_camera.width = 640
        self.recording_camera.height = 480
