"""Configuration for Franka table trajectory generation."""

from __future__ import annotations

from isaaclab.utils import configclass

from .franka_table_env_cfg import FrankaTableEnvCfg


@configclass
class FrankaTableDatagenCfg(FrankaTableEnvCfg):
    """Table environment and trajectory-generation settings."""

    seed: int | None = 0
    require_grasps: bool = True

    num_trajectories: int = 4
    grasp_index: int = 0

    pregrasp_offset: float = 0.1
    pregrasp_axis: str = "-z"
    pos_noise_std: float = 0.01
    rot_noise_std: float = 0.03
    init_ik_steps: int = 100
    reach_steps: int = 120
    close_steps: int = 30
    lift_steps: int = 120
    lift_height: float = 0.20
    open_width: float = 0.04 # open width of one finger, the gripper open width is 2x the amount, 0.08 m is the maximum gripper open width
    closed_width: float = 0.0

    output_hdf5: str = "outputs/frankatable/frankatable_datagen.hdf5"
    append: bool = False # set to false will overwrite the existing hdf5 file
    video_dir: str = "outputs/frankatable/videos"
    record_video_envs: int = 4
    video_fps: int = 30
    video_every: int = 2
