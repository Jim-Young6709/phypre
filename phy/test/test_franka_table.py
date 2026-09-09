"""
--------------------------------------------
Pure Codex generated scripts with no review.
--------------------------------------------

Run the Franka table environment in Isaac Lab.

"""

from __future__ import annotations

import argparse
import sys
import time

TASK_NAME = "Phy-Franka-Table-Direct-v0"


def countdown(env, test_name: str) -> None:
    """Print a three-second countdown while keeping the viewer responsive."""
    for seconds in range(3, -1, -1):
        print(f"[TEST] Testing {test_name} in {seconds} seconds", flush=True)
        if seconds > 0:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                env.sim.render()
                time.sleep(1.0 / 60.0)


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *hydra_args]
    simulation_app = AppLauncher(args).app

    try:
        import torch
        from curobo.geom.types import Cuboid, WorldConfig
        from isaaclab.utils.math import (
            axis_angle_from_quat,
            quat_error_magnitude,
            subtract_frame_transforms,
        )
        from isaaclab_tasks.utils.hydra import hydra_task_config

        from phy.cfg.franka_table_env_cfg import FrankaTableEnvCfg
        from phy.franka_table import FrankaTableEnv

        @hydra_task_config(TASK_NAME, None)
        def run(env_cfg: FrankaTableEnvCfg, _agent_cfg: None) -> None:
            env_cfg.sim.device = args.device
            env_cfg.use_eef_control = True
            env_cfg.action_space = 7
            env_cfg.observation_space = 23
            env_cfg.joint_reset_noise = 0.0
            env_cfg.episode_length_s = 1000.0 # long enough so env reset almost never happens during the test
            env_cfg.object_xy_offset = (0.5, 0.0)
            env = FrankaTableEnv(env_cfg)
            try:
                env.sim.set_camera_view(eye=[2.2, -2.2, 1.6], target=[0.55, 0.0, 0.45])
                print(f"[INFO] Loaded {len(env.selected_assets)} env(s).")

                eef_body_id = env.robot.find_bodies("panda_hand")[0][0]

                def run_test(test_name: str, test_fn) -> None:
                    countdown(env, test_name)
                    try:
                        result = test_fn()
                    except Exception:
                        print(f"[RESULT] {test_name}: FAILED", flush=True)
                        raise
                    print(f"[RESULT] {test_name}: PASSED - {result}", flush=True)

                def test_env_reset() -> str:
                    env.actions.fill_(1.0)
                    env.episode_length_buf.fill_(1)
                    env.reset()
                    assert torch.all(env.actions == 0.0)
                    assert torch.all(env.episode_length_buf == 0)
                    return "actions and episode lengths were reset"

                def test_curobo_ik() -> str:
                    eef_target_w = env.robot.data.body_pose_w[:, eef_body_id].clone()
                    eef_target_w[:, 2] += 0.3
                    obstacles = WorldConfig(
                        cuboid=[
                            Cuboid(
                                name="table",
                                dims=list(env.cfg.table_size),
                                pose=[*env.cfg.table_center, 1.0, 0.0, 0.0, 0.0],
                            )
                        ]
                    )
                    arm_targets, ik_success = env.franka_ik(
                        eef_target_w,
                        env.canonical_arm_joint_pos.expand(env.num_envs, -1),
                        obstacles,
                    )
                    assert arm_targets.shape == (env.num_envs, env.num_arm_actions)
                    assert torch.all(ik_success), (
                        "IK failed for env IDs "
                        f"{torch.nonzero(~ik_success, as_tuple=False).flatten().tolist()}."
                    )

                    joint_pos = env.robot.data.joint_pos.clone()
                    joint_pos[:, env.arm_dof_indices] = arm_targets
                    joint_vel = torch.zeros_like(joint_pos)
                    env.robot_dof_targets[:, env.arm_dof_indices] = arm_targets
                    env.robot.set_joint_position_target(
                        joint_pos[:, env.actuated_dof_indices],
                        joint_ids=env.actuated_dof_indices,
                    )
                    env.robot.write_joint_state_to_sim(joint_pos, joint_vel)
                    env.scene.write_data_to_sim()
                    env.sim.forward()
                    env.sim.render()

                    position_error = torch.linalg.vector_norm(
                        env.robot.data.body_pose_w[:, eef_body_id, :3]
                        - eef_target_w[:, :3],
                        dim=-1,
                    )
                    max_error = position_error.max().item()
                    assert max_error < 0.02, (
                        f"maximum end-effector position error was {max_error:.4f} m"
                    )
                    return f"maximum position error {max_error:.4f} m"

                def test_differential_ik() -> str:
                    target_pose_w = env.robot.data.body_pose_w[
                        :, eef_body_id
                    ].clone()
                    target_pose_w[:, 2] -= 0.3
                    target_pose_w[:, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.device)

                    actions = torch.zeros(
                        (env.num_envs, env.num_action_joints), device=env.device
                    )
                    max_steps = 500
                    for _ in range(max_steps):
                        current_pose_w = env.robot.data.body_pose_w[:, eef_body_id]
                        position_delta_local, rotation_delta_local = (
                            subtract_frame_transforms(
                                current_pose_w[:, :3],
                                current_pose_w[:, 3:7],
                                target_pose_w[:, :3],
                                target_pose_w[:, 3:7],
                            )
                        )
                        position_error = torch.linalg.vector_norm(
                            position_delta_local, dim=-1
                        )
                        orientation_error = quat_error_magnitude(
                            current_pose_w[:, 3:7], target_pose_w[:, 3:7]
                        )
                        if torch.all(
                            (position_error < 0.01) & (orientation_error < 0.02)
                        ):
                            break
                        actions.zero_()
                        actions[:, :3] = position_delta_local / (
                            env.cfg.eef_position_action_scale * env.dt
                        )
                        actions[:, 3:6] = axis_angle_from_quat(
                            rotation_delta_local
                        ) / (env.cfg.eef_rotation_action_scale * env.dt)
                        env.step(actions)

                    achieved_pose_w = env.robot.data.body_pose_w[:, eef_body_id]
                    position_error = torch.linalg.vector_norm(
                        achieved_pose_w[:, :3] - target_pose_w[:, :3], dim=-1
                    )
                    orientation_error = quat_error_magnitude(
                        achieved_pose_w[:, 3:7], target_pose_w[:, 3:7]
                    )
                    max_position_error = position_error.max().item()
                    max_orientation_error = orientation_error.max().item()
                    assert max_position_error < 0.02, (
                        "maximum end-effector position error was "
                        f"{max_position_error:.4f} m"
                    )
                    assert max_orientation_error < 0.04, (
                        "maximum end-effector orientation error was "
                        f"{max_orientation_error:.4f} rad"
                    )
                    return (
                        "moved down 0.3 m with maximum errors "
                        f"{max_position_error:.4f} m and "
                        f"{max_orientation_error:.4f} rad"
                    )

                run_test("environment reset", test_env_reset)
                run_test("CuRobo IK", test_curobo_ik)
                run_test("differential IK", test_differential_ik)

                actions = torch.zeros(
                    (env.num_envs, env.num_action_joints), device=env.device
                )
                while simulation_app.is_running():
                    # env.step(actions)
                    env.sim.render()
            finally:
                env.close()

        run()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
