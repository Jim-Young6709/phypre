"""
--------------------------------------------
Pure Codex generated scripts with no review.
--------------------------------------------

Run the Franka table environment in Isaac Lab.

"""

from __future__ import annotations

import argparse
import math
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
            quat_apply,
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

                eef_body_id = env.robot.find_bodies(env.cfg.eef_body_name)[0][0]

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

                def test_gripper_open_close() -> str:
                    joint_id = env.gripper_dof_index
                    is_robotiq = env.robot.joint_names[joint_id] == "finger_joint"
                    lower = env.robot_dof_lower_limits[:, joint_id : joint_id + 1]
                    upper = env.robot_dof_upper_limits[:, joint_id : joint_id + 1]
                    open_target, close_target = (lower, upper) if is_robotiq else (upper, lower)
                    finger_names = (
                        ("left_inner_finger", "right_inner_finger")
                        if is_robotiq
                        else ("panda_leftfinger", "panda_rightfinger")
                    )
                    finger_ids = [env.robot.find_bodies(name)[0][0] for name in finger_names]
                    # Robotiq body origins coincide when open; use contact-face
                    # points from its USD pad meshes, expressed in each body frame.
                    pad_points = torch.tensor(
                        [(0.0, -0.04355, 0.12942), (0.0, 0.04355, 0.12942)]
                        if is_robotiq
                        else [(0.0, 0.0, 0.045), (0.0, 0.0, 0.045)],
                        device=env.device,
                    ).expand(env.num_envs, -1, -1)
                    hold_pose_w = env.robot.data.body_pose_w[:, eef_body_id].clone()
                    joint_range = upper - lower
                    tolerance = 0.05 * joint_range
                    max_steps = math.ceil(
                        4.0 * joint_range.max().item() / (env.cfg.gripper_action_scale * env.dt)
                    ) + 60
                    unit = "rad" if is_robotiq else "m per finger"
                    print(
                        f"[INFO] Gripper {env.robot.joint_names[joint_id]}: "
                        f"open={open_target[0, 0].item():.4f}, "
                        f"closed={close_target[0, 0].item():.4f} {unit}; "
                        f"positive action {'closes' if is_robotiq else 'opens'}",
                        flush=True,
                    )

                    def drive_gripper(label: str, target: torch.Tensor) -> torch.Tensor:
                        for step in range(max_steps):
                            env.step(env.make_eef_w_actions(hold_pose_w, target))
                            error = (env.robot.data.joint_pos[:, joint_id : joint_id + 1] - target).abs()
                            if step >= 30 and torch.all(error < tolerance):
                                break
                        assert torch.all(error < tolerance), (
                            f"gripper {label} did not reach its target after {max_steps} steps; "
                            f"joint errors={error.flatten().tolist()} {unit}"
                        )
                        finger_poses = env.robot.data.body_pose_w[:, finger_ids]
                        pads_w = finger_poses[:, :, :3] + quat_apply(
                            finger_poses[:, :, 3:7], pad_points
                        )
                        gap = torch.linalg.vector_norm(pads_w[:, 1] - pads_w[:, 0], dim=-1)
                        print(f"[INFO] Gripper {label}: pad gaps={gap.tolist()} m", flush=True)
                        return gap

                    open_gap = drive_gripper("open", open_target)
                    closed_gap = drive_gripper("close", close_target)
                    reopened_gap = drive_gripper("reopen", open_target)
                    assert torch.all(open_gap > 0.05), f"gripper did not open: {open_gap.tolist()} m"
                    assert torch.all(closed_gap < 0.02), f"gripper did not close: {closed_gap.tolist()} m"
                    assert torch.all((reopened_gap - open_gap).abs() < 0.005), (
                        f"gripper did not reopen to its original gap: {reopened_gap.tolist()} m"
                    )
                    return f"open/close/reopen verified; positive action {'closes' if is_robotiq else 'opens'}"

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
                    max_orientation_error = quat_error_magnitude(
                        env.robot.data.body_pose_w[:, eef_body_id, 3:7],
                        eef_target_w[:, 3:7],
                    ).max().item()
                    assert max_orientation_error < 0.04, (
                        f"maximum end-effector orientation error was {max_orientation_error:.4f} rad"
                    )
                    return f"maximum errors {max_error:.4f} m and {max_orientation_error:.4f} rad"

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
                run_test("gripper open/close", test_gripper_open_close)
                run_test("CuRobo IK", test_curobo_ik)
                run_test("differential IK", test_differential_ik)

                actions = torch.zeros(
                    (env.num_envs, env.num_action_joints), device=env.device
                )
                while not args.headless and simulation_app.is_running():
                    # env.step(actions)
                    env.sim.render()
            finally:
                env.close()

        run()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
