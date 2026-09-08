"""
--------------------------------------------
Pure Codex generated scripts with no review.
--------------------------------------------

Run the Franka table environment in Isaac Lab.

"""

from __future__ import annotations

import argparse
import sys

TASK_NAME = "Phy-Franka-Table-Direct-v0"


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
        from isaaclab_tasks.utils.hydra import hydra_task_config

        from phy.cfg.franka_table_env_cfg import FrankaTableEnvCfg
        from phy.franka_table import FrankaTableEnv

        @hydra_task_config(TASK_NAME, None)
        def run(env_cfg: FrankaTableEnvCfg, _agent_cfg: None) -> None:
            env_cfg.sim.device = args.device
            env = FrankaTableEnv(env_cfg)
            try:
                env.sim.set_camera_view(eye=[2.2, -2.2, 1.6], target=[0.55, 0.0, 0.45])
                env.reset()
                print(f"[INFO] Loaded {len(env.selected_assets)} env(s).")

                reset_env_ids = torch.arange(0, env.num_envs, 2, device=env.device)
                unreset_env_ids = torch.arange(1, env.num_envs, 2, device=env.device)
                env.actions.fill_(1.0)
                env.episode_length_buf.fill_(1)
                env._reset_idx(reset_env_ids)
                assert torch.all(env.actions[reset_env_ids] == 0.0)
                assert torch.all(env.episode_length_buf[reset_env_ids] == 0)
                assert torch.all(env.actions[unreset_env_ids] == 1.0)
                assert torch.all(env.episode_length_buf[unreset_env_ids] == 1)
                env.reset()
                print(f"[INFO] Indexed reset passed for env IDs {reset_env_ids.tolist()}.")

                eef_body_id = env.robot.find_bodies("panda_hand")[0][0]
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
                    f"IK failed for env IDs "
                    f"{torch.nonzero(~ik_success, as_tuple=False).flatten().tolist()}."
                )
                env.robot_dof_targets[:, env.arm_dof_indices] = arm_targets
                print("[INFO] Collision-aware IK passed; moving panda_hand 5 cm upward.")

                actions = torch.zeros(
                    (env.num_envs, env.num_action_joints), device=env.device
                )
                while simulation_app.is_running():
                    env.step(actions)
            finally:
                env.close()

        run()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
