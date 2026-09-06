"""Run the Franka table environment in Isaac Lab."""

from __future__ import annotations

import argparse
import sys

TASK_NAME = "Phy-Franka-Table-Direct-v0"


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=0, help="0 runs until closed.")
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *hydra_args]
    simulation_app = AppLauncher(args).app

    try:
        import torch
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

                actions = torch.zeros(
                    (env.num_envs, env.num_action_joints), device=env.device
                )
                count = 0
                while simulation_app.is_running() and (
                    args.steps <= 0 or count < args.steps
                ):
                    env.step(actions)
                    count += 1
            finally:
                env.close()

        run()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
