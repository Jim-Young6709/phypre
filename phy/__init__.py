"""Physics pretraining environments and utilities."""

import gymnasium as gym

# env registration
gym.register(
    id="Phy-Franka-Table-Direct-v0",
    entry_point="phy.franka_table:FrankaTableEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "phy.cfg.franka_table_env_cfg:FrankaTableEnvCfg"},
)

# datagen registration
gym.register(
    id="Phy-Franka-Table-Datagen-v0",
    entry_point="phy.franka_table:FrankaTableEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "phy.cfg.franka_table_datagen_cfg:FrankaTableDatagenCfg"
        ),
    },
)
