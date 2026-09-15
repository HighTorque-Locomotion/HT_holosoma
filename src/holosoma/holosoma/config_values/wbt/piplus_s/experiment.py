"""PiPlus-S whole-body tracking experiment."""

from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, TrainingConfig
from holosoma.config_values import action, algo, curriculum, simulator, terrain
from holosoma.config_values.robot import piplus_s

from .command import piplus_s_wbt_command
from .observation import piplus_s_wbt_observation
from .randomization import piplus_s_wbt_randomization
from .reward import piplus_s_wbt_reward
from .termination import piplus_s_wbt_termination

piplus_s_wbt = ExperimentConfig(
    training=TrainingConfig(project="WholeBodyTracking", name="piplus_s_wbt", num_envs=2048),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    # Use the PPO settings tuned for Holosoma's WBT task.  The generic PPO
    # preset has 1e-5 actor/critic learning rates and no empirical observation
    # normalization; that preset left this motion policy close to its default
    # crouch even after 10k updates.
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=30000,
            num_learning_epochs=5,
            entropy_coef=0.005,
            init_noise_std=1.0,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            empirical_normalization=True,
            use_symmetry=False,
            actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.0),
            critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.0),
        ),
    ),
    simulator=simulator.isaacsim,
    robot=piplus_s,
    terrain=terrain.terrain_piplus_climb_23,
    observation=piplus_s_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=piplus_s_wbt_termination,
    randomization=piplus_s_wbt_randomization,
    command=piplus_s_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=piplus_s_wbt_reward,
)
