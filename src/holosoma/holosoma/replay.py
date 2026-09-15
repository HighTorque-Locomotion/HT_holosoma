from __future__ import annotations

import time

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.eval_utils import (
    init_sim_imports,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app


def replay(tyro_config: ExperimentConfig):
    simulation_app = init_sim_imports(tyro_config)

    import torch

    from holosoma.utils.common import seeding

    seeding(42, torch_deterministic=False)

    env_target = tyro_config.env_class
    tyro_env_config = get_tyro_env_config(tyro_config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = get_class(env_target)(tyro_env_config, device=device)

    motion_command = env.command_manager.get_state("motion_command")
    frame_dt = 1.0 / float(motion_command.motion.fps)
    next_frame_time = time.perf_counter()
    done = False
    while not done:
        # Motion replay is kinematic: ``step_visualize_motion`` writes the
        # reference root/joint state, runs forward kinematics, and renders it.
        # Advancing physics here first made every displayed frame alternate
        # between a gravity/contact/actuator-integrated pose and the exact next
        # reference pose, which appeared as high-frequency twitching.
        done = env.step_visualize_motion(None)  # type: ignore[attr-defined]
        next_frame_time += frame_dt
        now = time.perf_counter()
        remaining = next_frame_time - now
        if remaining > 0.0:
            time.sleep(remaining)
        elif remaining < -frame_dt:
            # Rendering/debugging can pause for longer than a frame.  Drop the
            # stale wall-clock deadline without skipping any motion frames.
            next_frame_time = now

    close_simulation_app(simulation_app)


def main() -> None:
    from holosoma.config_values.experiment import get_annotated_experiment_config
    from holosoma.utils.config_registry import parse_config

    # Pass the factory uncalled so parse_config builds it after plugins load.
    tyro_cfg = parse_config(get_annotated_experiment_config)
    replay(tyro_cfg)


if __name__ == "__main__":
    main()
