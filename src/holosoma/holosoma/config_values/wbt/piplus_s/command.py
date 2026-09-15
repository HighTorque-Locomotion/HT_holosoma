"""PiPlus-S whole-body tracking command preset."""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig

motion_config = MotionConfig(
    motion_file="holosoma/data/motions/piplus_s/whole_body_tracking/climb_23_z_scale_1.0_piplus_holosoma_fps50.npz",
    body_names_to_track=[
        "base_link", "torso_link", "l_shoulder_pitch_link", "r_shoulder_pitch_link",
        "l_shoulder_roll_link", "r_shoulder_roll_link", "l_upper_arm_link", "r_upper_arm_link",
        "l_elbow_link", "r_elbow_link", "l_hip_roll_link", "r_hip_roll_link",
        "l_hip_pitch_link", "r_hip_pitch_link", "l_calf_link", "r_calf_link",
        "l_ankle_roll_link", "r_ankle_roll_link", "l_wrist_link", "r_wrist_link",
    ],
    body_name_ref=["torso_link"],
    use_adaptive_timesteps_sampler=True,
    # Keep enough starts spread over the whole clip to learn the initial
    # approach/first step instead of collapsing onto a frequently-failed late
    # bin.  Eval still starts at frame zero deterministically.
    adaptive_uniform_ratio=0.35,
    start_at_timestep_zero_prob=0.25,
)

piplus_s_wbt_command = CommandManagerCfg(
    setup_terms={"motion_command": CommandTermCfg(
        func="holosoma.managers.command.terms.wbt:MotionCommand", params={"motion_config": motion_config}
    )},
    reset_terms={"motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt:MotionCommand")},
    step_terms={"motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt:MotionCommand")},
)
