"""PiPlus-S WBT rewards.

The tracking objective follows the G1 preset, but contact exclusions must use
PiPlus-S link names.  Reusing G1's ``left_*``/``right_*`` regex verbatim makes
the PiPlus ``l_*``/``r_*`` feet undesired contacts and penalizes every normal
stance step.
"""

from dataclasses import replace

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward

_base_undesired_contacts = g1_29dof_wbt_reward.terms["undesired_contacts"]

piplus_s_wbt_reward = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.3},
            weight=1.5,
        ),
        # Link tracking alone leaves serial-chain joint angles underconstrained.
        # These dense terms make the PiPlus policy follow the converted qpos/qvel
        # targets as well as the retained body-link objectives.
        "motion_joint_pos_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_joint_pos_error_exp",
            params={"sigma": 0.5},
            weight=1.0,
        ),
        "motion_joint_vel_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_joint_vel_error_exp",
            params={"sigma": 2.0},
            weight=0.5,
        ),
        "undesired_contacts": replace(
            _base_undesired_contacts,
            params={
                **_base_undesired_contacts.params,
                "undesired_contacts_body_names": (
                    "^(?!l_ankle_roll_link$)(?!r_ankle_roll_link$)"
                    "(?!l_thigh_link$)(?!r_thigh_link$)"
                    "(?!l_calf_link$)(?!r_calf_link$)"
                    "(?!l_wrist_link$)(?!r_wrist_link$).+$"
                ),
            },
        ),
    }
)
