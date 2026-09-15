"""Contract tests for the PiPlus-S joint and actuator metadata.

These checks are intentionally backend-free.  IsaacSim resolves its native joint IDs from the
URDF, while Holosoma policies use the canonical order in ``RobotConfig.dof_names``; keeping the
name-level metadata complete prevents a silent left/right mismatch in action or symmetry
augmentation code.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from defusedxml import ElementTree
from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.config_values.robot import piplus_s
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward
from holosoma.config_values.wbt.piplus_s.command import motion_config
from holosoma.config_values.wbt.piplus_s.experiment import piplus_s_wbt
from holosoma.config_values.wbt.piplus_s.reward import piplus_s_wbt_reward
from holosoma.config_values.wbt.piplus_s.termination import piplus_s_wbt_termination
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.managers.command.terms.wbt import AdaptiveTimestepsSampler
from holosoma.managers.termination.manager import TerminationManager
from holosoma.managers.termination.terms.wbt import BodyContact

pytestmark = pytest.mark.no_sim


_DOFS = [
    "head_yaw_joint",
    "l_hip_pitch_joint",
    "l_shoulder_pitch_joint",
    "r_hip_pitch_joint",
    "r_shoulder_pitch_joint",
    "head_pitch_joint",
    "l_hip_roll_joint",
    "l_shoulder_roll_joint",
    "r_hip_roll_joint",
    "r_shoulder_roll_joint",
    "l_thigh_joint",
    "l_upper_arm_joint",
    "r_thigh_joint",
    "r_upper_arm_joint",
    "l_calf_joint",
    "l_elbow_joint",
    "r_calf_joint",
    "r_elbow_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
]

_MIRROR = {
    "head_yaw_joint": "head_yaw_joint",
    "head_pitch_joint": "head_pitch_joint",
    "l_hip_pitch_joint": "r_hip_pitch_joint",
    "r_hip_pitch_joint": "l_hip_pitch_joint",
    "l_hip_roll_joint": "r_hip_roll_joint",
    "r_hip_roll_joint": "l_hip_roll_joint",
    "l_thigh_joint": "r_thigh_joint",
    "r_thigh_joint": "l_thigh_joint",
    "l_calf_joint": "r_calf_joint",
    "r_calf_joint": "l_calf_joint",
    "l_ankle_pitch_joint": "r_ankle_pitch_joint",
    "r_ankle_pitch_joint": "l_ankle_pitch_joint",
    "l_ankle_roll_joint": "r_ankle_roll_joint",
    "r_ankle_roll_joint": "l_ankle_roll_joint",
    "l_shoulder_pitch_joint": "r_shoulder_pitch_joint",
    "r_shoulder_pitch_joint": "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint": "r_shoulder_roll_joint",
    "r_shoulder_roll_joint": "l_shoulder_roll_joint",
    "l_upper_arm_joint": "r_upper_arm_joint",
    "r_upper_arm_joint": "l_upper_arm_joint",
    "l_elbow_joint": "r_elbow_joint",
    "r_elbow_joint": "l_elbow_joint",
}

_SIGN_FLIP = {
    "head_yaw_joint",
    "l_hip_roll_joint",
    "r_hip_roll_joint",
    "l_shoulder_roll_joint",
    "r_shoulder_roll_joint",
    "l_thigh_joint",
    "r_thigh_joint",
    "l_upper_arm_joint",
    "r_upper_arm_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
}


def _urdf_root():
    package_dir = Path(__file__).resolve().parents[2]
    urdf_path = package_dir / "data/robots/piplus_s/urdf/PiPlus_S_12L8A0G2H0W.urdf"
    return ElementTree.parse(urdf_path).getroot()


def test_piplus_canonical_dof_order_is_complete_and_unique() -> None:
    """The policy/action dimension must cover exactly the 22 actuated URDF joints."""
    assert piplus_s.dof_names == _DOFS
    assert piplus_s.actions_dim == len(_DOFS) == 22
    assert len(set(piplus_s.dof_names)) == len(_DOFS)
    assert len(piplus_s.dof_effort_limit_list) == len(_DOFS)
    assert len(piplus_s.dof_armature_list) == len(_DOFS)


def test_piplus_robot_config_matches_training_urdf() -> None:
    """The configured bodies and DOFs must resolve one-for-one in the actual training URDF."""
    root = _urdf_root()
    urdf_links = [link.attrib["name"] for link in root.findall("link")]
    urdf_dofs = [
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib["type"] not in {"fixed", "floating"}
    ]

    assert piplus_s.asset.collapse_fixed_joints is False
    assert piplus_s.num_bodies == len(urdf_links) == 27
    assert piplus_s.body_names == urdf_links
    assert set(piplus_s.dof_names) == set(urdf_dofs)
    assert len(urdf_dofs) == 22


def test_piplus_joint_limits_match_training_urdf() -> None:
    """Position, velocity, and effort limits must follow the active 260908 asset."""
    joints = {joint.attrib["name"]: joint for joint in _urdf_root().findall("joint")}
    for index, name in enumerate(piplus_s.dof_names):
        limit = joints[name].find("limit")
        assert limit is not None
        assert piplus_s.dof_pos_lower_limit_list[index] == pytest.approx(float(limit.attrib["lower"]))
        assert piplus_s.dof_pos_upper_limit_list[index] == pytest.approx(float(limit.attrib["upper"]))
        assert piplus_s.dof_vel_limit_list[index] == pytest.approx(float(limit.attrib["velocity"]))
        assert piplus_s.dof_effort_limit_list[index] == pytest.approx(float(limit.attrib["effort"]))


def test_piplus_leg_joint_chain_matches_expected_links() -> None:
    """Left and right leg actions must address the corresponding URDF child links."""
    root = _urdf_root()
    joints = {
        joint.attrib["name"]: (
            joint.find("parent").attrib["link"],
            joint.find("child").attrib["link"],
        )
        for joint in root.findall("joint")
    }
    expected_suffixes = ["hip_pitch", "hip_roll", "thigh", "calf", "ankle_pitch", "ankle_roll"]

    for side in ("l", "r"):
        expected_parent = "base_link"
        for suffix in expected_suffixes:
            joint_name = f"{side}_{suffix}_joint"
            child_name = f"{side}_{suffix}_link"
            assert joints[joint_name] == (expected_parent, child_name)
            expected_parent = child_name


def test_piplus_tracking_and_termination_links_exist_in_urdf() -> None:
    """Every motion-tracked or failure-checked body must be a retained PiPlus link."""
    links = set(piplus_s.body_names)
    bad_tracking_cfg = piplus_s_wbt_termination.terms["bad_tracking"].params

    assert set(motion_config.body_names_to_track).issubset(links)
    assert set(bad_tracking_cfg["body_names_to_track"]).issubset(links)
    assert bad_tracking_cfg["body_names_to_track"] == motion_config.body_names_to_track
    assert set(bad_tracking_cfg["bad_motion_body_pos_body_names"]).issubset(links)


def test_piplus_filters_only_overlapping_arm_collision_pairs() -> None:
    """Keep global self-collision while suppressing the two authored arm overlaps."""
    assert piplus_s.asset.enable_self_collisions is True
    assert piplus_s.asset.replace_cylinder_with_capsule is True
    assert piplus_s.asset.self_collision_filter_pairs == (
        ("l_upper_arm_link", "l_wrist_link"),
        ("r_upper_arm_link", "r_wrist_link"),
    )


def test_piplus_base_and_torso_contact_terminate() -> None:
    term_cfg = piplus_s_wbt_termination.terms["base_torso_contact"]
    assert term_cfg.params["body_names"] == ["base_link", "torso_link"]
    threshold = term_cfg.params["force_threshold"]

    forces = torch.zeros(3, 2, 3, 3)
    forces[1, 0, 0, 2] = threshold + 0.01  # base contact
    forces[2, 1, 1, 0] = threshold + 1.0  # torso contact retained in history
    env = SimpleNamespace(
        device="cpu",
        simulator=SimpleNamespace(
            body_names=["base_link", "torso_link", "l_ankle_roll_link"],
            contact_forces_history=forces,
        ),
    )
    term = BodyContact(term_cfg, env)
    assert term(env).tolist() == [False, True, True]


def test_termination_manager_retains_each_reason_mask() -> None:
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        logger=None,
        episode_length_buf=torch.tensor([0, 2, 3]),
        max_episode_length=1,
    )
    cfg = TerminationManagerCfg(
        terms={
            "failure": TerminationTermCfg(
                func="holosoma.managers.termination.terms.common:timeout_exceeded"
            ),
            "timeout": TerminationTermCfg(
                func="holosoma.managers.termination.terms.common:timeout_exceeded",
                is_timeout=True,
            ),
        }
    )
    manager = TerminationManager(cfg, env, "cpu")
    terminated, time_outs = manager.check()
    assert terminated.tolist() == [False, True, True]
    assert time_outs.tolist() == [False, True, True]
    assert manager.term_dones["failure"].tolist() == [False, True, True]
    assert manager.term_dones["timeout"].tolist() == [False, True, True]

    manager.reset(torch.tensor([1]))
    assert manager.term_dones["failure"].tolist() == [False, False, True]
    assert manager.term_dones["timeout"].tolist() == [False, False, True]


def test_episode_extras_expose_termination_fractions() -> None:
    task = BaseTask.__new__(BaseTask)
    task.reward_manager = None
    task.termination_manager = SimpleNamespace(
        term_dones={
            "bad_tracking": torch.tensor([False, True, False]),
            "base_torso_contact": torch.tensor([False, False, True]),
        }
    )
    task.extras = {}
    task.time_out_buf = torch.zeros(3, dtype=torch.bool)

    BaseTask._fill_extras(task, torch.tensor([1, 2]))

    assert task.extras["episode"]["termination_bad_tracking_fraction"].tolist() == [1.0, 0.0]
    assert task.extras["episode"]["termination_base_torso_contact_fraction"].tolist() == [0.0, 1.0]


def test_piplus_mirror_and_sign_metadata_cover_every_joint() -> None:
    """Left/right augmentation must be an involution with the HT_lab sign convention."""
    mapping = piplus_s.symmetry_joint_names
    assert mapping is not None
    assert mapping == _MIRROR
    assert set(mapping) == set(_DOFS)
    assert all(mapping[mapping[name]] == name for name in _DOFS)

    signs = piplus_s.flip_sign_joint_names
    assert signs is not None
    assert set(signs) == _SIGN_FLIP
    assert set(signs).issubset(_DOFS)


def test_piplus_arm_metadata_uses_l_r_joint_names() -> None:
    """Optional arm subsets must not leak the ``left_*``/``right_*`` G1 defaults."""
    expected_left = [
        "l_shoulder_pitch_joint",
        "l_shoulder_roll_joint",
        "l_upper_arm_joint",
        "l_elbow_joint",
    ]
    expected_right = [name.replace("l_", "r_", 1) for name in expected_left]
    assert piplus_s.left_arm_dof_names == expected_left
    assert piplus_s.right_arm_dof_names == expected_right
    assert piplus_s.arm_dof_names == expected_left + expected_right


def test_piplus_has_no_inherited_waist_dof_metadata() -> None:
    """PiPlus has a fixed torso joint; G1's three-DOF waist metadata is invalid here."""
    assert piplus_s.waist_dof_names is None
    assert piplus_s.waist_yaw_dof_name is None
    assert piplus_s.waist_roll_dof_name is None
    assert piplus_s.waist_pitch_dof_name is None


@pytest.mark.parametrize(
    ("joint", "expected_effort", "expected_armature"),
    [
        ("head_yaw_joint", 3.7, 0.001976),
        ("l_hip_pitch_joint", 21.0, 0.013212),
        ("l_shoulder_pitch_joint", 10.0, 0.008234),
        ("l_calf_joint", 21.0, 0.013212),
    ],
)
def test_piplus_motor_limits_follow_joint_groups(joint: str, expected_effort: float, expected_armature: float) -> None:
    """Representative head/leg/arm values stay associated with the named joint, not list position."""
    index = piplus_s.dof_names.index(joint)
    assert piplus_s.dof_effort_limit_list[index] == pytest.approx(expected_effort)
    assert piplus_s.dof_armature_list[index] == pytest.approx(expected_armature)


def test_piplus_contact_reward_allows_supporting_limbs() -> None:
    """Feet, wrists, thighs, and calves may contact the climbing terrain."""
    term = piplus_s_wbt_reward.terms["undesired_contacts"]
    pattern = term.params["undesired_contacts_body_names"]
    allowed = {
        "l_ankle_roll_link",
        "r_ankle_roll_link",
        "l_wrist_link",
        "r_wrist_link",
        "l_thigh_link",
        "r_thigh_link",
        "l_calf_link",
        "r_calf_link",
    }

    assert all(re.match(pattern, body_name) is None for body_name in allowed)
    assert re.match(pattern, "base_link") is not None
    assert re.match(pattern, "torso_link") is not None


def test_piplus_contact_reward_does_not_mutate_g1_preset() -> None:
    """The robot-specific override must leave the shared G1 configuration intact."""
    g1_pattern = g1_29dof_wbt_reward.terms["undesired_contacts"].params["undesired_contacts_body_names"]
    piplus_pattern = piplus_s_wbt_reward.terms["undesired_contacts"].params[
        "undesired_contacts_body_names"
    ]

    assert "left_ankle_roll_link" in g1_pattern
    assert "l_ankle_roll_link" in piplus_pattern
    assert piplus_pattern != g1_pattern


def test_piplus_reward_includes_dense_joint_tracking_terms() -> None:
    """PiPlus WBT must constrain qpos/qvel in addition to body-link poses."""
    pos = piplus_s_wbt_reward.terms["motion_joint_pos_error_exp"]
    vel = piplus_s_wbt_reward.terms["motion_joint_vel_error_exp"]
    assert pos.params["sigma"] == pytest.approx(0.5)
    assert pos.weight == pytest.approx(1.0)
    assert vel.params["sigma"] == pytest.approx(2.0)
    assert vel.weight == pytest.approx(0.5)


def test_piplus_uses_tighter_link_orientation_tracking() -> None:
    orientation = piplus_s_wbt_reward.terms["motion_relative_body_orientation_error_exp"]
    assert orientation.params["sigma"] == pytest.approx(0.3)
    assert orientation.weight == pytest.approx(1.5)


def test_piplus_clamps_position_targets_and_uses_staged_torso_threshold() -> None:
    assert piplus_s.control.clip_position_targets_to_joint_limits is True
    bad_tracking = piplus_s_wbt_termination.terms["bad_tracking"]
    assert bad_tracking.params["bad_ref_pos_threshold"] == pytest.approx(0.20)


def test_piplus_uses_wbt_ppo_hyperparameters() -> None:
    """PiPlus tracking must not silently fall back to the slow generic PPO preset."""
    cfg = piplus_s_wbt.algo.config
    assert cfg.actor_learning_rate == pytest.approx(1e-3)
    assert cfg.critic_learning_rate == pytest.approx(1e-3)
    assert cfg.num_learning_epochs == 5
    assert cfg.entropy_coef == pytest.approx(0.005)
    assert cfg.empirical_normalization is True
    assert cfg.actor_optimizer.weight_decay == pytest.approx(0.0)
    assert cfg.critic_optimizer.weight_decay == pytest.approx(0.0)


def test_piplus_disables_only_external_push_randomization() -> None:
    """PiPlus keeps G1 domain randomization while disabling external pushes."""
    from holosoma.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization

    cfg = piplus_s_wbt.randomization
    assert cfg.setup_terms["push_randomizer_state"].params["enabled"] is False
    assert set(cfg.setup_terms) == set(g1_29dof_wbt_randomization.setup_terms)
    assert cfg.reset_terms == g1_29dof_wbt_randomization.reset_terms
    assert cfg.step_terms == g1_29dof_wbt_randomization.step_terms
    for name, term in g1_29dof_wbt_randomization.setup_terms.items():
        if name != "push_randomizer_state":
            assert cfg.setup_terms[name] == term


def test_piplus_motion_sampler_keeps_initial_approach_in_training() -> None:
    """Training must continue seeing frame zero even when late bins fail often."""
    assert motion_config.use_adaptive_timesteps_sampler is True
    assert motion_config.adaptive_uniform_ratio == pytest.approx(0.35)
    assert motion_config.start_at_timestep_zero_prob == pytest.approx(0.25)


def test_adaptive_sampler_uniform_ratio_is_a_real_probability_floor() -> None:
    """A late failure spike must not erase early-motion sampling."""
    sampler = AdaptiveTimestepsSampler(
        motion_time_step_total=561,
        device="cpu",
        env_fps=50,
        adaptive_uniform_ratio=0.35,
    )
    sampler.bin_failed_count[-1] = 1000.0
    probabilities = sampler.sampling_probabilities

    assert torch.isclose(probabilities.sum(), torch.tensor(1.0))
    assert probabilities[0] >= torch.tensor(0.35 / sampler.num_bins)
    assert probabilities.max() <= torch.tensor(1.0 - 0.35 + 0.35 / sampler.num_bins + 1e-6)
