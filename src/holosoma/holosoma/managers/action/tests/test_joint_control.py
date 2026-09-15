from types import SimpleNamespace

import torch
from holosoma.config_types.action import ActionTermCfg
from holosoma.managers.action.terms.joint_control import JointPositionActionTerm


class _NativeSimulator:
    uses_native_pd_actuators = True

    def __init__(self):
        self.dof_pos = torch.tensor([[0.4, -0.3], [0.2, 0.1]])
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.hard_dof_pos_limits = torch.tensor([[-0.5, 0.5], [-0.25, 0.25]])
        self.simulator_config = SimpleNamespace(sim=SimpleNamespace(control_decimation_steps=2))
        self.targets = []

    def apply_position_targets_at_dof(self, targets, feedforward_efforts=None):
        self.targets.append(targets.clone())

    def update_native_actuator_gains(self, kp_scale, kd_scale):
        pass

    def reset_actuators(self, env_ids=None):
        pass


def _make_term():
    simulator = _NativeSimulator()
    control = SimpleNamespace(
        control_type="P",
        stiffness={"joint": 10.0},
        damping={"joint": 1.0},
        integral={},
        action_scale=1.0,
        action_scales_by_effort_limit_over_p_gain=False,
        clip_actions=True,
        action_clip_value=100.0,
        clip_torques=True,
        clip_position_targets_to_joint_limits=False,
    )
    robot_config = SimpleNamespace(
        control=control,
        init_state=SimpleNamespace(default_joint_angles={"joint_a": 0.0, "joint_b": 0.0}),
        dof_effort_limit_list=[20.0, 20.0],
    )
    env = SimpleNamespace(
        num_envs=2,
        num_dof=2,
        device="cpu",
        dof_names=["joint_a", "joint_b"],
        simulator=simulator,
        robot_config=robot_config,
        default_dof_pos=torch.tensor([[-0.2, 0.3], [-0.2, 0.3]]),
        torque_limits=torch.tensor([20.0, 20.0]),
        log_dict={},
        _randomize_ctrl_delay=False,
        _pending_torque_rfi=(False, 0.0),
        randomization_manager=None,
    )
    term = JointPositionActionTerm(ActionTermCfg(func="unused:Unused"), env)
    term.setup()
    return term, env


def test_native_reset_pose_hold_covers_one_control_frame_only():
    term, env = _make_term()
    reset_pose = env.simulator.dof_pos[0].clone()

    term.hold_reset_pose_for_next_control_frame(torch.tensor([0]))
    term.process_actions(torch.zeros(2, 2))
    term.apply_actions()

    # The target is a snapshot of the reset pose, not a target that follows the
    # robot as it moves during the bootstrap control frame.
    env.simulator.dof_pos[0] += 0.1
    term.apply_actions()

    assert torch.equal(env.simulator.targets[0][0], reset_pose)
    assert torch.equal(env.simulator.targets[1][0], reset_pose)
    assert torch.equal(env.simulator.targets[0][1], env.default_dof_pos[1])
    assert not torch.any(term._reset_pose_hold_mask)
    assert torch.count_nonzero(term.raw_actions) == 0

    # The next policy/control frame uses its ordinary action-derived target.
    term.process_actions(torch.zeros(2, 2))
    term.apply_actions()
    assert torch.equal(env.simulator.targets[2], env.default_dof_pos)


def test_position_targets_can_be_clamped_to_hard_joint_limits():
    term, env = _make_term()
    env.robot_config.control.clip_position_targets_to_joint_limits = True

    term.process_actions(torch.tensor([[2.0, -2.0], [2.0, -2.0]]))
    term.apply_actions()

    expected = torch.tensor([[0.5, -0.25], [0.5, -0.25]])
    assert torch.equal(term.position_targets, expected)
    assert torch.equal(env.simulator.targets[-1], expected)
    # The diagnostic/nominal PD torque must use the same clamped targets that
    # are sent to the native actuator.
    assert torch.allclose(term.torques, 10.0 * (expected - env.simulator.dof_pos))
