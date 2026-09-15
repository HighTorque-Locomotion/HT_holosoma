"""Backend-free tests for the HT motor torque--speed model.

The actual :class:`HTMotor` actuator is an IsaacLab actuator and therefore cannot be imported
without the IsaacSim/``pxr`` runtime.  Its clipping law is intentionally exposed as the small
``clip_ht_effort`` helper; keeping these checks on that helper makes the most important motor
contract run in the CPU ``no_sim`` CI job as well as locally.

The reference values below are evaluated from the identification curves used by HT_lab.  They are
not generic DC-motor approximations: changing the curve equation, the motoring/braking quadrant
rules, or the high-speed behavior should make these tests fail.
"""

from __future__ import annotations

import pytest
import torch

from holosoma.simulator.isaacsim.ht_motor import clip_ht_effort


pytestmark = pytest.mark.no_sim


_MOTOR_CURVES = {
    "5036": {
        "curve_param_a": -0.006667,
        "curve_param_b": -0.113990,
        "curve_param_c": 7.732552,
        "max_torque": 23.7,
        "max_velocity": 7.95,
    },
    "4438": {
        "curve_param_a": -0.128416,
        "curve_param_b": -0.699618,
        "curve_param_c": 19.833274,
        "max_torque": 10.0,
        "max_velocity": 20.0,
    },
}


@pytest.mark.parametrize(
    ("motor", "velocities", "expected"),
    [
        (
            "5036",
            [0.0, 1.0, 2.0, 4.0, 6.0, 7.0, 7.732552, 7.95, 8.0],
            [23.7, 23.7, 21.99495218, 16.60942011, 9.69815455, 4.97743610, 0.0, 0.0, 0.0],
        ),
        (
            "4438",
            [0.0, 1.0, 2.0, 4.0, 6.0, 7.0, 10.0, 19.0, 19.833274, 20.0],
            [9.99862107, 9.68881248, 9.37107094, 8.70912718, 8.00642663, 7.63722029, 6.44079035, 1.00547457, 0.0, 0.0],
        ),
    ],
)
def test_motoring_curve_limits(motor: str, velocities: list[float], expected: list[float]) -> None:
    """A positive command at positive speed follows the identified motoring curve."""
    result = clip_ht_effort(
        torch.full((len(velocities),), 100.0),
        torch.tensor(velocities),
        **_MOTOR_CURVES[motor],
    )
    assert torch.allclose(result, torch.tensor(expected), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("motor", ["5036", "4438"])
def test_motoring_is_sign_symmetric(motor: str) -> None:
    """Mirroring both torque and velocity preserves the absolute torque limit."""
    cfg = _MOTOR_CURVES[motor]
    effort = torch.tensor([100.0, -100.0, 7.0, -7.0])
    velocity = torch.tensor([2.0, -2.0, 4.0, -4.0])
    result = clip_ht_effort(effort, velocity, **cfg)
    mirrored = clip_ht_effort(-effort, -velocity, **cfg)
    assert torch.allclose(result, -mirrored, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("motor", ["5036", "4438"])
def test_braking_quadrant_uses_full_stall_limit_even_above_no_load_speed(motor: str) -> None:
    """Opposing torque/speed is braking and is limited only by max torque."""
    cfg = _MOTOR_CURVES[motor]
    effort = torch.tensor([100.0, -100.0, 3.0, -3.0])
    # Include speeds above each motor's no-load speed: braking remains available there.
    velocity = torch.tensor([-4.0, 4.0, -2.0 * cfg["max_velocity"], 2.0 * cfg["max_velocity"]])
    expected = torch.tensor([cfg["max_torque"], -cfg["max_torque"], 3.0, -3.0])
    result = clip_ht_effort(effort, velocity, **cfg)
    assert torch.allclose(result, expected, atol=1e-6, rtol=1e-6)


def test_zero_effort_stays_zero() -> None:
    """The sign restoration must not turn a zero command into a nonzero torque."""
    result = clip_ht_effort(
        torch.zeros(4),
        torch.tensor([-100.0, -1.0, 0.0, 100.0]),
        **_MOTOR_CURVES["5036"],
    )
    assert torch.equal(result, torch.zeros(4))


def test_zero_velocity_is_motoring_for_both_effort_signs() -> None:
    """The HT convention treats the zero-product boundary as motoring, with signed output."""
    effort = torch.tensor([100.0, -100.0])
    result = clip_ht_effort(effort, torch.zeros_like(effort), **_MOTOR_CURVES["5036"])
    assert torch.allclose(result, torch.tensor([23.7, -23.7]), atol=1e-6, rtol=1e-6)


def test_batch_shape_dtype_and_finite_output() -> None:
    """The helper is vectorized and preserves the effort tensor's shape and dtype."""
    effort = torch.tensor([[100.0, -100.0], [1.0, -1.0]], dtype=torch.float64)
    velocity = torch.tensor([[2.0, -2.0], [8.0, -8.0]], dtype=torch.float64)
    result = clip_ht_effort(effort, velocity, **_MOTOR_CURVES["5036"])
    assert result.shape == effort.shape
    assert result.dtype == effort.dtype
    assert torch.isfinite(result).all()
