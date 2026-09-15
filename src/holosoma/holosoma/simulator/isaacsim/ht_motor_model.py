"""Dependency-light mathematical model for the identified HT motors."""

from __future__ import annotations

import torch


def clip_ht_effort(
    effort: torch.Tensor,
    joint_vel: torch.Tensor,
    *,
    curve_param_a: float,
    curve_param_b: float,
    curve_param_c: float,
    max_torque: float,
    max_velocity: float,
) -> torch.Tensor:
    """Clip efforts using the HT four-quadrant torque-speed law.

    In the motoring quadrant the maximum effort is obtained by solving the
    identified quadratic speed curve.  In the braking quadrant the full stall
    torque remains available, even above the no-load speed.  This is the same
    convention used by ``HT_lab.actuators.HT_motor.HTMotor``.
    """

    effort, joint_vel = torch.broadcast_tensors(effort, joint_vel)
    dtype = effort.dtype
    device = effort.device

    curve_a = torch.as_tensor(curve_param_a, dtype=dtype, device=device)
    curve_b = torch.as_tensor(curve_param_b, dtype=dtype, device=device)
    curve_c = torch.as_tensor(curve_param_c, dtype=dtype, device=device)
    torque_cap = torch.as_tensor(max_torque, dtype=dtype, device=device)
    velocity_cap = torch.as_tensor(max_velocity, dtype=dtype, device=device)

    abs_velocity = torch.abs(joint_vel)
    abs_effort = torch.abs(effort)
    is_motoring = effort * joint_vel >= 0.0

    # Rearrange ``|omega| = a*T^2 + b*T + c`` into
    # ``(-a)*T^2 + (-b)*T + (|omega|-c) = 0``.  This form also handles the
    # positive quadratic coefficient used by the 40 V 3536 identification.
    a = -curve_a
    b = -curve_b
    c = abs_velocity - curve_c
    discriminant = torch.clamp(b * b - 4.0 * a * c, min=0.0)

    eps = torch.finfo(dtype).eps
    denominator = 2.0 * a
    safe_denominator = torch.where(torch.abs(denominator) > eps, denominator, torch.ones_like(denominator))
    quadratic_root = (-b + torch.sqrt(discriminant)) / safe_denominator
    linear_root = torch.where(
        torch.abs(b) > eps,
        # In the degenerate linear case, solve
        # ``abs_velocity = curve_param_b * T + curve_param_c``
        # directly.  Retaining the sign also covers identified curves whose
        # linear coefficient is positive (for example some 40 V variants).
        (abs_velocity - curve_c) / torch.where(torch.abs(curve_b) > eps, curve_b, torch.ones_like(curve_b)),
        torque_cap,
    )
    max_torque_motoring = torch.where(torch.abs(a) > eps, quadratic_root, linear_root)
    max_torque_motoring = torch.clamp(max_torque_motoring, min=0.0, max=torque_cap)
    max_torque_motoring = torch.where(
        abs_velocity > velocity_cap,
        torch.zeros_like(max_torque_motoring),
        max_torque_motoring,
    )

    max_allowed = torch.where(
        is_motoring,
        max_torque_motoring,
        torch.full_like(abs_velocity, torque_cap),
    )
    return torch.clamp(effort, min=-max_allowed, max=max_allowed)


__all__ = ["clip_ht_effort"]
