"""HT identified motor actuator for the IsaacSim backend.

The pure torque--speed equation lives in :mod:`ht_motor_model` so it can be
tested/imported without starting IsaacSim (and without the ``pxr`` modules).
The IsaacLab wrapper is defined when the simulator dependencies are available.
"""

from __future__ import annotations

from .ht_motor_model import clip_ht_effort

try:  # IsaacLab imports require the SimulationApp/pxr runtime in some installs.
    import torch
    from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.types import ArticulationActions
except (ImportError, ModuleNotFoundError):  # pragma: no cover - exercised in no-sim jobs
    HTMotor = None  # type: ignore[assignment,misc]
    HTMotorCfg = None  # type: ignore[assignment,misc]
    HTMotorCfg_5036 = None  # type: ignore[assignment,misc]
    HTMotorCfg_4438 = None  # type: ignore[assignment,misc]
    HTMotorCfg_5047 = None  # type: ignore[assignment,misc]
    HTMotorCfg_5031 = None  # type: ignore[assignment,misc]
    HTMotor40VCfg_3536 = None  # type: ignore[assignment,misc]
    HTMotor40VCfg_4438 = None  # type: ignore[assignment,misc]
    HTMotor40VCfg_5036 = None  # type: ignore[assignment,misc]
else:

    class HTMotor(DelayedPDActuator):
        """Delayed PD actuator with the HT identified torque-speed law."""

        cfg: "HTMotorCfg"

        def __init__(self, cfg: "HTMotorCfg", *args, **kwargs):
            super().__init__(cfg, *args, **kwargs)
            # DelayedPDActuator invokes _clip_effort after delaying q/v/effort
            # targets.  The HT curve uses the current measured joint velocity,
            # not a delayed velocity.
            self._joint_vel = torch.zeros_like(self.computed_effort)
            self._curve_param_a = float(cfg.curve_param_a)
            self._curve_param_b = float(cfg.curve_param_b)
            self._curve_param_c = float(cfg.curve_param_c)
            self._max_torque = float(cfg.max_torque)
            self._max_velocity = float(cfg.max_velocity)
            self._use_torque_speed_curve = bool(cfg.use_torque_speed_curve)
            # Keep the attribute names exposed by HT_lab for diagnostics and
            # downstream code that introspects an actuator instance.
            self._curve_a = self._curve_param_a
            self._curve_b = self._curve_param_b
            self._curve_c = self._curve_param_c
            self._saturation_effort = self._max_torque

        def compute(
            self,
            control_action: ArticulationActions,
            joint_pos: torch.Tensor,
            joint_vel: torch.Tensor,
        ) -> ArticulationActions:
            self._joint_vel[:] = joint_vel
            # Parent implementation: DelayBuffer(q_des, v_des, effort) -> PD
            # computation -> this class's _clip_effort.
            return super().compute(control_action, joint_pos, joint_vel)

        def reset(self, env_ids) -> None:
            """Reset delay buffers and the cached measured velocity."""
            super().reset(env_ids)
            if env_ids is None:
                self._joint_vel.zero_()
            else:
                self._joint_vel[env_ids] = 0.0

        def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
            if self._use_torque_speed_curve:
                clipped = clip_ht_effort(
                    effort,
                    self._joint_vel,
                    curve_param_a=self._curve_param_a,
                    curve_param_b=self._curve_param_b,
                    curve_param_c=self._curve_param_c,
                    max_torque=self._max_torque,
                    max_velocity=self._max_velocity,
                )
            else:
                # The PiPlus head is an ideal 3536 actuator in HT_lab and does
                # not use the nonlinear curve; retain its box torque cap.
                clipped = torch.clamp(effort, min=-self._max_torque, max=self._max_torque)

            # Match HT_lab semantics: the identified ``max_torque``/curve is
            # the actuator-model cap.  ``effort_limit_sim`` remains an
            # independent PhysX solver limit and is not applied a second time.
            return clipped


    @configclass
    class HTMotorCfg(DelayedPDActuatorCfg):
        """Configuration for :class:`HTMotor`."""

        class_type: type = HTMotor

        curve_param_a: float = -0.0141
        curve_param_b: float = -0.0709
        curve_param_c: float = 6.2756
        max_torque: float = 20.0
        max_velocity: float = 6.0
        use_torque_speed_curve: bool = True
        # Kept as a compatibility alias for HT_lab's original config.  The
        # implementation uses ``max_torque`` as the identified stall limit;
        # callers that still populate ``saturation_effort`` can do so without
        # their config being rejected by IsaacLab's config parser.
        saturation_effort: float = 20.0


    @configclass
    class HTMotorCfg_5036(HTMotorCfg):
        """Identified 0W HT5036 actuator used by PiPlus-S legs."""

        curve_param_a = -0.006667
        curve_param_b = -0.113990
        curve_param_c = 7.732552
        max_torque = 23.7
        max_velocity = 7.95
        saturation_effort = 23.7


    @configclass
    class HTMotorCfg_4438(HTMotorCfg):
        """Identified 0W HT4438 actuator used by PiPlus-S arms."""

        curve_param_a = -0.128416
        curve_param_b = -0.699618
        curve_param_c = 19.833274
        max_torque = 10.0
        max_velocity = 20.0
        saturation_effort = 10.0


    @configclass
    class HTMotorCfg_5047(HTMotorCfg):
        """Identified HT5047 actuator (provided for HT_lab config compatibility)."""

        curve_param_a = -0.0141
        curve_param_b = -0.0709
        curve_param_c = 6.2756
        max_torque = 18.7
        max_velocity = 6.28
        saturation_effort = 18.7


    @configclass
    class HTMotorCfg_5031(HTMotorCfg):
        """Identified HT5031 actuator (provided for HT_lab config compatibility)."""

        curve_param_a = -0.0141
        curve_param_b = -0.0709
        curve_param_c = 6.2756
        max_torque = 20.0
        max_velocity = 6.0
        saturation_effort = 20.0


    @configclass
    class HTMotor40VCfg_3536(HTMotorCfg):
        """Identified 40 V HT3536 actuator from the HT_lab compatibility set."""

        curve_param_a = 2.860840068
        curve_param_b = -22.221680648
        curve_param_c = 41.753409187
        max_torque = 3.3
        max_velocity = 37.18
        saturation_effort = 3.3


    @configclass
    class HTMotor40VCfg_4438(HTMotorCfg):
        """Identified 40 V HT4438 actuator from the HT_lab compatibility set."""

        curve_param_a = -0.184120895
        curve_param_b = -0.637858724
        curve_param_c = 24.600869510
        max_torque = 10.2
        max_velocity = 24.6
        saturation_effort = 10.2


    @configclass
    class HTMotor40VCfg_5036(HTMotorCfg):
        """Identified 40 V HT5036 actuator from the HT_lab compatibility set."""

        curve_param_a = -0.021735007
        curve_param_b = -0.030980508
        curve_param_c = 14.063931256
        max_torque = 22.3
        max_velocity = 14.45
        saturation_effort = 22.3


__all__ = [
    "HTMotor",
    "HTMotorCfg",
    "HTMotorCfg_5036",
    "HTMotorCfg_4438",
    "HTMotorCfg_5047",
    "HTMotorCfg_5031",
    "HTMotor40VCfg_3536",
    "HTMotor40VCfg_4438",
    "HTMotor40VCfg_5036",
    "clip_ht_effort",
]
