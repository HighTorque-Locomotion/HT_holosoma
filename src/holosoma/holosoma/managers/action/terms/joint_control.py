"""Action terms for joint-level control."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from holosoma.managers.action.base import ActionTermBase

if TYPE_CHECKING:
    from holosoma.config_types.action import ActionTermCfg


class JointPositionActionTerm(ActionTermBase):
    """Action term for joint position control with PD controller.

    This term processes raw actions as joint position targets and computes
    torques using a PD controller. Supports:
    - Action scaling
    - Action clipping
    - Action delay (if configured)
    - Torque randomization (if configured)
    - Torque clipping
    """

    def __init__(self, cfg: ActionTermCfg, env: Any):
        """Initialize joint position action term.

        Args:
            cfg: Configuration for this action term
            env: Environment instance (typically a ``BaseTask`` subclass)
        """
        super().__init__(cfg, env)

        # Get action dimension from environment
        self._action_dim = env.num_dof

        # Initialize action buffers
        self._raw_actions = torch.zeros(env.num_envs, self._action_dim, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, self._action_dim, device=env.device)
        self._actions_after_delay = torch.zeros(env.num_envs, self._action_dim, device=env.device)

        # Initialize torque buffer
        self.torques = torch.zeros(env.num_envs, self._action_dim, device=env.device)

        # Sub-step torque history: [num_envs, decimation, num_dof], allocated in setup().
        self._substep_idx: int = 0

        # Cache previous DOF velocities for derivative control
        self._prev_dof_vel = torch.zeros(env.num_envs, env.num_dof, device=env.device)

        # Default actuator scaling (may be overridden by randomization terms)
        self._kp_scale = torch.ones(env.num_envs, self._action_dim, device=env.device)
        self._kd_scale = torch.ones_like(self._kp_scale)
        self._rfi_lim_scale = torch.ones_like(self._kp_scale)
        self._rfi_lim: float = 0.0
        self._randomize_torque_rfi: bool = False

        # PD gains and action scales
        self.p_gains = torch.zeros(self._action_dim, dtype=torch.float, device=env.device)
        self.d_gains = torch.zeros_like(self.p_gains)
        self.i_gains = torch.zeros_like(self.p_gains)
        self.action_scales = torch.zeros_like(self.p_gains)

        self._configure_pd_gains(env)
        self._configure_action_scales(env)

        # IsaacSim can delegate position-control dynamics to a native actuator
        # model (PiPlus-S uses the HT delayed/nonlinear actuator).  Other
        # backends retain the historical Holosoma-side torque computation.
        self._uses_native_pd_actuators = bool(getattr(env.simulator, "uses_native_pd_actuators", False))
        self.position_targets = torch.zeros_like(self._processed_actions)
        self._reset_pose_hold_targets = torch.zeros_like(self.position_targets)
        self._reset_pose_hold_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        # Expose references on the environment for backward compatibility
        env.p_gains = self.p_gains
        env.d_gains = self.d_gains
        env.i_gains = self.i_gains
        env.action_scales = self.action_scales

        # Action delay queue will be initialized in setup() after randomization manager is ready
        self.action_queue: torch.Tensor | None = None

    def setup(self) -> None:
        """Setup action term after all managers are initialized.

        Initialize action delay queue if control delay randomization is enabled.
        This must be called after the randomization manager is set up.
        """
        super().setup()

        # Initialize action delay queue if randomization is enabled
        if getattr(self.env, "_randomize_ctrl_delay", False):
            max_delay = self.env._ctrl_delay_step_range[1]
            self.action_queue = torch.zeros(self.env.num_envs, max_delay + 1, self._action_dim, device=self.env.device)

        # Allocate sub-step torque history buffer
        decimation = self.env.simulator.simulator_config.sim.control_decimation_steps
        self.torques_substep = torch.zeros(self.env.num_envs, decimation, self._action_dim, device=self.env.device)
        self.dof_pos_substep = torch.zeros(self.env.num_envs, decimation, self._action_dim, device=self.env.device)
        self.dof_vel_substep = torch.zeros(self.env.num_envs, decimation, self._action_dim, device=self.env.device)

        # IsaacGym creates randomization buffers before the action manager exists.
        # Once we reach setup(), try attaching any pre-created actuator scales.
        self._attach_actuator_randomizer_scales()
        # Native IsaacSim actuators own the PD computation.  Push the initial
        # (usually all-one) scales into those actuator objects after they have
        # been created by the simulator.
        self._sync_native_actuator_gains()

        enabled, rfi_lim = self.env._pending_torque_rfi
        self.configure_torque_rfi(enabled=enabled, rfi_lim=rfi_lim)
        self.env._pending_torque_rfi = (False, 0.0)

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return self._action_dim

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process raw actions: clip and apply delay if configured.

        Args:
            actions: Raw action tensor [num_envs, action_dim]
        """
        self._substep_idx = 0
        # Store raw actions
        assert self._raw_actions is not None
        self._raw_actions[:] = actions

        # Clip actions
        if self.env.robot_config.control.clip_actions:
            clip_limit = self.env.robot_config.control.action_clip_value
            assert self._processed_actions is not None
            self._processed_actions[:] = torch.clip(actions, -clip_limit, clip_limit)
            # Log clipping fraction
            self.env.log_dict["action_clip_frac"] = (
                self._processed_actions.abs() == clip_limit
            ).sum() / self._processed_actions.numel()
        else:
            assert self._processed_actions is not None
            self._processed_actions[:] = actions
            self.env.log_dict["action_clip_frac"] = torch.tensor(0.0)

        # Apply action delay if configured
        if getattr(self.env, "_randomize_ctrl_delay", False):
            self._apply_action_delay()
        else:
            assert self._processed_actions is not None
            self._actions_after_delay[:] = self._processed_actions

    def _apply_action_delay(self) -> None:
        """Apply action delay based on domain randomization settings."""
        assert self.action_queue is not None, "action_queue must be initialized in setup()"
        assert self._processed_actions is not None

        # Update action queue
        self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
        self.action_queue[:, 0] = self._processed_actions.clone()

        # Apply uniform delay
        self._actions_after_delay[:] = self.action_queue[
            torch.arange(self.env.num_envs), self.env.action_delay_idx
        ].clone()

    def apply_actions(self) -> None:
        """Apply processed actions through the configured control path."""
        if self._uses_native_pd_actuators:
            self._apply_native_position_targets()
            return

        # Compute torques using PD controller
        self.torques[:] = self._compute_torques(self._actions_after_delay)
        # Record sub-step torques/dof_pos/dof_vel
        self.torques_substep[:, self._substep_idx] = self.torques
        self.dof_pos_substep[:, self._substep_idx] = self.env.simulator.dof_pos
        self.dof_vel_substep[:, self._substep_idx] = self.env.simulator.dof_vel
        self._substep_idx += 1
        # Apply torques to simulator
        self.env.simulator.apply_torques_at_dof(self.torques)
        # Cache velocities for next derivative computation
        self._prev_dof_vel.copy_(self.env.simulator.dof_vel)

    def _apply_native_position_targets(self) -> None:
        """Send position set-points to an IsaacSim actuator model.

        The HTMotor actuator then applies its physics-step delay, computes the
        PD effort with the configured gains, and applies the identified
        torque-speed saturation.  This avoids calculating PD twice (once here
        and once inside IsaacLab).
        """

        control_type = self.env.robot_config.control.control_type
        if control_type != "P":
            raise ValueError("Native IsaacSim actuator models currently require P (position) control.")

        # A custom randomization term may resample gains between environment
        # steps.  Synchronize once per control frame (this method is called at
        # every physics substep) before sending the first target.
        if self._substep_idx == 0:
            self._sync_native_actuator_gains()

        assert self._actions_after_delay is not None
        self.position_targets[:] = self._compute_position_targets(self._actions_after_delay)
        # Boolean indexing is a no-op when no environment is being held; avoid
        # a host synchronisation from ``torch.any`` on every physics sub-step.
        self.position_targets[self._reset_pose_hold_mask] = self._reset_pose_hold_targets[
            self._reset_pose_hold_mask
        ]

        # Keep the nominal Holosoma-side torque buffer for diagnostics,
        # termination terms, and trajectory recording.  The actual applied
        # torque is produced by the IsaacLab actuator after this call.
        self.torques[:] = self._compute_torques(self._actions_after_delay, include_rfi=False)
        self.torques_substep[:, self._substep_idx] = self.torques
        self.dof_pos_substep[:, self._substep_idx] = self.env.simulator.dof_pos
        self.dof_vel_substep[:, self._substep_idx] = self.env.simulator.dof_vel
        self._substep_idx += 1

        feedforward = None
        if self._randomize_torque_rfi:
            feedforward = self._compute_rfi_torques()

        apply_targets = getattr(self.env.simulator, "apply_position_targets_at_dof", None)
        if not callable(apply_targets):
            raise RuntimeError(
                "The environment requested native actuator PD, but the simulator does not implement "
                "apply_position_targets_at_dof()."
            )
        apply_targets(self.position_targets, feedforward)
        self._prev_dof_vel.copy_(self.env.simulator.dof_vel)

        # The hold covers every physics sub-step in exactly one control frame.
        # ``process_actions`` resets ``_substep_idx`` at the next frame.
        decimation = self.env.simulator.simulator_config.sim.control_decimation_steps
        if self._substep_idx >= decimation:
            self._reset_pose_hold_mask.zero_()

    def _compute_position_targets(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert policy actions to position set-points and apply optional joint limits."""
        targets = actions * self.action_scales + self.env.default_dof_pos
        if getattr(self.env.robot_config.control, "clip_position_targets_to_joint_limits", False):
            limits = self.env.simulator.hard_dof_pos_limits
            targets = torch.clamp(targets, min=limits[:, 0], max=limits[:, 1])
        return targets

    def _compute_torques(self, actions: torch.Tensor, *, include_rfi: bool = True) -> torch.Tensor:
        """Compute torques from actions using PD controller.

        Args:
            actions: Action tensor [num_envs, action_dim]

        Returns:
            Torque tensor [num_envs, action_dim]
        """
        # Scale actions
        actions_scaled = actions * self.action_scales

        # Compute torques based on control type
        control_type = self.env.robot_config.control.control_type

        if control_type == "P":
            # Position control
            position_targets = self._compute_position_targets(actions)
            torques = (
                self._kp_scale * self.p_gains * (position_targets - self.env.simulator.dof_pos)
                - self._kd_scale * self.d_gains * self.env.simulator.dof_vel
            )
        elif control_type == "V":
            # Velocity control
            torques = (
                self._kp_scale * self.p_gains * (actions_scaled - self.env.simulator.dof_vel)
                - self._kd_scale * self.d_gains * (self.env.simulator.dof_vel - self._prev_dof_vel) / self.env.sim_dt
            )
        elif control_type == "T":
            # Torque control
            torques = actions_scaled
        else:
            raise ValueError(f"Unknown controller type: {control_type}")

        # Apply torque randomization if configured
        if include_rfi and self._randomize_torque_rfi:
            torques = torques + self._compute_rfi_torques()

        # Clip torques if configured
        if self.env.robot_config.control.clip_torques:
            torques = torch.clip(torques, -self.env.torque_limits, self.env.torque_limits)

        return torques

    def _compute_rfi_torques(self) -> torch.Tensor:
        """Draw residual force-injection torques for the native actuator path."""
        return (
            (torch.rand_like(self.torques) * 2.0 - 1.0)
            * self._rfi_lim
            * self._rfi_lim_scale
            * self.env.torque_limits
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset action term state.

        Args:
            env_ids: Environment IDs to reset. If None, reset all.
        """
        super().reset(env_ids)

        # Reset action delay queue if applicable
        if self.env._randomize_ctrl_delay and self.action_queue is not None:
            if env_ids is None:
                self.action_queue.zero_()
            else:
                self.action_queue[env_ids] = 0.0

        # Reset torques
        if env_ids is None:
            self.torques.zero_()
            self._actions_after_delay.zero_()
            self.position_targets.zero_()
            self._reset_pose_hold_targets.zero_()
            self._reset_pose_hold_mask.zero_()
        else:
            self.torques[env_ids] = 0.0
            self._actions_after_delay[env_ids] = 0.0
            self.position_targets[env_ids] = 0.0
            self._reset_pose_hold_targets[env_ids] = 0.0
            self._reset_pose_hold_mask[env_ids] = False

        # Reset cached velocities
        if env_ids is None:
            self._prev_dof_vel.zero_()
        else:
            self._prev_dof_vel[env_ids] = 0.0

        # IsaacLab does not receive Holosoma's manager-level reset callback
        # automatically.  Resetting here clears HTMotor DelayBuffers and draws
        # a fresh per-environment 0..3 physics-step lag.
        if self._uses_native_pd_actuators:
            reset_actuators = getattr(self.env.simulator, "reset_actuators", None)
            if callable(reset_actuators):
                reset_actuators(env_ids)

    def hold_reset_pose_for_next_control_frame(self, env_ids: torch.Tensor | None = None) -> None:
        """Keep reset joint positions during the next native-PD control frame.

        This is used only by WBT's initial zero-action bootstrap step.  Raw and
        processed actions remain zero, so the observation/action history keeps
        its usual reset semantics.
        """
        if not self._uses_native_pd_actuators:
            return

        if env_ids is None:
            self._reset_pose_hold_targets.copy_(self.env.simulator.dof_pos)
            self._reset_pose_hold_mask.fill_(True)
        else:
            self._reset_pose_hold_targets[env_ids] = self.env.simulator.dof_pos[env_ids]
            self._reset_pose_hold_mask[env_ids] = True

    # ------------------------------------------------------------------
    # Hooks for randomization manager

    def attach_actuator_scales(
        self, kp_scale: torch.Tensor, kd_scale: torch.Tensor, rfi_lim_scale: torch.Tensor
    ) -> None:
        """Attach shared actuator scaling tensors provided by the randomization manager."""
        self._kp_scale = kp_scale
        self._kd_scale = kd_scale
        self._rfi_lim_scale = rfi_lim_scale

    def update_pd_scales(self, env_ids: torch.Tensor, kp_values: torch.Tensor, kd_values: torch.Tensor) -> None:
        """Fallback PD-scale update when no shared buffers are registered."""
        self._kp_scale[env_ids] = kp_values
        self._kd_scale[env_ids] = kd_values

    def update_rfi_scales(self, env_ids: torch.Tensor, rfi_values: torch.Tensor) -> None:
        """Fallback RFI-scale update when no shared buffers are registered."""
        self._rfi_lim_scale[env_ids] = rfi_values

    def configure_torque_rfi(self, *, enabled: bool, rfi_lim: float | None = None) -> None:
        """Configure residual force injection behaviour."""
        self._randomize_torque_rfi = enabled
        if rfi_lim is not None:
            self._rfi_lim = float(rfi_lim)

    def get_pd_scale_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return references to the PD gain scale buffers."""
        return self._kp_scale, self._kd_scale

    def get_rfi_scale_tensor(self) -> torch.Tensor:
        """Return reference to the RFI limit scale buffer."""
        return self._rfi_lim_scale

    def get_prev_dof_vel(self) -> torch.Tensor:
        """Return cached previous DOF velocities."""
        return self._prev_dof_vel

    # ------------------------------------------------------------------
    # Internal helpers

    def _attach_actuator_randomizer_scales(self) -> None:
        """Attach shared actuator randomizer buffers if they exist."""
        rand_manager = getattr(self.env, "randomization_manager", None)
        if rand_manager is None:
            return

        get_state = getattr(rand_manager, "get_state", None)
        if not callable(get_state):
            return

        state = get_state("actuator_randomizer_state")
        if state is None:
            return

        self.attach_actuator_scales(state.kp_scale_tensor, state.kd_scale_tensor, state.rfi_lim_scale_tensor)

    def _sync_native_actuator_gains(self) -> None:
        """Synchronize Holosoma gain scales with IsaacLab's native actuators."""
        if not self._uses_native_pd_actuators:
            return
        updater = getattr(self.env.simulator, "update_native_actuator_gains", None)
        if callable(updater):
            updater(self._kp_scale, self._kd_scale)

    def _configure_pd_gains(self, env: Any) -> None:
        control_cfg = env.robot_config.control
        stiffness_cfg = control_cfg.stiffness
        damping_cfg = control_cfg.damping
        integral_cfg = getattr(control_cfg, "integral", {})

        for i, name in enumerate(env.dof_names):
            if name not in env.robot_config.init_state.default_joint_angles:
                raise ValueError(f"Missing default joint angle for DOF '{name}' in robot configuration.")

            matched = False
            for dof_name, stiffness in stiffness_cfg.items():
                if dof_name in name:
                    self.p_gains[i] = stiffness
                    self.d_gains[i] = damping_cfg[dof_name]
                    self.i_gains[i] = integral_cfg.get(dof_name, 0.0)
                    matched = True
            if not matched:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                self.i_gains[i] = 0.0
                if control_cfg.control_type in ["P", "V"]:
                    raise ValueError(
                        f"PD gains for joint '{name}' were not defined. Please specify them in the YAML configuration."
                    )

    def _configure_action_scales(self, env: Any) -> None:
        control_cfg = env.robot_config.control
        if control_cfg.action_scales_by_effort_limit_over_p_gain:
            dof_effort_limit_list = env.robot_config.dof_effort_limit_list
            for i, effort in enumerate(dof_effort_limit_list):
                stiffness = self.p_gains[i]
                if stiffness == 0.0:
                    self.action_scales[i] = 0.0
                else:
                    self.action_scales[i] = control_cfg.action_scale * effort / stiffness
        else:
            self.action_scales[:] = control_cfg.action_scale
