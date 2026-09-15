"""PiPlus-S WBT randomization with external pushes disabled."""

from dataclasses import replace

from holosoma.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization


_push_state = g1_29dof_wbt_randomization.setup_terms["push_randomizer_state"]

# Retain G1's material, base-CoM, joint-bias, and actuator randomization
# configuration.  Only external velocity pushes are disabled for the initial
# PiPlus climbing stage.  Keeping the push state registered lets its reset/step
# hooks remain valid while ``apply_pushes`` exits immediately.
piplus_s_wbt_randomization = replace(
    g1_29dof_wbt_randomization,
    setup_terms={
        **g1_29dof_wbt_randomization.setup_terms,
        "push_randomizer_state": replace(
            _push_state,
            params={**_push_state.params, "enabled": False},
        ),
    },
)


__all__ = ["piplus_s_wbt_randomization"]
