"""PiPlus-S whole-body tracking termination terms."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

from .command import motion_config

_TRACKED_BODIES = list(motion_config.body_names_to_track)

piplus_s_wbt_termination = TerminationManagerCfg(terms={
    "timeout": TerminationTermCfg(
        func="holosoma.managers.termination.terms.common:timeout_exceeded", is_timeout=True
    ),
    "bad_tracking": TerminationTermCfg(
        func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
        params={
            "bad_ref_pos_threshold": 0.20,
            "bad_ref_ori_threshold": 0.4,
            "bad_motion_body_pos_threshold": 0.15,
            "body_names_to_track": _TRACKED_BODIES,
            "bad_motion_body_pos_body_names": ["l_ankle_roll_link", "r_ankle_roll_link"],
            "bad_object_pos_threshold": 0.25,
            "bad_object_ori_threshold": 0.8,
        },
    ),
    "base_torso_contact": TerminationTermCfg(
        func="holosoma.managers.termination.terms.wbt:BodyContact",
        params={"body_names": ["base_link", "torso_link"], "force_threshold": 2.0},
    ),
})
