"""Configuration types for motion data format."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from holosoma_retargeting.config_types.robot import (
    RobotDefaults,
    _default_robot_defaults,
    _validate_robot_type,
)
from holosoma_retargeting.data_utils.parc_ms import PARC_MS_DEMO_JOINTS as _PARC_MS_DEMO_JOINTS

# Pre-defined constants for each data format
LAFAN_DEMO_JOINTS = [
    "Hips",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
]

SMPLH_DEMO_JOINTS = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
]

MOCAP_DEMO_JOINTS = [
    "Hips",
    "Spine",
    "Spine1",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftFootMod",
    "RightFootMod",
]

SMPLX_DEMO_JOINTS = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
]

# PARC MS uses a compact 15-body humanoid (plus two toe points synthesized by
# ``data_utils.parc_ms``).  Names intentionally follow the source rig so the
# same data can be inspected in human-humanoid-tools and Holosoma.
PARC_MS_DEMO_JOINTS = _PARC_MS_DEMO_JOINTS

# Joint mappings - organized by (data_format, robot_type)
JOINTS_MAPPINGS = {
    ("lafan", "g1"): {
        "Spine1": "pelvis_contour_link",
        "LeftUpLeg": "left_hip_pitch_link",
        "RightUpLeg": "right_hip_pitch_link",
        "LeftLeg": "left_knee_link",
        "RightLeg": "right_knee_link",
        "LeftArm": "left_shoulder_roll_link",
        "RightArm": "right_shoulder_roll_link",
        "LeftForeArm": "left_elbow_link",
        "RightForeArm": "right_elbow_link",
        "LeftFoot": "left_ankle_intermediate_1_link",
        "RightFoot": "right_ankle_intermediate_1_link",
        "LeftToeBase": "left_ankle_roll_sphere_5_link",
        "RightToeBase": "right_ankle_roll_sphere_5_link",
        "LeftHand": "left_rubber_hand_link",
        "RightHand": "right_rubber_hand_link",
    },
    ("lafan", "t1"): {
        "Spine1": "Trunk",
        "LeftUpLeg": "Hip_Pitch_Left",
        "RightUpLeg": "Hip_Pitch_Right",
        "LeftLeg": "Shank_Left",
        "RightLeg": "Shank_Right",
        "LeftArm": "AL1",
        "RightArm": "AR1",
        "LeftForeArm": "left_hand_link",
        "RightForeArm": "right_hand_link",
        "LeftFoot": "Ankle_Cross_Left",
        "RightFoot": "Ankle_Cross_Right",
        "LeftToeBase": "left_foot_sphere_5_link",
        "RightToeBase": "right_foot_sphere_5_link",
        "LeftHand": "left_hand_sphere_link",
        "RightHand": "right_hand_sphere_link",
    },
    ("smplh", "g1"): {
        "Pelvis": "pelvis_contour_link",
        "L_Hip": "left_hip_pitch_link",
        "R_Hip": "right_hip_pitch_link",
        "L_Knee": "left_knee_link",
        "R_Knee": "right_knee_link",
        "L_Shoulder": "left_shoulder_roll_link",
        "R_Shoulder": "right_shoulder_roll_link",
        "L_Elbow": "left_elbow_link",
        "R_Elbow": "right_elbow_link",
        "L_Ankle": "left_ankle_intermediate_1_link",
        "R_Ankle": "right_ankle_intermediate_1_link",
        "L_Toe": "left_ankle_roll_sphere_5_link",
        "R_Toe": "right_ankle_roll_sphere_5_link",
        "L_Wrist": "left_rubber_hand_link",
        "R_Wrist": "right_rubber_hand_link",
    },
    ("smplh", "t1"): {
        "Pelvis": "Trunk",
        "L_Hip": "Hip_Pitch_Left",
        "R_Hip": "Hip_Pitch_Right",
        "L_Knee": "Shank_Left",
        "R_Knee": "Shank_Right",
        "L_Shoulder": "AL1",
        "R_Shoulder": "AR1",
        "L_Elbow": "left_hand_link",
        "R_Elbow": "right_hand_link",
        "L_Ankle": "Ankle_Cross_Left",
        "R_Ankle": "Ankle_Cross_Right",
        "L_Toe": "left_foot_sphere_5_link",
        "R_Toe": "right_foot_sphere_5_link",
        "L_Wrist": "left_hand_sphere_link",
        "R_Wrist": "right_hand_sphere_link",
    },
    ("smplx", "g1"): {
        "Pelvis": "pelvis_contour_link",
        "L_Hip": "left_hip_pitch_link",
        "R_Hip": "right_hip_pitch_link",
        "L_Knee": "left_knee_link",
        "R_Knee": "right_knee_link",
        "L_Shoulder": "left_shoulder_roll_link",
        "R_Shoulder": "right_shoulder_roll_link",
        "L_Elbow": "left_elbow_link",
        "R_Elbow": "right_elbow_link",
        "L_Ankle": "left_ankle_intermediate_1_link",
        "R_Ankle": "right_ankle_intermediate_1_link",
        "L_Foot": "left_ankle_roll_sphere_5_link",
        "R_Foot": "right_ankle_roll_sphere_5_link",
        "L_Wrist": "left_rubber_hand_link",
        "R_Wrist": "right_rubber_hand_link",
    },
    ("mocap", "g1"): {
        "Spine1": "pelvis_contour_link",
        "LeftUpLeg": "left_hip_pitch_link",
        "LeftLeg": "left_knee_link",
        "LeftToeBase": "left_ankle_roll_sphere_5_link",
        "RightUpLeg": "right_hip_pitch_link",
        "RightLeg": "right_knee_link",
        "RightToeBase": "right_ankle_roll_sphere_5_link",
        "LeftArm": "left_shoulder_roll_link",
        "LeftForeArm": "left_elbow_link",
        "LeftHandMiddle3": "left_sphere_hand_link",
        "RightArm": "right_shoulder_roll_link",
        "RightForeArm": "right_elbow_link",
        "RightHandMiddle3": "right_sphere_hand_link",
        "LeftFoot": "left_ankle_intermediate_1_link",
        "RightFoot": "right_ankle_intermediate_1_link",
    },
    ("mocap", "t1"): {
        "Spine1": "Trunk",
        "LeftUpLeg": "Hip_Pitch_Left",
        "LeftLeg": "Shank_Left",
        "LeftToeBase": "left_foot_sphere_5_link",
        "RightUpLeg": "Hip_Pitch_Right",
        "RightLeg": "Shank_Right",
        "RightToeBase": "right_foot_sphere_5_link",
        "LeftArm": "AL1",
        "LeftForeArm": "left_hand_link",
        "LeftHandMiddle3": "left_hand_sphere_link",
        "RightArm": "AR1",
        "RightForeArm": "right_hand_link",
        "RightHandMiddle3": "right_hand_sphere_link",
        "LeftFoot": "Ankle_Cross_Left",
        "RightFoot": "Ankle_Cross_Right",
    },
    # PiPlus S (22 DoF).  The fixed-wrist model still has a hand/palm marker
    # near the end of the wrist mesh.  For the feet, ``LeftFoot``/``RightFoot``
    # must be the ankle-pitch hinge centres (the G1 reference bodies are fixed
    # hinge-centre markers, not points that rotate with ankle pitch).
    ("mocap", "piplus_s"): {
        # Use the fixed torso-center marker rather than the full torso mesh.
        # This is the PiPlus equivalent of G1's pelvis_contour_link marker.
        "Spine1": "torso_center_link",
        "LeftUpLeg": "l_hip_pitch_link",
        "LeftLeg": "l_calf_link",
        "LeftFoot": "l_ankle_intermediate_1_link",
        "LeftToeBase": "l_foot_contact_toe",
        "RightUpLeg": "r_hip_pitch_link",
        "RightLeg": "r_calf_link",
        "RightFoot": "r_ankle_intermediate_1_link",
        "RightToeBase": "r_foot_contact_toe",
        "LeftArm": "l_shoulder_roll_link",
        "LeftForeArm": "l_elbow_link",
        # The source G1 marker is the palm sphere centre.  PiPlus has the
        # corresponding palm sphere at the distal end of the fixed wrist;
        # mapping to the wrist body origin is about 11 cm too proximal.
        "LeftHandMiddle3": "l_hand_center_link",
        "RightArm": "r_shoulder_roll_link",
        "RightForeArm": "r_elbow_link",
        "RightHandMiddle3": "r_hand_center_link",
    },
    # PARC MS (15-body rig + synthetic left_toe/right_toe points).
    ("parc_ms", "g1"): {
        "pelvis": "pelvis_contour_link",
        "torso": "torso_link",
        "right_upper_arm": "right_shoulder_roll_link",
        "right_lower_arm": "right_elbow_link",
        "right_hand": "right_sphere_hand_link",
        "left_upper_arm": "left_shoulder_roll_link",
        "left_lower_arm": "left_elbow_link",
        "left_hand": "left_sphere_hand_link",
        "right_thigh": "right_hip_pitch_link",
        "right_shin": "right_knee_link",
        "right_foot": "right_ankle_intermediate_1_link",
        "right_toe": "right_ankle_roll_sphere_5_link",
        "left_thigh": "left_hip_pitch_link",
        "left_shin": "left_knee_link",
        "left_foot": "left_ankle_intermediate_1_link",
        "left_toe": "left_ankle_roll_sphere_5_link",
    },
    ("parc_ms", "piplus_s"): {
        "pelvis": "base_link",
        # Match the compact PARC torso point to the same fixed marker used by
        # MOCAP.  The marker has no DoF and is visual-only in the MuJoCo XML.
        "torso": "torso_center_link",
        "right_upper_arm": "r_shoulder_roll_link",
        "right_lower_arm": "r_elbow_link",
        "right_hand": "r_hand_center_link",
        "left_upper_arm": "l_shoulder_roll_link",
        "left_lower_arm": "l_elbow_link",
        "left_hand": "l_hand_center_link",
        "right_thigh": "r_hip_pitch_link",
        "right_shin": "r_calf_link",
        "right_foot": "r_ankle_pitch_link",
        "right_toe": "r_foot_contact_toe",
        "left_thigh": "l_hip_pitch_link",
        "left_shin": "l_calf_link",
        "left_foot": "l_ankle_pitch_link",
        "left_toe": "l_foot_contact_toe",
    },
}

# Data format specific constants
TOE_NAMES_BY_FORMAT = {
    "lafan": ["LeftToeBase", "RightToeBase"],
    "smplh": ["L_Toe", "R_Toe"],
    "mocap": ["LeftToeBase", "RightToeBase"],
    "smplx": ["L_Foot", "R_Foot"],
    "parc_ms": ["left_toe", "right_toe"],
}


# Data format specific scaling/preprocessing constants
class FormatConstants(TypedDict, total=False):
    default_scale_factor: float | None
    default_human_height: float | None


DATA_FORMAT_CONSTANTS: dict[str, FormatConstants] = {
    "lafan": {
        "default_scale_factor": 1.27 / 1.7,
    },
    "mocap": {
        "default_human_height": 1.78,
    },
    # PARC's ankle-terminated source rig represents a canonical adult human;
    # use the same 1.70 m assumption as human-humanoid-tools.
    "parc_ms": {
        "default_human_height": 1.70,
    },
}

# Unified registry: Maps format name to demo joints
# This is the SINGLE PLACE to add new formats - just add an entry here!
# No need to update any Literal types - DataFormat is now str with runtime validation
DEMO_JOINTS_REGISTRY: dict[str, list[str]] = {
    "lafan": LAFAN_DEMO_JOINTS,
    "smplh": SMPLH_DEMO_JOINTS,
    "mocap": MOCAP_DEMO_JOINTS,
    "smplx": SMPLX_DEMO_JOINTS,
    "parc_ms": PARC_MS_DEMO_JOINTS,
}

# Type alias for data formats - use str to allow dynamic data formats via DEMO_JOINTS_REGISTRY
# No need to update this when adding new formats - just add to DEMO_JOINTS_REGISTRY above
DataFormat = str


def _validate_data_format(data_format: str) -> None:
    """Validate that data_format exists in DEMO_JOINTS_REGISTRY."""
    if data_format not in DEMO_JOINTS_REGISTRY:
        available = ", ".join(sorted(DEMO_JOINTS_REGISTRY.keys()))
        raise ValueError(
            f"Invalid data_format: '{data_format}'. "
            f"Available data formats: {available}. "
            f"Add your format to DEMO_JOINTS_REGISTRY in config_types/data_type.py"
        )


@dataclass(frozen=True)
class MotionDataConfig:
    # Use str instead of Literal to allow dynamic data formats via DEMO_JOINTS_REGISTRY
    data_format: str = "smplh"
    # Use str instead of Literal to allow dynamic robot types
    robot_type: str = "g1"
    robot_defaults: dict[str, RobotDefaults] = field(default_factory=_default_robot_defaults)

    def __post_init__(self) -> None:
        """Validate data_format and robot_type."""
        _validate_data_format(self.data_format)

        _validate_robot_type(self.robot_type, self.robot_defaults)

    # Optional overrides - if None, will use defaults from data_format
    demo_joints: list[str] | None = None
    joints_mapping: dict[str, str] | None = None

    @property
    def resolved_demo_joints(self) -> list[str]:
        """Get demo joints - use override if provided, else use data_format default."""
        if self.demo_joints is not None:
            return self.demo_joints

        if self.data_format not in DEMO_JOINTS_REGISTRY:
            raise ValueError(
                f"Unknown data_format: {self.data_format}. "
                f"Available formats: {list(DEMO_JOINTS_REGISTRY.keys())}. "
                f"Add your format to DEMO_JOINTS_REGISTRY in config_types/data_type.py"
            )
        return DEMO_JOINTS_REGISTRY[self.data_format]

    @property
    def resolved_joints_mapping(self) -> dict[str, str]:
        """Get joints mapping - use override if provided, else lookup by (data_format, robot_type)."""
        if self.joints_mapping is not None:
            return self.joints_mapping

        key = (self.data_format, self.robot_type)
        if key in JOINTS_MAPPINGS:
            return JOINTS_MAPPINGS[key]

        raise ValueError(f"No joint mapping found for data_format={self.data_format}, robot_type={self.robot_type}")

    @property
    def toe_names(self) -> list[str]:
        """Get toe joint names for this data format."""
        if self.data_format not in TOE_NAMES_BY_FORMAT:
            raise ValueError(
                f"Toe names not defined for data_format: {self.data_format}. "
                f"Add entry to TOE_NAMES_BY_FORMAT in config_types/data_type.py"
            )
        return TOE_NAMES_BY_FORMAT[self.data_format]

    @property
    def default_scale_factor(self) -> float | None:
        """Get default scale factor for this data format (None if calculated per subject)."""
        format_constants: FormatConstants = DATA_FORMAT_CONSTANTS.get(self.data_format, {})
        return format_constants.get("default_scale_factor")

    @property
    def default_human_height(self) -> float | None:
        """Get default human height for this data format (None if not applicable)."""
        format_constants: FormatConstants = DATA_FORMAT_CONSTANTS.get(self.data_format, {})
        return format_constants.get("default_human_height")

    def legacy_constants(self) -> dict[str, Any]:
        """Return uppercase legacy constants for backward compatibility."""
        return {
            "DEMO_JOINTS": self.resolved_demo_joints,
            "JOINTS_MAPPING": self.resolved_joints_mapping,
            "TOE_NAMES": self.toe_names,
            "DEFAULT_SCALE_FACTOR": self.default_scale_factor,
            "DEFAULT_HUMAN_HEIGHT": self.default_human_height,
        }
