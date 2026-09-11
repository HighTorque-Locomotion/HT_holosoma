## Instructions for Adding Custom Human Motion Data Format

This guide shows you how to add a new data format (e.g., "myformat") to the retargeting pipeline. We use SMPLX as an example, which is already implemented.

### PARC MS support

This repository now includes a `parc_ms` adapter based on the PARC MS reader in
`human-humanoid-tools`.  A complete clip must have this layout:

```text
<dataset-root>/<clip>/<clip>.pkl
<dataset-root>/<clip>/<clip>_terrain.obj
```

The adapter reconstructs the canonical 15-body PARC humanoid with forward
kinematics, synthesizes left/right toe contact points, and attaches a scaled
terrain mesh to the supplied robot MuJoCo XML.  For collision checking, the
non-convex terrain mesh is kept visual-only and replaced by per-heightfield-cell
box geoms; this avoids MuJoCo's convex-hull approximation lifting the robot off
the flat parts of the terrain.  It then calls the existing
`InteractionMeshRetargeter`; no `human-humanoid-tools` Python import is required
at runtime.

Example:

```bash
python examples/robot_retarget.py \
  --data_path /path/to/parc_ms_selected \
  --task-type climbing \
  --task-name climbing_up_down_terrain_001_aug017_dm_aug5 \
  --data_format parc_ms \
  --robot piplus_s \
  --robot-config.robot-urdf-file models/PiPlus_S_12L8A0G2H0W_SSE_ZedMini_260831/urdf/PiPlus_S_12L8A0G2H0W_SSE_ZedMini_260831_raw.urdf \
  --robot-config.robot-xml-file models/PiPlus_S_12L8A0G2H0W_SSE_ZedMini_260831/xml/PiPlus_S_12L8A0G2H0W_SSE_ZedMini_260831.xml \
  --task-config.object-name multi_boxes \
  --save-dir demo_results/piplus_s/climbing/parc_ms
```

To export only the converted `global_joint_positions` representation:

```bash
python -m holosoma_retargeting.data_utils.convert_parc_ms \
  --input /path/to/clip/clip.pkl \
  --output /tmp/clip.npz
```

### PiPlus qpos to Holosoma WBT

Resample retargeted PiPlus qpos to the policy control rate before computing FK
and velocities.  For the default Holosoma WBT configuration this is 50 Hz:

```bash
python -m holosoma_retargeting.data_utils.convert_piplus_to_holosoma \
  /path/to/retargeted_mapping.npz \
  /path/to/motion_holosoma_fps50.npz \
  --model-xml /path/to/PiPlus_S_12L8A0G2H0W_Soccer_New.xml \
  --output-fps 50
```

Root translation and joint angles are linearly interpolated; root orientation
uses quaternion SLERP. FK and all finite-difference velocities are recomputed
after resampling. This preserves the original duration and avoids playing a
30 FPS source clip at 50 frames per second during training.

### Overview

1. **Prepare your data** (prepare .npz files which contain global joint positions and human height information)
2. **Add your joint names** (create a constant like `MYFORMAT_DEMO_JOINTS`)
3. **Register your format** in the unified registry (`DEMO_JOINTS_REGISTRY`)
4. **Add format-specific constants** (toe names, joint mappings, height)

### Step 1: Prepare Your Data Files

Prepare `.npz` files for each motion sequence:
- **`.npz` format**: Should contain `global_joint_positions` array (T X J X 3) and `height` scalar

**Example**: We provide `data_utils/prep_amass_smplx_for_rt.py` for converting AMASS SMPLX data:
```bash
# Install dependencies
git clone https://github.com/nghorbani/human_body_prior.git
pip install tqdm dotmap PyYAML omegaconf loguru
cd human_body_prior/
python setup.py develop

# Run data processing
python prep_amass_smplx_for_rt.py \
  --amass-root-folder /path/to/amass \
  --output-folder /path/to/output \
  --model-root-folder /path/to/models
```
Please follow the [AMASS](https://amass.is.tue.mpg.de/) instructions to download original data. And follow the [SMPL-X](https://smpl-x.is.tue.mpg.de/index.html) instructions to download SMPL-X models. For AMASS data, we tested on SMPL-X N format. The AMASS data structure should be `/path/to/amass/dataset_name/subject_name/*.npz`. For SMPL-X models, the structure should be `/path/to/models/smplx/SMPLX_NEUTRAL.npz`.

### Step 2: Add Your Format to `config_types/data_type.py`

Edit **only this file** - all format configuration is centralized here!

#### 2.1: Define Joint Names

Add a constant with your joint names. **Important**: The order of joint names must match the order of joints in the `J` dimension of your `global_joint_positions` array from Step 1.

For example, if your `global_joint_positions` array has shape `(T, J, 3)` where:
- `T` = number of timesteps
- `J` = number of joints
- `3` = x, y, z coordinates

Then `MYFORMAT_DEMO_JOINTS[0]` should be the name of the joint at `global_joint_positions[:, 0, :]`, `MYFORMAT_DEMO_JOINTS[1]` should be the name of the joint at `global_joint_positions[:, 1, :]`, and so on.

```python
MYFORMAT_DEMO_JOINTS = [
    "Joint1",  # Corresponds to global_joint_positions[:, 0, :]
    "Joint2",  # Corresponds to global_joint_positions[:, 1, :]
    # ... list all joints in order matching the J dimension
]
```

#### 2.2: Register Your Format

Add your format to the unified registry (this is the **main place** to register):

```python
DEMO_JOINTS_REGISTRY: dict[str, list[str]] = {
    "lafan": LAFAN_DEMO_JOINTS,
    "smplh": SMPLH_DEMO_JOINTS,
    "mocap": MOCAP_DEMO_JOINTS,
    "smplx": SMPLX_DEMO_JOINTS,
    "myformat": MYFORMAT_DEMO_JOINTS,  # ← Add your format here
}
```

#### 2.3: Add Format-Specific Constants

Add entries to these dictionaries.

**Required:**
- `TOE_NAMES_BY_FORMAT` - Must include toe joint names for foot-sticking constraint
- `JOINTS_MAPPINGS` - Must include mappings for each robot type you support

**Toe names** (used for foot sticking constraint):
```python
TOE_NAMES_BY_FORMAT = {
    # ... existing formats ...
    "myformat": ["LeftToe", "RightToe"],  # ← Add your toe joint names
}
```

**Joint mappings** (human joint → robot joint):
```python
JOINTS_MAPPINGS = {
    # ... existing mappings ...
    ("myformat", "g1"): {  # ← Add for each robot type you support
        "HumanJoint1": "robot_joint_1",
        "HumanJoint2": "robot_joint_2",
        # ... map all relevant joints
    },
    ("myformat", "t1"): {  # ← If supporting t1 robot
        # ... mappings for t1
    },
}
```

### Step 3: Add Data Loading Logic (if needed)

**Important**: If you processed your data to the `.npz` format in Step 1 (with `global_joint_positions` and `height` keys), you can **skip this step entirely**. The code automatically handles `.npz` files with this structure via a fallback mechanism.

If your format needs special loading logic (different file extension, custom preprocessing, etc.), edit `examples/robot_retarget.py` in the `load_motion_data()` function. Add your format before the fallback `else` clause:

```python
def load_motion_data(...):
    # ... existing code ...
    elif data_format == "smplx":
        npz_file = data_path / f"{task_name}.npz"
        human_data = np.load(str(npz_file))
        human_joints = human_data["global_joint_positions"]
        human_height = human_data["height"]
        smpl_scale = constants.ROBOT_HEIGHT / human_height
    elif data_format == "myformat":
        # Add your custom loading logic here
        npy_path = data_path / f"{task_name}.npy"
        human_joints = np.load(str(npy_path))
        # ... any preprocessing ...
        smpl_scale = constants.ROBOT_HEIGHT / default_human_height
    else:
        # Fallback: handles .npz files with global_joint_positions and height
        # ... (automatic handling for standard .npz format)
```


### Summary: What You Need to Edit

**In `config_types/data_type.py`** (main configuration file):
1. ✅ **Required**: Create `MYFORMAT_DEMO_JOINTS` constant
2. ✅ **Required**: Add to `DEMO_JOINTS_REGISTRY`
3. ✅ **Required**: Add to `TOE_NAMES_BY_FORMAT`
4. ✅ **Required**: Add to `JOINTS_MAPPINGS`

**In `examples/robot_retarget.py`** (only if needed):
7. ⚠️ **Optional**: Add loading logic in `load_motion_data()` (only if format needs special handling beyond standard `.npz` format)


### Ready to Run

Once configured, you can use your custom format:

```bash
python examples/robot_retarget.py \
  --data_path /path/to/your/data \
  --task-type robot_only \
  --task-name your_sequence_name \
  --data_format myformat \
  --retargeter.debug \
  --retargeter.visualize
```
