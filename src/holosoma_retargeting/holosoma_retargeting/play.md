cd /home/sunteng/HT/tool/holosoma/src/holosoma_retargeting/holosoma_retargeting

python examples/robot_retarget.py \
  --data_path demo_data/climb \
  --task-type climbing \
  --task-name mocap_climb_seq_4 \
  --data_format mocap \
  --robot piplus_s \
  --robot-config.robot-urdf-file models/PiPlus_S_12L8A0G2H0W_SSE_ZedMini_260831/urdf/PiPlus_S_12L8A0G2H0W_SSE_ZedMini_260831_raw.urdf \
  --robot-config.robot-xml-file models/PiPlus_S_12L8A0G2H0W_SSE_ZedMini_260831/xml/PiPlus_S_12L8A0G2H0W_SSE_ZedMini_260831.xml \
  --task-config.object-name multi_boxes \
  --save-dir demo_results/piplus_s/climbing/piplus_selfcollision_test \
  --retargeter.self-collision.enable \
--retargeter.self-collision.pairs \
l_ankle_roll_link r_ankle_roll_link \
l_ankle_pitch_link r_ankle_pitch_link \
l_ankle_roll_link r_ankle_pitch_link \
l_ankle_pitch_link r_ankle_roll_link \
--retargeter.self-collision.tolerance 0.02 \
--retargeter.step-size 0.05





cd /home/sunteng/HT/tool/holosoma/src/holosoma_retargeting/holosoma_retargeting

python viser_player.py \
  --qpos-npz demo_results/piplus_s/climbing/piplus_selfcollision_test/mocap_climb_seq_4_original.npz \
  --robot-urdf models/PiPlus_S_12L8A0G2H0W_SSE_ZedMini_260831/urdf/PiPlus_S_12L8A0G2H0W_SSE_ZedMini_260831_raw.urdf \
  --object-urdf demo_data/climb/mocap_climb_seq_4/multi_boxes_scaled_0.38_0.38_0.38.urdf \
  --no-assume-object-in-qpos \
  --fps 30 \
  --loop








  重定向G1

source /home/sunteng/.holosoma_deps/miniconda3/etc/profile.d/conda.sh
conda activate hsretargeting


cd /home/sunteng/HT/tool/holosoma/src/holosoma_retargeting/holosoma_retargeting

python examples/robot_retarget.py \
  --data_path demo_data/climb \
  --task-type climbing \
  --task-name mocap_climb_seq_2 \
  --data_format mocap \
  --robot g1 \
  --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf \
  --robot-config.robot-xml-file models/g1/g1_29dof_spherehand.xml \
  --task-config.object-name multi_boxes \
  --save-dir demo_results/g1/climbing/mocap_climb

  python viser_player.py \
  --qpos-npz demo_results/g1/climbing/mocap_climb/mocap_climb_seq_4_original.npz \
  --robot-urdf models/g1/g1_29dof_spherehand.urdf \
  --object-urdf demo_data/climb/mocap_climb_seq_4/multi_boxes_scaled_0.74_0.74_0.74.urdf \
  --no-assume-object-in-qpos \
  --fps 30 \
  --loop



  从G1->pi
  python examples/g1_qpos_retarget.py \
  --g1-qpos-npz /home/sunteng/HT/tool/OmniRetarget_Dataset/robot-terrain/climb_04_z_scale_1.0.npz \
  --terrain-urdf /home/sunteng/HT/tool/OmniRetarget_Dataset/models/terrain/climb_04/multi_boxes_z_scale_1.0.urdf \
  --g1-xml-file models/g1/g1_29dof_spherehand.xml \
  --piplus-urdf-file models/PiPlus_S_12L8A0G2H0W_SSE_ZedMini_260831/urdf/PiPlus_S_12L8A0G2H0W_SSE_ZedMini_260831_raw.urdf \
  --piplus-xml-file models/PiPlus_S_12L8A0G2H0W_SSE_ZedMini_260831/xml/PiPlus_S_12L8A0G2H0W_SSE_ZedMini_260831.xml \
  --save-dir demo_results/piplus_s/climbing/omniretarget/climb_04_z_scale_1.0_piplus_mapping \
  --output-name climb_04_z_scale_1.0_piplus_mapping.npz \
  <!-- --retargeter.self-collision.enable \
  --retargeter.self-collision.pairs \
  l_ankle_roll_link r_ankle_roll_link \
  l_ankle_pitch_link r_ankle_pitch_link \
  l_ankle_roll_link r_ankle_pitch_link \
  --retargeter.self-collision.tolerance 0.00  -->
  <!-- l_ankle_pitch_link r_ankle_roll_link \
  r_wrist_link r_hip_roll_link \
  l_wrist_link l_hip_roll_link \ -->

python examples/g1_qpos_retarget.py \
  --g1-qpos-npz /home/sunteng/HT/tool/OmniRetarget_Dataset/robot-terrain/climb_23_z_scale_1.0.npz \
  --terrain-urdf /home/sunteng/HT/tool/OmniRetarget_Dataset/models/terrain/climb_23/multi_boxes_z_scale_1.0.urdf \
  --g1-xml-file models/g1/g1_29dof_spherehand.xml \
  --piplus-urdf-file models/PiPlus_S_12L8A0G2H0W_LSE_ZedMini_40V_260908/urdf/PiPlus_S_12L8A0G2H0W_LSE_ZedMini_40V_260908_raw.urdf \
  --piplus-xml-file models/PiPlus_S_12L8A0G2H0W_LSE_ZedMini_40V_260908/xml/PiPlus_S_12L8A0G2H0W_LSE_ZedMini_40V_260908.xml \
  --piplus-collision-geometry raw-mesh \
  --bundle-dir demo_results/piplus_s/climbing/omniretarget/climb_23_z_scale_1.0_piplus_mapping \
  --output-name climb_23_z_scale_1.0_piplus_mapping.npz \
  --retargeter.self-collision.enable \
  --retargeter.self-collision.pairs \
    l_ankle_roll_link r_ankle_roll_link \
    l_ankle_pitch_link r_ankle_pitch_link \
    l_ankle_roll_link r_ankle_pitch_link \
    l_ankle_pitch_link r_ankle_roll_link \
    r_wrist_link r_hip_roll_link \
    l_wrist_link l_hip_roll_link \
  --retargeter.self-collision.tolerance 0.0



python viser_player.py \
  --qpos-npz demo_results/piplus_s/climbing/omniretarget/climb_23_z_scale_1.0_piplus_mapping/climb_23_z_scale_1.0_piplus_mapping.npz \
  --robot-urdf models/PiPlus_S_12L8A0G2H0W_LSE_ZedMini_40V_260908/urdf/PiPlus_S_12L8A0G2H0W_LSE_ZedMini_40V_260908_raw.urdf \
  --object-urdf demo_results/piplus_s/climbing/omniretarget/climb_23_z_scale_1.0_piplus_mapping/climb_23_z_scale_1.0_terrain_piplus.urdf \
  --no-assume-object-in-qpos \
  --fps 30 \
  --visual-fps-multiplier 1 \
  --loop