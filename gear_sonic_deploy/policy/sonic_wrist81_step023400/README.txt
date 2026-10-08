SONIC release checkpoint bundle — model_step_023400
source: gear:/wbc_data/models/TRL_G1_Track/20260815_150022-manager/universal_token/
        all_modes/sonic_release_3pt_heading_thor_overheating_ankle_adaptivefix_wrist_81_from100k/
run: sonic_release_3pt_heading_thor_overheating_ankle_adaptivefix_wrist_81_from100k (from100k)
exported 2026-08-19 16:34 UTC

ONNX (exported/):
  model_step_023400_encoder.onnx    50,147,326 B  tokenizer/encoder
  model_step_023400_decoder.onnx   149,784,569 B  token -> action decoder
  model_step_023400_g1.onnx        166,914,519 B  g1 policy
  model_step_023400_smpl.onnx      168,572,807 B  smpl policy (deployment encoder path)
  model_step_023400_teleop.onnx    163,891,819 B  teleop policy

configs:
  config.yaml         full hydra training config (32,544 B)
  model_config.yaml   model-only config (8,973 B)

NOT included (1.16 GB, available on gear at the path above and in
out/20260921_sonic_023400_bundle/sonic_wrist81_step023400_with_configs.zip):
  model_step_023400.pt   torch checkpoint
