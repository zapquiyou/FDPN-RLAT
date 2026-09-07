python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vits_sceneflow/defomstereo_vits_sceneflow_70000.pth  \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets things \
--dinov2_encoder vits \
--result_path /data/StereoDatasets/results/DEFOM-Stereo_rltrain_70000.json \
--use_rl rltrain 

python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vits_sceneflow/defomstereo_vits_sceneflow_80000.pth  \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets things \
--dinov2_encoder vits \
--result_path /data/StereoDatasets/results/DEFOM-Stereo_rltrain_80000.json \
--use_rl rltrain

python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vits_sceneflow/defomstereo_vits_sceneflow_90000.pth  \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets things \
--dinov2_encoder vits \
--result_path /data/StereoDatasets/results/DEFOM-Stereo_rltrain_90000.json \
--use_rl rltrain

python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vits_sceneflow/defomstereo_vits_sceneflow_100000.pth  \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets things \
--dinov2_encoder vits \
--result_path /data/StereoDatasets/results/DEFOM-Stereo_rltrain_100000.json \
--use_rl rltrain

python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vits_sceneflow/defomstereo_vits_sceneflow_70000.pth  \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets things \
--dinov2_encoder vits \
--result_path /data/StereoDatasets/results/DEFOM-Stereo_rltest_70000.json \
--use_rl rltest

python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vits_sceneflow/defomstereo_vits_sceneflow_80000.pth  \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets things \
--dinov2_encoder vits \
--result_path /data/StereoDatasets/results/DEFOM-Stereo_rltest_80000.json \
--use_rl rltest

python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vits_sceneflow/defomstereo_vits_sceneflow_90000.pth  \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets things \
--dinov2_encoder vits \
--result_path /data/StereoDatasets/results/DEFOM-Stereo_rltest_90000.json \
--use_rl rltest

python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vits_sceneflow/defomstereo_vits_sceneflow_100000.pth  \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--datasets things \
--dinov2_encoder vits \
--result_path /data/StereoDatasets/results/DEFOM-Stereo_rltest_100000.json \
--use_rl rltest