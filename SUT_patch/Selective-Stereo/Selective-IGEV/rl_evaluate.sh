python evaluate_stereo.py --restore_ckpt checkpoints/sceneflow/70000_igev-stereo.pth --result_path /data/StereoDatasets/results/Selective_Stereo_rltrain_70000.json --use_rl rltrain

python evaluate_stereo.py --restore_ckpt checkpoints/sceneflow/80000_igev-stereo.pth --result_path /data/StereoDatasets/results/Selective_Stereo_rltrain_80000.json --use_rl rltrain

python evaluate_stereo.py --restore_ckpt checkpoints/sceneflow/90000_igev-stereo.pth --result_path /data/StereoDatasets/results/Selective_Stereo_rltrain_90000.json --use_rl rltrain

python evaluate_stereo.py --restore_ckpt checkpoints/sceneflow/100000_igev-stereo.pth --result_path /data/StereoDatasets/results/Selective_Stereo_rltrain_100000.json --use_rl rltrain

python evaluate_stereo.py --restore_ckpt checkpoints/sceneflow/70000_igev-stereo.pth --result_path /data/StereoDatasets/results/Selective_Stereo_rltest_70000.json --use_rl rltest

python evaluate_stereo.py --restore_ckpt checkpoints/sceneflow/80000_igev-stereo.pth --result_path /data/StereoDatasets/results/Selective_Stereo_rltest_80000.json --use_rl rltest

python evaluate_stereo.py --restore_ckpt checkpoints/sceneflow/90000_igev-stereo.pth --result_path /data/StereoDatasets/results/Selective_Stereo_rltest_90000.json --use_rl rltest

python evaluate_stereo.py --restore_ckpt checkpoints/sceneflow/100000_igev-stereo.pth --result_path /data/StereoDatasets/results/Selective_Stereo_rltest_100000.json --use_rl rltest