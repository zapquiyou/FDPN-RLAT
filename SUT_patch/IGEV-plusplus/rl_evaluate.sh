python evaluate_stereo.py --restore_ckpt checkpoints/70000_igev-stereo.pth --result_path /data/StereoDatasets/results/IGEV_plusplus_rltrain_70000.json --use_rl rltrain --dataset sceneflow

python evaluate_stereo.py --restore_ckpt checkpoints/80000_igev-stereo.pth --result_path /data/StereoDatasets/results/IGEV_plusplus_rltrain_80000.json --use_rl rltrain --dataset sceneflow

python evaluate_stereo.py --restore_ckpt checkpoints/90000_igev-stereo.pth --result_path /data/StereoDatasets/results/IGEV_plusplus_rltrain_90000.json --use_rl rltrain --dataset sceneflow

python evaluate_stereo.py --restore_ckpt checkpoints/100000_igev-stereo.pth --result_path /data/StereoDatasets/results/IGEV_plusplus_rltrain_100000.json --use_rl rltrain --dataset sceneflow

python evaluate_stereo.py --restore_ckpt checkpoints/70000_igev-stereo.pth --result_path /data/StereoDatasets/results/IGEV_plusplus_rltest_70000.json --use_rl rltest --dataset sceneflow

python evaluate_stereo.py --restore_ckpt checkpoints/80000_igev-stereo.pth --result_path /data/StereoDatasets/results/IGEV_plusplus_rltest_80000.json --use_rl rltest --dataset sceneflow

python evaluate_stereo.py --restore_ckpt checkpoints/90000_igev-stereo.pth --result_path /data/StereoDatasets/results/IGEV_plusplus_rltest_90000.json --use_rl rltest --dataset sceneflow

python evaluate_stereo.py --restore_ckpt checkpoints/100000_igev-stereo.pth --result_path /data/StereoDatasets/results/IGEV_plusplus_rltest_100000.json --use_rl rltest --dataset sceneflow