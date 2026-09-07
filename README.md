# Table of Contents
1. [Introduce](#introduction)
2. [Installation](#installation)
3. [Experimental step](#experimental-steps)

# Introduction

This repository contains the implementation of FDPN-RLAT. This work proposes an intelligent testing strategy based on reinforcement learning with a Feedback-Driven Dual-Path Pointer Network (FDPN), which formulates test case selection and test termination as a sequential decision-making process. The method leverages both test-case context information and execution feedback to adaptively prioritize test cases and determine when testing should stop, with the goal of improving failure detection efficiency for intelligent systems.

# Installation
It is recommended to use conda to configure the environment in which the program runs. You can use the following command to create a python environment and activate it.

```bash
conda create -n fdpn_rlat python=3.8
conda activate fdpn_rlat
```

Then install the libraries needed in the experiment using requirements.txt.

```bash
pip install -r requirements.txt
```

In addition, if you want to conduct a complete experiment, you need to configure the dependent environment according to the README files of each SUT.

**Note:**  This experiment is run on the Ubuntu system.

# Experimental steps

## Stereo matching task

The experimental steps are as follows: 
1. Download the Scene Flow dataset and organize it according to the following directory structure. The website link is https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html;


```
/data/StereoDatasets/sceneflow/    # This is an absolute path
├── disparity/                     
│   ├── driving/             
│   └── flying/               
│   └── monkaa/
└── frames_finalpass/
    ├── driving/             
    └── flying/               
    └── monkaa/         
```

2. Run `dataset_split.py` to partition the dataset. To restore the original dataset partition, run `dataset_recovery.py`;


3. Download the repository of each SUT (IGEV++, DEFOM-Stereo, MonSter, and Selective-Stereo) and replace the corresponding files with those provided in the `SUT_patch` directory. Then, train each SUT for 70k, 80k, 90k, and 100k steps, respectively. After training, use the corresponding `.sh` script provided in the patch directory to evaluate each checkpoint;

4. Run ViT.py to preprocess the sample;

5. To train the FDPN-RLAT, you can run:

```bash
python FDPN_RLAT.py --mode train --model_path your_checkpoint_name.pth
```

Then evaluate FDPN-RLAT (default parameters are for stereo matching task):

```bash
python FDPN_RLAT.py --mode eval --model_path your_checkpoint_name.pth
```

In order to collect experimental results, it is recommended to run it with the nohup command, the following steps are the same:

```bash
nohup python -u <script_name>.py <arguments> > <log_name>.out 2>&1 &
```

## Keyword spotting task

The experimental steps are as follows: 
1. Download the Google Speech Commands v2 dataset and organize it according to the following directory structure. The download link is https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz;


```
/data/gsc/              # This is an absolute path
├── <label>
.
.
.
└── <label>       
```

2. Run gsc_split.py to partition the dataset;

3. Download the repository of kws_streaming and replace the corresponding files with those provided in the `SUT_patch` directory. Then use the corresponding `.sh` script provided in the directory to train and evaluate each checkpoint;

4. Run command `wav2vec_attention_pool.py --train-splits training rl_train` to train the attention pooling module, and then run `wav2vec.py` to extract fixed-length speech features;

5. To train the FDPN-RLAT, you can run:

```bash
python FDPN_RLAT.py --mode train --model_path your_checkpoint_name.pth --task gsc
```

Then evaluate FDPN-RLAT:

```bash
python FDPN_RLAT.py --mode eval --model_path your_checkpoint_name.pth --task gsc
```

In order to collect experimental results, it is recommended to run it with the nohup command, the following steps are the same:

```bash
nohup python -u <script_name>.py <arguments> > <log_name>.out 2>&1 &
```
