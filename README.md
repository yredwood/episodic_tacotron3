Mellotron based GST-tacotron2 + episodic training + ...

## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Setup
1. Clone this repo: `git clone -b dev2 https://github.com/yredwood/episodic_tacotron3`
2. `cd episodic_tacotron3`
3. `git submodule init; git submodule update`
4. Intall Pytorch & Apex
5. Install requirements with `source docker_pip.sh`


## Data preparation (only VCTK supported)
1. wget http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz
2. uncompress the file
3. locate `VCTK-Corpus` to data-bin/
4. `./run.sh preproc`

## Training from LJSpeech pretrained model
1. download pretrained model from https://drive.google.com/file/d/1ZesPPyRRKloltRIuRnGZ2LIUEuMSVjkI/view
2. `./run.sh train`


## Evalate / Inference model
1. download vocoder model (or griffin-lim can be used) [WaveGlow](https://drive.google.com/open?id=1Rm5rV5XaWWiUbIpg5385l5sh68z2bVOE)
2. `./run.sh test 0` # 0: GPU_ID 
3. Generated samples will be saved in audios/...


