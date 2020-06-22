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



## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence  
By default, the speaker embedding layer is [ignored]

1. Download our published [Mellotron] model trained on LibriTTS
2. `python train.py --output_directory=outdir --log_directory=logdir -c models/mellotron_libritts.pt --warm_start`

## Multi-GPU (distributed) and Automatic Mixed Precision Training
1. `python -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,fp16_run=True`

## Inference demo
1. `jupyter notebook --ip=127.0.0.1 --port=31337`
2. Load inference.ipynb 
3. (optional) Download our published [WaveGlow](https://drive.google.com/open?id=1Rm5rV5XaWWiUbIpg5385l5sh68z2bVOE) model

## Related repos
[WaveGlow](https://github.com/NVIDIA/WaveGlow) Faster than real time Flow-based
Generative Network for Speech Synthesis.

## Acknowledgements
This implementation uses code from the following repos: [Keith
Ito](https://github.com/keithito/tacotron/), [Prem
Seetharaman](https://github.com/pseeth/pytorch-stft), 
[Chengqi Deng](https://github.com/KinglittleQ/GST-Tacotron),
[Patrice Guyot](https://github.com/patriceguyot/Yin), as described in our code.

[ignored]: https://github.com/NVIDIA/mellotron/blob/master/hparams.py#L22
[paper]: https://arxiv.org/abs/1910.11997
[WaveGlow]: https://drive.google.com/file/d/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx/view?usp=sharing
[Mellotron]: https://drive.google.com/open?id=1ZesPPyRRKloltRIuRnGZ2LIUEuMSVjkI
[pytorch]: https://github.com/pytorch/pytorch#installation
[website]: https://nv-adlr.github.io/Mellotron
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp
