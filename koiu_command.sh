name=vctk_lin_spec
model_dir=/nfs/maximoff/ext01/mike/models/mellotron/models/${name}
log_dir=/nfs/maximoff/ext01/mike/models/mellotron/logs/${name}
#model_dir=models
#log_dir=logs
#pretrain_dir=/nfs/maximoff/ext01/mike/models/mellotron/models/pretrained/mellotron_libritts.pt


koiu deploy -p 3.6.4 -v --image nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04 -g 8 -m maximoff -- \
    python -m multiproc train.py \
    --hparams=training_files=filelists/vctk_train.txt,validation_files=filelists/vctk_val.txt,distributed_run=True,sampling_rate=48000 \
    -c /nfs/maximoff/ext01/mike/models/mellotron/models/pretrained/mellotron_libritts.pt --warm_start \
    --output_directory=${model_dir} --log_directory=${log_dir}


#CUDA_VISIBLE_DEVICES=0,1 python -m multiproc train.py \
#    --hparams=training_files=filelists/vctk_train.txt,validation_files=filelists/vctk_val.txt,distributed_run=True,sampling_rate=48000 \
#    -c /nfs/maximoff/ext01/mike/models/mellotron/models/pretrained/mellotron_libritts.pt --warm_start \
#    --output_directory=${model_dir} --log_directory=${log_dir}
