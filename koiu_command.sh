name=vctk_gst_pretrained_2gpu
model_dir=/nfs/maximoff/ext01/mike/models/mellotron/models/${name}
log_dir=/nfs/maximoff/ext01/mike/models/mellotron/logs/${name}

koiu deploy -p 3.6.4 -v --image nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04 -g 2 -m maximoff -- \
    python -m multiproc train.py \
    -c /nfs/maximoff/ext01/mike/models/mellotron/models/${name}/checkpoint_23000 \
    --hparams=distributed_run=True \
    --output_directory=${model_dir} --log_directory=${log_dir}

    #-c /nfs/maximoff/ext01/mike/models/mellotron/models/pretrained/mellotron_libritts.pt --warm_start \
