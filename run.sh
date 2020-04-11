cmd1=$1
cmd2=$2




#name=original_mellotron_nof0nosp_transpose_from_pretrained
#name=bigger_gst_with_speakerembeddim_lr1e-3_episodic
name=transformer3
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m multiproc train.py --hparams=distributed_run=True \
    -c models/pretrained/mellotron_libritts.pt --warm_start \
    --output_directory=models/$name --log_directory=logs/$name 

    #-c models/$name/checkpoint_2000 \

#CUDA_VISIBLE_DEVICES=3 python train.py --hparams=distributed_run=False \
#    -c models/pretrained/mellotron_libritts.pt --warm_start \
#    --output_directory=models/$name --log_directory=logs/$name 
    #-c models/$name/checkpoint_2000 \
