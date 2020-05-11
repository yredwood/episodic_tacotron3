cmd1=$1
cmd2=$2




#name=original_mellotron_nof0nosp_transpose_from_pretrained
name=vctk_gst_las_scratch
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m multiproc train.py --hparams=distributed_run=True \
    --output_directory=models/$name --log_directory=logs/$name 

    #-c models/$name/checkpoint_18000 \
    #-c models/pretrained/mellotron_libritts.pt --warm_start \
    #-c models/$name/checkpoint_2000 \

#CUDA_VISIBLE_DEVICES=3 python train.py --hparams=distributed_run=False \
#    -c models/pretrained/mellotron_libritts.pt --warm_start \
#    --output_directory=models/$name --log_directory=logs/$name 
    #-c models/$name/checkpoint_2000 \
