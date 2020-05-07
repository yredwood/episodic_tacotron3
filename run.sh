cmd1=$1
cmd2=$2



#name=original_mellotron_nof0nosp_transpose_from_pretrained
#name=bigger_gst_with_speakerembeddim_lr1e-3_episodic
#name=gst_transformer_pretrained-anealedfrom45k
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m multiproc train.py --hparams=distributed_run=True \
#    -c models/pretrained/mellotron_libritts.pt --warm_start \
#    --output_directory=models/$name --log_directory=logs/$name 

#name=libri__episodic_dual
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m multiproc train.py --hparams=distributed_run=True \
#    -c models/pretrained/mellotron_libritts.pt --warm_start \
#    --output_directory=models/$name --log_directory=logs/$name 
    #-c models/episodic_dual/checkpoint_70000 \


#name=test2
#CUDA_VISIBLE_DEVICES=0 python train.py --hparams=distributed_run=False \
#    -c models/episodic_dual/checkpoint_70000 --warm_start \
#    --output_directory=models/$name --log_directory=logs/$name 
    #-c models/pretrained/mellotron_libritts.pt --warm_start \

name=vctk_episodic_dual_24k
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m multiproc train.py \
        --hparams=training_files=filelists/vctk_train.txt,validation_files=filelists/vctk_val.txt,distributed_run=True \
        -c models/pretrained/mellotron_libritts.pt --warm_start \
        --output_directory=models/$name --log_directory=logs/$name
