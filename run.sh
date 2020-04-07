cmd1=$1
cmd2=$2





name=original_mellotron_nof0nosp_transpose
CUDA_VISIBLE_DEVICES=6,7 python -m multiproc train.py --hparams=distributed_run=True \
    -c models/pretrained/mellotron_libritts.pt --warm_start \
    --output_directory=models/$name --log_directory=logs/$name 

#CUDA_VISIBLE_DEVICES=4 python train.py --hparams=distributed_run=False \
#    --output_directory=models/$name --log_directory=logs/$name 
