cmd1=$1
cmd2=$2


if [ $cmd1 = preproc ]
then
    python data/vctk_preprocessor.py data-bin/VCTK-Corpus/
fi

IFS=''
hp=("training_files=filelists/vctk_train.txt,"
    "validation_files=filelists/vctk_val.txt,"
    "distributed_run=True,"
    "model_name=gst-tacotron,"
    "use_mine=False,"
    "p_teacher_forcing=1.0,"
    "episodic_training=False,"
    "num_query=16,"
    "num_support=16,"
    "num_common=16,"
    "dist_url=tcp://localhost:54320")
hp="${hp[*]}" 

if [ $cmd1 = train ]
then
    name=gst_tacotron2_vctk
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m multiproc train.py \
        --hparams=${hp} \
        -c models/pretrained/mellotron_libritts.pt --warm_start \
        --output_directory=models/$name --log_directory=logs/$name 

        #-c models/vctk_episodic_dual_24k/checkpoint_49000 --warm_start \
        #-c models/pretrained/mellotron_libritts.pt --warm_start \
        #-c models/vctk_gst_pretrained_2gpu/checkpoint_250000 \
fi

if [ $cmd1 = test ]
then
    # p287 (male), p265 (female)
    # ref >= 0: same reference / ref -1: difference references for each pred
    sid=p304
    ref_idx=-1
    seed=2020
    model=vctk_gst_pretrained_2gpu
    hp+=",ref_ind=-1,num_common=0"
    echo $hp
    for (( ckpt_n=280; ckpt_n<=280; ckpt_n+=10 ))
    do
        name=${model}_${ckpt_n}k_${sid}_${ref_idx}
        echo $name
        CUDA_VISIBLE_DEVICES=${cmd2} python evaluation.py \
            --save_audio=1 \
            --output_dir=audios/${name} \
            --audio_path=filelists/vctk_val.txt \
            --hparam_string=${hp} \
            --ckpt=models/${model}/checkpoint_${ckpt_n}000 \
            --seed=${seed} \
            --supportset_sid=${sid}
    done
fi



IFS=''
hp=("training_files=filelists/vctk_train.txt,"
    "validation_files=filelists/vctk_val.txt,"
    "distributed_run=True,"
    "model_name=episodic-baseline,"
    "use_mine=False,"
    "p_teacher_forcing=1.0,"
    "episodic_training=False,"
    "dist_url=tcp://localhost:54320")
hp="${hp[*]}" 

if [ $cmd1 = koiu ]
then
    name=0529_gst_slightly_diff
    model_dir=/nfs/maximoff/ext01/mike/models/mellotron/models/${name}
    log_dir=/nfs/maximoff/ext01/mike/models/mellotron/logs/${name}
    time koiu deploy -p 3.6.4 -v --image nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04 -g 4 -m maximoff -- \
        python -m multiproc train.py \
        --hparams=${hp} \
        -c /nfs/maximoff/ext01/mike/models/mellotron/models/pretrained/mellotron_libritts.pt --warm_start \
        --output_directory=${model_dir} --log_directory=${log_dir}

fi









#
