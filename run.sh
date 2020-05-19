cmd1=$1
cmd2=$2


IFS=''
hp=("training_files=filelists/vctk_train.txt,"
    "validation_files=filelists/vctk_val.txt,"
    "distributed_run=True,"
    "episodic_training=True,"
    "model_name=episodic-baseline,"
    "transformer_type=dual_baseline,"
    "use_mine=False,"
    "dist_url=tcp://localhost:54325")
hp="${hp[*]}" 


if [ $cmd1 = train ]
then
#name=vctk_episodic_dual_scratchpretrained
    name=vctk_episodic
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m multiproc train.py \
        --hparams=${hp} \
        -c models/vctk_gst_pretrained_2gpu/checkpoint_250000 --warm_start \
        --output_directory=models/$name --log_directory=logs/$name 

        #-c models/pretrained/mellotron_libritts.pt --warm_start \
        #-c models/vctk_episodic_dual_24k/checkpoint_49000 --warm_start \
fi



# test script
if [ $cmd1 = test ]
then
    # p287 (male), p265 (female)
    # ref >= 0: same reference / ref -1: difference references for each pred
    sid=p287
    ref_idx=-1
    seed=2020
    model=vctk_gst_pretrained_2gpu
    #model=vctk_episodic_mine
    hp+=",ref_ind=0,num_common=0"
    echo $hp
    for (( ckpt_n=280; ckpt_n<=280; ckpt_n+=1 ))
    do
        name=${model}_${ckpt_n}k_${sid}_${ref_idx}
        CUDA_VISIBLE_DEVICES=${cmd2} python evaluation.py \
            --save_audio=1 \
            --output_dir=audios/${name} \
            --audio_path=filelists/vctk_24val.txt \
            --hparam_string=${hp} \
            --ckpt=models/${model}/checkpoint_${ckpt_n}000 \
            --seed=${seed} \
            --supportset_sid=${sid}
    done
fi


