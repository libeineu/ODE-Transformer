#!/usr/bin/bash
set -e

model_root_dir=checkpoints

# set task
task=wmt-en2fr
# set tag
model_dir_tag=RK2-learnbale-layer12-Big-RPR
# set device
gpu=0
cpu=

# data set
who=test

if [ $task == "wmt-en2de" ]; then
        data_dir=google
        ensemble=5
        batch_size=64
        beam=4
        length_penalty=0.6
        src_lang=en
        tgt_lang=de
        sacrebleu_set=wmt14/full
elif [ $task == "wmt-en2fr" ]; then
        data_dir=wmt_en_fr_joint_bpe
        ensemble=5
        batch_size=64
        beam=5
        length_penalty=0.8
        src_lang=en
        tgt_lang=fr
        sacrebleu_set=wmt14/full
elif [ $task == "wmt-en2ro" ]; then
        data_dir=wmt-en2ro
        ensemble=5
        batch_size=128
        beam=5
        length_penalty=1.3
        src_lang=en
        tgt_lang=ro
        sacrebleu_set=
else
        echo "unknown task=$task"
        exit
fi

model_dir=$model_root_dir/$task/$model_dir_tag

checkpoint=checkpoint_best.pt

if [ -n "$ensemble" ]; then
        if [ ! -e "$model_dir/last$ensemble.ensemble.pt" ]; then
                PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
        fi
        checkpoint=last$ensemble.ensemble.pt
fi

output=$model_dir/translation.log

if [ -n "$cpu" ]; then
        use_cpu=--cpu
fi

export CUDA_VISIBLE_DEVICES=$gpu

python3 generate.py \
data-bin/$data_dir \
--path $model_dir/$checkpoint \
--gen-subset $who \
--batch-size $batch_size \
--beam $beam \
--lenpen $length_penalty \
--output $model_dir/hypo.txt \
--quiet \
--remove-bpe $use_cpu | tee $output

python3 rerank.py $model_dir/hypo.txt $model_dir/hypo.sorted

if [ $data_dir == "google" ]; then
	sh $get_ende_bleu $model_dir/hypo.sorted
	perl $detokenizer -l de < $model_dir/hypo.sorted > $model_dir/hypo.dtk
fi

if [ $sacrebleu_set == "wmt14/full" ]; then

        echo -e "\n>> BLEU-13a"
        cat $model_dir/hypo.dtk | sacrebleu ../en-de.de -tok 13a

        echo -e "\n>> BLEU-intl"
        cat $model_dir/hypo.dtk | sacrebleu ../en-de.de -tok intl
fi