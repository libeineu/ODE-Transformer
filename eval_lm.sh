save_dir=checkpoints/lm/wiki/baseline
export CUDA_VISIBLE_DEVICES=0
python3 eval_lm.py \
    data-bin/wikitext-103 \
    --path ${save_dir}/checkpoint_best.pt \
    --sample-break-mode complete \
    --max-tokens 3072 \
    --max-sentences 1 \
    --context-window 2560 \
    --softmax-batch 1024 \
