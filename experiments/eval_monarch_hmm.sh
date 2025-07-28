DATASET=lm1b

for HIDDEN_STATES_BLOCK_SIZE in 32768
do
    CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=gpu eval_hmm.py \
        --init_model_path ./workspace/models/hmm_${DATASET}_${HIDDEN_STATES_BLOCK_SIZE}/checkpoint-590 \
        --dataset ${DIR}/data/${DATASET}_chunk${SEQ_LEN} \
        --batch_size 128 # batch size affects evaluation results, the smaller the more accurate        
done

