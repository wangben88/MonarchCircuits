DATASET=text8
SEQ_LEN=256
CUDA_CORES=0,1,2,3
DIR=../workspace/

python ../monarch_hmm/preprocess_text8.py --output_path ${DIR}/data

for HIDDEN_STATES_BLOCK_SIZE in 512 1024
do
    echo $HIDDEN_STATES_BLOCK_SIZE
    mkdir -p ${DIR}/models/dense-hmm_text8_${HIDDEN_STATES_BLOCK_SIZE}/
    CUDA_VISIBLE_DEVICES=$CUDA_CORES torchrun --standalone --nproc_per_node=gpu \
        ../monarch_hmm/train_hmm.py \
        --model_path ${DIR}/models/dense-hmm_text8_${HIDDEN_STATES_BLOCK_SIZE}/ \
        --hidden_states_block_size $HIDDEN_STATES_BLOCK_SIZE \
        --checkpoint 0 \
        --dataset ${DIR}/data/${DATASET}_chunk${SEQ_LEN} \
        --batch_size 1024 \
        --grad_accum_iters 1 \
        --em_schedule "epoch,20,1.0,0.0,linear" \
        --log_file ${DIR}/logs/dense-hmm_text8_${HIDDEN_STATES_BLOCK_SIZE}_log.txt \
        --eval_per_steps 100
done
