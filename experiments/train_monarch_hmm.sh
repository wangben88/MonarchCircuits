DATASET=text8
SEQ_LEN=256
CUDA_CORES=
DIR=../workspace/
HIDDEN_STATES_BLOCK_SIZE=524288_512_1024

# preprocess text8 dataset
python ../monarch_hmm/preprocess_text8.py --output_path ${DIR}/data

# train two dense hmms to initialize monarch hmm of hidden size 524288
bash train_dense_hmm.sh

# multiply dense hmms of hidden sizes 512 and 10124
python ../monarch_hmm/multiply_dense_hmm.py \
    --model_paths \
    ${DIR}/models/dense-hmm_text8_512/checkpoint-1700 \
    ${DIR}/models/dense-hmm_text8_1024/checkpoint-1700 \
    --output_path \
    ${DIR}/workspace/models/monarch-hmm_text8_${HIDDEN_STATES_BLOCK_SIZE}/checkpoint-0

# train monarch hmm
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=gpu \
    ../monarch_hmm/train_hmm.py \
    --init_model_path ${DIR}/workspace/models/monarch-hmm_text8_${HIDDEN_STATES_BLOCK_SIZE}/checkpoint-0 \
    --model_path ${DIR}/workspace/models/monarch-hmm_text8_${HIDDEN_STATES_BLOCK_SIZE}/ \
    --checkpoint 0 \
    --dataset ${DIR}/data/${DATASET}_chunk${SEQ_LEN} \
    --batch_size 128 \
    --grad_accum_iters 8 \
    --em_schedule "epoch,20,1.0,0.0,linear" \
    --log_file ${DIR}/logs/monarch-hmm_text8_${HIDDEN_STATES_BLOCK_SIZE}_log.txt \
    --eval_per_steps 100 \
    --monarch

# evaluate monarch hmm on test set, treating it as a one giant sequence; only supports single gpu
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=gpu \
    ../monarch_hmm/eval_hmm.py \
    --init_model_path ${DIR}/workspace/models/monarch-hmm_text8_${HIDDEN_STATES_BLOCK_SIZE}/checkpoint-1700 \
    --dataset ${DIR}/data/${DATASET}_chunk${SEQ_LEN} \
    --batch_size 128 # batch size affects evaluation results, the smaller the more accurate