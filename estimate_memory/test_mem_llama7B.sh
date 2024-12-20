#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=/home/xueh/projects/simulation/llm_training_tools/Megatron-LM:$PYTHONPATH

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --seq-length $6 \
    --max-position-embeddings $6 \
    --micro-batch-size $7 \
    --global-batch-size $8 \
    --lr 0.00015 \
    --train-iters 100 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --disable-bias-linear \
    --bf16 \
    --fp8-format hybrid \
    --fp8-init \
"
    
    # --optimizer-dtype bf16 \
    # --gradient-dtype bf16 \

# we can set both tp, pp, dp, cp to estimate different num of gpu
PARALLEL_ARGS="
    --tensor-model-parallel-size $1 \
    --pipeline-model-parallel-size $2 \
    --context-parallel-size $3 \
    --data-parallel-size $4 \
    --recompute-granularity selective \
    --use-distributed-optimizer \
"
SP=$5
if [[ $SP -eq 1 ]]; then
    PARALLEL_ARGS+=" --sequence-parallel "
fi

DATA_ARGS="
    --mock-data \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model /home/xueh/model/llama2-7b-hf/tokenizer.model \
    --vocab-size 32000 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 5000000	\
    --eval-interval 10000 \
    --eval-iters 1
"

python estimate_memory.py \
    $GPT_ARGS \
    $PARALLEL_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl
