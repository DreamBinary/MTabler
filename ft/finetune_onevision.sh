#!/bin/bash

export WANDB_MODE=offline

export OMP_NUM_THREADS=16
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^lo,docker,veth
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

cache_path="/bohr/cach-rxl3/v4/cache"
export HUGGINGFACE_HUB_CACHE=$cache_path
export HF_HOME=$cache_path

NUM_GPUS=4
NNODES=1
RANK=0
ADDR="localhost"
PORT=29500
MID_RUN_NAME="ov_7b_si_v1"
DATA_DIR="/bohr/data-oy2j/v2/data"


LLM_VERSION="lmms-lab/llava-onevision-qwen2-7b-si"
#LLM_VERSION="Qwen/Qwen2-7B-Instruct"

# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION="qwen_1_5"

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

CKPT_PATH=$LLM_VERSION # this could also be the previous stage checkpoint

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    train_mem.py \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 64  \
    --deepspeed scripts/zero2_offload.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path "${DATA_DIR}/gather_data/data.json" \
    --image_folder "${DATA_DIR}/gather_data/images" \
    --pretrain_mm_mlp_adapter="${DATA_DIR}/mm_projector.bin" \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "./checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --attn_implementation sdpa
