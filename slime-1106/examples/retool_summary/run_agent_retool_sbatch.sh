#!/bin/bash
# Training script for Agent A (Retool Agent)

set -euxo pipefail


export PYTHONBUFFERED=16

SLIME_DIR=${SLIME_DIR:-/home/projects/polyullm/zhouqi/slime_kj/slime}
EXAMPLE_DIR="$SLIME_DIR/examples/retool_summary"
source "$SLIME_DIR/scripts/models/qwen3-4B.sh"


# Network related settings
# In multi-node sbatch mode, MASTER_ADDR should be set to head node IP
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
echo "MASTER_ADDR: ${MASTER_ADDR}"

RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
export RAY_ADDRESS=${RAY_ADDRESS:-"http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}"}


# ====== proxy ======
export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}

# Get host IP
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "${HOST_IP}" ]; then
  HOST_IP=$(hostname -i | awk '{print $1}')
fi

# Build NO_PROXY to avoid proxying local cluster endpoints
NO_PROXY_LIST="localhost,127.0.0.1,${MASTER_ADDR},${HOST_IP},${DATABASE_SERVER_IP:-},klb-dgx-*"
export NO_PROXY="${NO_PROXY:-${NO_PROXY_LIST}}"
export no_proxy="${no_proxy:-${NO_PROXY}}"



# ====== 训练/数据 ======

HF_CKPT_PATH=${HF_CKPT_PATH:-"/lustre/projects/polyullm/caishuo/cs_models/Qwen3-4B"}
DIST_CKPT_PATH=${DIST_CKPT_PATH:-"/lustre/projects/polyullm/caishuo/cs_models/Qwen3-4B_torch_dist"}


LOAD_DIR=/home/projects/polyullm/zhouqi/slime_kj/ckpt/retool-agent_test_2
SAVE_DIR=/home/projects/polyullm/zhouqi/slime_kj/ckpt/retool-agent_test_2


PROMPT_DATA_PATH=${PROMPT_DATA_PATH:-"${EXAMPLE_DIR}/dapo-math-17k.jsonl"}


# indicate generate/reward function
export AGENT_A_IMPL_GENERATE_PATH=${AGENT_A_IMPL_GENERATE_PATH:-"agent_a_retool_full_loss.generate"}
export AGENT_A_IMPL_RM_PATH=${AGENT_A_IMPL_RM_PATH:-"agent_a_retool_full_loss.reward_func"}



# Database/key settings (must match summary agent)
# Require DATABASE_SERVER_IP to avoid mistakenly using localhost on multi-node
if [ -z "${DATABASE_SERVER_IP:-}" ]; then
  echo "Error: DATABASE_SERVER_IP is not set. Set it to the Summary node DB host/IP." >&2
  exit 1
fi


CKPT_ARGS=(
   --hf-checkpoint "$HF_CKPT_PATH"
   --ref-load "$DIST_CKPT_PATH"
   --load "$LOAD_DIR"
   --save "$SAVE_DIR"
   --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA_PATH}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --reward-key score

   --num-rollout 100
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1.0

   --global-batch-size 256
   --num-steps-per-rollout 1
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-agent_kj_dev
   --wandb-group slime_agent_retool
   --wandb-mode offline
   --wandb-dir /home/projects/polyullm/zhouqi/slime_kj/wandb/
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.5
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path ${AGENT_A_IMPL_GENERATE_PATH}
   --custom-rm-path ${AGENT_A_IMPL_RM_PATH}
)


RUNTIME_ENV_JSON="{
  \"working_dir\": \"${SLIME_DIR}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${EXAMPLE_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:2048\",
    \"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD\": \"1\",
    \"TORCH_NCCL_AVOID_RECORD_STREAMS\": \"1\",
    \"NVTE_ALLOW_NONDETERMINISTIC_ALGO\": \"1\",
    \"NVTE_DEBUG\": \"0\",
    \"MAX_TURNS\": \"${MAX_TURNS:-16}\",
    \"CONTEXT_LENGTH_THRESHOLD\": \"${CONTEXT_LENGTH_THRESHOLD:-4096}\",
    \"KEY_SUFFIX\": \"${KEY_SUFFIX}\",
    \"DATABASE_SERVER_IP\": \"${DATABASE_SERVER_IP}\",
    \"http_proxy\": \"${http_proxy}\",
    \"https_proxy\": \"${https_proxy}\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"no_proxy\": \"${no_proxy}\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
    \"WANDB_MODE\": \"offline\",
    \"WANDB_DIR\": \"\/home\/projects\/polyullm\/zhouqi\/slime_kj\/wandb\/\"
  }
}"

ray job submit --address="${RAY_ADDRESS}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}



