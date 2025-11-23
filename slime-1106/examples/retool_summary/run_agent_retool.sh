#!/bin/bash
# Training script for Agent A (Retool Agent)

# Clean up existing processes to avoid port or resource conflicts
pkill -9 sglang || true
sleep 2
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 2

set -ex
export PYTHONBUFFERED=16

SLIME_DIR=${SLIME_DIR:-/home/projects/polyullm/zhouqi/slime_kj/slime/}
EXAMPLE_DIR="$SLIME_DIR/examples/retool_summary"
source "$SLIME_DIR/scripts/models/qwen3-4B.sh"

export TRAIN_MODE=${TRAIN_MODE:-"local"}

# network related setting
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8365}
export DATABASE_SERVER_IP=klb-dgx-006
# proxy and network interface settings (align with working setup)
export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}

# Build NO_PROXY to avoid proxying local cluster endpoints
_HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
export NO_PROXY="localhost,127.0.0.1,${MASTER_ADDR},${_HOST_IP},${DATABASE_SERVER_IP},${RETOOL_AGENT_HOST},klb-dgx-*"
export no_proxy="${NO_PROXY}"

# indicate generate/reward function
# Options:
#   agent_a_retool.py           (legacy)
#   agent_a_retool_v2.py        (summary-aware generation)
#   agent_a_retool_full_loss.py (train on summary tokens)
#   agent_a_retool_poor_loss.py (cheap loss)
export AGENT_A_IMPL_GENERATE_PATH=${AGENT_A_IMPL_GENERATE_PATH:-"agent_a_retool_v2.generate"}
export AGENT_A_IMPL_RM_PATH=${AGENT_A_IMPL_RM_PATH:-"agent_a_retool_v2.reward_func"}

# Set environment variables based on TRAIN_MODE
if [ "$TRAIN_MODE" = "local" ]; then
   : # RAY will be started below; RAY_ADDRESS set after start
else
   if [ -z "$RAY_ADDRESS" ]; then
       echo "Error: RAY_ADDRESS environment variable is not set" && exit 1
   fi
fi

# ====== 训练/数据 ======

HF_CKPT_PATH=${HF_CKPT_PATH:-"/lustre/projects/polyullm/caishuo/cs_models/Qwen3-4B"}
DIST_CKPT_PATH=${DIST_CKPT_PATH:-"/lustre/projects/polyullm/caishuo/cs_models/Qwen3-4B_torch_dist"}


LOAD_DIR=/home/projects/polyullm/zhouqi/slime_kj/ckpt/retool-agent_test
SAVE_DIR=/home/projects/polyullm/zhouqi/slime_kj/ckpt/retool-agent_test


PROMPT_DATA_PATH=${PROMPT_DATA_PATH:-"${EXAMPLE_DIR}/dapo-math-17k.jsonl"}

# ====== Retool Agent 的推理后端（独立的 SGLang Router）=====
DEFAULT_NODE_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
export RETOOL_AGENT_HOST=${RETOOL_AGENT_HOST:-${DEFAULT_NODE_IP:-${MASTER_ADDR}}}
export RETOOL_AGENT_PORT=${RETOOL_AGENT_PORT:-3334}

# Database/key settings (must match summary agent)
# Require DATABASE_SERVER_IP to avoid mistakenly using localhost on multi-node
if [ -z "${DATABASE_SERVER_IP:-}" ]; then
  echo "Error: DATABASE_SERVER_IP is not set. Set it to the Summary node DB host/IP." >&2
  exit 1
fi
export KEY_SUFFIX=${KEY_SUFFIX:-exp_summary_$(date +%s)}

# Allow overriding rollout parameters via env (to reduce load and avoid KV OOM)
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-2048}


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


   --num-epoch 1
   --num-rollout 100
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 4096
   --rollout-temperature 1.0

   --global-batch-size 64
   --num-steps-per-rollout 1
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --no-check-for-nan-in-loss-and-grad
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
   --wandb-dir /home/projects/polyullm/zhouqi/slime_kj/slime/wandb
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
   --sglang-router-ip "${RETOOL_AGENT_HOST}"
   --sglang-router-port "${RETOOL_AGENT_PORT}"
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

# Launch Ray in local mode
if [ "$TRAIN_MODE" = "local" ]; then
   ray start --head \
      --node-ip-address "${MASTER_ADDR}" \
      --num-gpus 8 \
      --disable-usage-stats \
      --dashboard-host=0.0.0.0 \
      --dashboard-port="${RAY_DASHBOARD_PORT}"
   export RAY_ADDRESS="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}"

   # SGLang Router (Retool only)
   python -m sglang_router.launch_router \
      --host "${RETOOL_AGENT_HOST}" \
      --port "${RETOOL_AGENT_PORT}" \
      >/dev/null 2>&1 &

   # Wait for the Retool SGLang router
   until nc -z "${RETOOL_AGENT_HOST}" "${RETOOL_AGENT_PORT}"; do
     echo "Waiting for the Retool SGLang router at ${RETOOL_AGENT_HOST}:${RETOOL_AGENT_PORT}..."
     sleep 1
   done
fi

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
    \"CONTEXT_LENGTH_THRESHOLD\": \"${CONTEXT_LENGTH_THRESHOLD:-6144}\",
    \"KEY_SUFFIX\": \"${KEY_SUFFIX}\",
    \"DATABASE_SERVER_IP\": \"${DATABASE_SERVER_IP}\",
    \"SUMMARY_AGENT_IP\": \"${RETOOL_AGENT_HOST}\",
    \"SUMMARY_AGENT_PORT\": \"${RETOOL_AGENT_PORT}\",
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


