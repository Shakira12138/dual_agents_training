#!/bin/bash
# Training script for Agent A (Retool Agent) - Router-based connection
# This script connects to Agent B's router at SUMMARY_AGENT_HOST:SUMMARY_AGENT_PORT

set -euxo pipefail

# ====== Agent B Router 配置（必须在这里设置）=====
# 取消下面的注释并填写 Agent B的 地址和端口
SUMMARY_AGENT_HOST=10.127.128.154
SUMMARY_AGENT_PORT=3333  # 填写 Agent B 的 router 端口（默认 3333）
DATABASE_SERVER_IP=10.127.128.154
# ====== KEY_SUFFIX 配置（重要！）=====
# 确保这里设置的 KEY_SUFFIX 与 run_agent_summary_sbatch_router.sh 中的相同
# 这样两个 agent 才能共享同一个数据库队列
# 如果未设置，将从 run_agent_summary_sbatch_router.sh 的 .out 文件中找到
KEY_SUFFIX=exp_summary_router_1763799008
export PYTHONBUFFERED=16

SLIME_DIR=${SLIME_DIR:-/work/projects/polyullm/zhouqi/slime/slime-1106}
EXAMPLE_DIR="$SLIME_DIR/examples/retool_summary"
source "$SLIME_DIR/scripts/models/qwen3-4B.sh"

# Network related settings
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

# Build NO_PROXY
NO_PROXY_LIST="localhost,127.0.0.1,${MASTER_ADDR},${HOST_IP},${SUMMARY_AGENT_HOST:-},${DATABASE_SERVER_IP:-},kb3-a1-nv-dgx-*"
export NO_PROXY="${NO_PROXY:-${NO_PROXY_LIST}}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

# ====== Summary Agent Router 配置 =====
# 检查是否设置了 SUMMARY_AGENT_HOST（必须在脚本开头设置）
if [ -z "${SUMMARY_AGENT_HOST:-}" ]; then
  echo "Error: SUMMARY_AGENT_HOST is not set." >&2
  echo "Please uncomment and set SUMMARY_AGENT_HOST at the top of this script (around line 10-11)." >&2
  echo "Example: SUMMARY_AGENT_HOST=192.168.1.100" >&2
  exit 1
fi

export SUMMARY_AGENT_HOST=${DATABASE_SERVER_IP}
export SUMMARY_AGENT_PORT=${SUMMARY_AGENT_PORT:-3333}

echo "Summary Agent Router: ${SUMMARY_AGENT_HOST}:${SUMMARY_AGENT_PORT}"

# ====== Database Server 配置检查 =====
# 检查是否设置了 DATABASE_SERVER_IP（必须在脚本开头设置）
if [ -z "${DATABASE_SERVER_IP:-}" ]; then
  echo "Error: DATABASE_SERVER_IP is not set." >&2
  echo "Please uncomment and set DATABASE_SERVER_IP at the top of this script (around line 12-14)." >&2
  echo "You can find the Database Server IP in the .out file from run_agent_summary_sbatch_router.sh" >&2
  echo "Look for: 'Database Server will be at: <IP>:18888'" >&2
  echo "Example: DATABASE_SERVER_IP=10.153.1.53" >&2
  exit 1
fi

export DATABASE_SERVER_IP=${SUMMARY_AGENT_HOST}
export DATABASE_SERVER_PORT=${DATABASE_SERVER_PORT:-18888}

echo "Database Server: ${DATABASE_SERVER_IP}:${DATABASE_SERVER_PORT}"

# ====== 训练/数据 ======
HF_CKPT_PATH=${HF_CKPT_PATH:-"/work/projects/polyullm/zhouqi/models/qwen3-4b-sft-SGLang-RL"}
DIST_CKPT_PATH=${DIST_CKPT_PATH:-"/work/projects/polyullm/zhouqi/models/qwen3-4b-sft_torch_dist"}

LOAD_DIR=/work/projects/polyullm/zhouqi/dual_agent/retool-agent_test_23
SAVE_DIR=/work/projects/polyullm/zhouqi/dual_agent/retool-agent_test_23

PROMPT_DATA_PATH=${PROMPT_DATA_PATH:-"${EXAMPLE_DIR}/polaris.jsonl"}
#PROMPT_DATA_PATH="/work/projects/polyullm/caishuo/workspace/data/polaris-data-53K-indexed_c1c2c3.jsonl"
# indicate generate/reward function
export AGENT_A_IMPL_GENERATE_PATH=${AGENT_A_IMPL_GENERATE_PATH:-"agent_a_retool_full_loss_router.generate"}
export AGENT_A_IMPL_RM_PATH=${AGENT_A_IMPL_RM_PATH:-"agent_a_retool_full_loss_router.reward_func"}

CKPT_ARGS=(
   --hf-checkpoint "$HF_CKPT_PATH"
   --ref-load "$DIST_CKPT_PATH"
   --load "$LOAD_DIR"
   --save "$SAVE_DIR"
   --save-interval 10
   --rotary-base 5000000
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA_PATH}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --reward-key score

   --num-rollout 500
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1.0

   --global-batch-size 256
   --num-steps-per-rollout 1
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 4

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
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
   --wandb-group slime_agent_retool_router
   --wandb-mode offline
   --wandb-dir /work/projects/polyullm/zhouqi/slime/wandb/
)

# Agent A 使用自己的 router（默认启动，不指定 --sglang-router-ip/port）
# Agent A 通过 agent_a_retool_full_loss_router.py 中的代码直接调用 Agent B 的 router
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
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

EVAL_ARGS=(
   --eval-interval 30
   --eval-reward-key acc
   --eval-prompt-data aime24 /work/projects/polyullm/kejing/slime_workspace/data/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 30000
   --eval-top-p 0.7
)


RUNTIME_ENV_JSON="{
  \"working_dir\": \"${SLIME_DIR}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${EXAMPLE_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:3072\",
    \"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD\": \"1\",
    \"TORCH_NCCL_AVOID_RECORD_STREAMS\": \"1\",
    \"NVTE_ALLOW_NONDETERMINISTIC_ALGO\": \"1\",
    \"NVTE_DEBUG\": \"0\",
    \"MAX_TURNS\": \"${MAX_TURNS:-16}\",
    \"CONTEXT_LENGTH_THRESHOLD\": \"${CONTEXT_LENGTH_THRESHOLD:-2048}\",
    \"KEY_SUFFIX\": \"${KEY_SUFFIX:-exp_summary_router_$(date +%s)}\",
    \"DATABASE_SERVER_IP\": \"${DATABASE_SERVER_IP}\",
    \"SUMMARY_AGENT_HOST\": \"${SUMMARY_AGENT_HOST}\",
    \"SUMMARY_AGENT_PORT\": \"${SUMMARY_AGENT_PORT}\",
    \"http_proxy\": \"${http_proxy}\",
    \"https_proxy\": \"${https_proxy}\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"no_proxy\": \"${no_proxy}\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
    \"WANDB_MODE\": \"offline\",
    \"WANDB_DIR\": \"\/work\/projects\/polyullm\/zhouqi\/slime\/wandb\/\"
  }
}"

ray job submit --address="${RAY_ADDRESS}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 2 \
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
   ${EVAL_ARGS[@]} \
   ${CUSTOM_ARGS[@]} || {
   echo "Warning: ray job submit exited with non-zero code, but job may have succeeded." >&2
   echo "Check the .out log file for 'Job ... succeeded' message to confirm." >&2
   exit 0
}
