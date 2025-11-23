#!/bin/bash
# Training script for Agent B (Summary Agent) - Router-based connection
# This script starts an external router that Agent A can connect to for summary generation

set -euxo pipefail

# Ensure Ray head address is available (explicit local default)
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_ADDR

echo "MASTER_ADDR: ${MASTER_ADDR}"

RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
export RAY_ADDRESS=${RAY_ADDRESS:-"http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}"}

export PYTHONBUFFERED=16

SLIME_DIR=${SLIME_DIR:-/home/projects/polyullm/zhouqi/slime_kj/slime}
EXAMPLE_DIR="${SLIME_DIR}/examples/retool_summary"
source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

# ====== proxy ======
export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"

HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "${HOST_IP}" ]; then
  HOST_IP=$(hostname -i | awk '{print $1}')
fi

NO_PROXY_LIST="localhost,127.0.0.1,${MASTER_ADDR},${HOST_IP},klb-dgx-*"
export NO_PROXY="${NO_PROXY:-${NO_PROXY_LIST}}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}

# ====== Summary Router for sglang =====
# 配置 Agent B 自己的 router 地址（用于接收摘要请求）
# Agent A 将通过这个地址调用 Agent B 的 router
#
# 自动获取节点 IP（按优先级）：
# 1. 如果设置了 SUMMARY_AGENT_HOST 环境变量，使用它（手动设置优先）
# 2. 使用 hostname 获取的 IP（在容器内执行时，这通常能获取到正确的节点 IP）
# 3. 如果都失败，使用 MASTER_ADDR（通常不是实际 IP）
if [ -z "${SUMMARY_AGENT_HOST:-}" ]; then
  # 在容器内执行时，hostname 通常能返回节点的实际 IP
  # 优先使用从 hostname 获取的 IP
  if [ -n "${HOST_IP}" ]; then
    SUMMARY_AGENT_HOST="${HOST_IP}"
  elif [ "${MASTER_ADDR}" != "127.0.0.1" ] && [ "${MASTER_ADDR}" != "localhost" ]; then
    # 如果 MASTER_ADDR 不是本地地址，使用它
    SUMMARY_AGENT_HOST="${MASTER_ADDR}"
  else
    # 最后尝试使用 hostname -i
    SUMMARY_AGENT_HOST=$(hostname -i 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")
  fi
fi
export SUMMARY_AGENT_HOST
export SUMMARY_AGENT_IP=${SUMMARY_AGENT_IP:-${SUMMARY_AGENT_HOST}}
export SUMMARY_AGENT_PORT=${SUMMARY_AGENT_PORT:-3333}

echo "Summary Agent Router will be at: ${SUMMARY_AGENT_HOST}:${SUMMARY_AGENT_PORT}"

# ====== Database Server for offline training =====
# 配置数据库服务器地址（Agent A 会连接到这里）
# 数据库服务器和 Router 在同一台机器上，使用相同的 IP
if [ -z "${DATABASE_SERVER_IP:-}" ]; then
  DATABASE_SERVER_IP="${SUMMARY_AGENT_HOST}"
fi
export DATABASE_SERVER_IP
export DATABASE_SERVER_PORT=${DATABASE_SERVER_PORT:-18888}

# ====== 训练/数据 ======
# KEY_SUFFIX 用于标识数据库队列，两个 agent 必须使用相同的 KEY_SUFFIX
export KEY_SUFFIX=${KEY_SUFFIX:-exp_summary_router_$(date +%s)}

echo "=========================================="
echo "Database Server will be at: ${DATABASE_SERVER_IP}:${DATABASE_SERVER_PORT}"
echo "=========================================="
echo "IMPORTANT: Copy these values to run_agent_retool_sbatch_router.sh:"
echo "  DATABASE_SERVER_IP=${DATABASE_SERVER_IP}"
echo "  KEY_SUFFIX=${KEY_SUFFIX}"
echo "=========================================="

HF_CKPT_PATH=${HF_CKPT_PATH:-"/lustre/projects/polyullm/caishuo/cs_models/Qwen3-4B"}
DIST_CKPT_PATH=${DIST_CKPT_PATH:-"/lustre/projects/polyullm/caishuo/cs_models/Qwen3-4B_torch_dist"}

LOAD_DIR=${LOAD_DIR:-/home/projects/polyullm/zhouqi/slime_kj/ckpt/summary-agent_test3}
SAVE_DIR=${SAVE_DIR:-/home/projects/polyullm/zhouqi/slime_kj/ckpt/summary-agent_test3}

# reward model for llm-as-judge
export JUDGE_MODEL_BASE_URL=${JUDGE_MODEL_BASE_URL:-http://klb-dgx-003:8039/v1}
export JUDGE_MODEL_NAME=${JUDGE_MODEL_NAME:-qwen3-30b-a3b-thinking}
export JUDGE_MODEL_API_KEY=${JUDGE_MODEL_API_KEY:-EMPTY}

CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT_PATH}"
  --ref-load "${DIST_CKPT_PATH}"
  --load "${LOAD_DIR}"
  --save "${SAVE_DIR}"
  --save-interval 10
)
PROMPT_DATA_PATH=${PROMPT_DATA_PATH:-"${EXAMPLE_DIR}/example_data.jsonl"}
ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA_PATH}"
  --input-key problem
  --label-key answer
  --rollout-shuffle
  --num-rollout 200
  --rollout-batch-size 128
  --n-samples-per-prompt 8
  --rollout-max-response-len 4096
  --rollout-temperature 0.8
  --global-batch-size 1024
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
   --wandb-group slime_agent_summary_router
   --wandb-mode offline
   --wandb-dir /home/projects/polyullm/zhouqi/slime_kj/wandb/
)

# ====== 启动独立的 SGLang Router（用于摘要生成）=====
# 必须在容器内启动 router，因为需要访问容器内的 Python 环境
# 注意：router 必须在 slime 训练之前启动，且需要使用后台进程
# Agent A 将通过 SUMMARY_AGENT_HOST:SUMMARY_AGENT_PORT 连接到这个 router
echo "Starting Summary Agent Router at ${SUMMARY_AGENT_HOST}:${SUMMARY_AGENT_PORT}..."

# 检查 router 是否已经启动
if ! nc -z "${SUMMARY_AGENT_HOST}" "${SUMMARY_AGENT_PORT}" 2>/dev/null; then
  # 启动 router（后台运行）
  python -m sglang_router.launch_router \
    --host "${SUMMARY_AGENT_HOST}" \
    --port "${SUMMARY_AGENT_PORT}" \
    >/dev/null 2>&1 &

  ROUTER_PID=$!

  # 等待 router 启动完成
  echo "Waiting for Summary Agent Router to start..."
  max_attempts=30
  attempt=0
  while [ $attempt -lt $max_attempts ]; do
    if nc -z "${SUMMARY_AGENT_HOST}" "${SUMMARY_AGENT_PORT}" 2>/dev/null; then
      echo "Summary Agent Router ready at ${SUMMARY_AGENT_HOST}:${SUMMARY_AGENT_PORT}"
      break
    fi
    sleep 2
    attempt=$((attempt + 1))
    # 检查进程是否还在运行
    if ! kill -0 $ROUTER_PID 2>/dev/null; then
      echo "Warning: Router process died. Attempt: $attempt"
    fi
  done

  if [ $attempt -eq $max_attempts ]; then
    echo "Error: Router failed to start after $max_attempts attempts" >&2
    exit 1
  fi
else
  echo "Summary Agent Router already running at ${SUMMARY_AGENT_HOST}:${SUMMARY_AGENT_PORT}"
fi

# ====== 启动数据库服务器（用于 offline 训练数据存储）=====
# 检查数据库服务器是否已经启动
if ! nc -z "${DATABASE_SERVER_IP}" "${DATABASE_SERVER_PORT}" 2>/dev/null; then
  echo "Starting Database Server at ${DATABASE_SERVER_IP}:${DATABASE_SERVER_PORT}..."

  # 切换到示例目录启动数据库服务器
  cd "${EXAMPLE_DIR}"
  export DATABASE_SERVER_IP
  bash start_database_server.sh >/dev/null 2>&1 &
  DB_SERVER_PID=$!
  cd "${SLIME_DIR}"

  # 等待数据库服务器启动完成
  echo "Waiting for Database Server to start..."
  max_attempts=30
  attempt=0
  while [ $attempt -lt $max_attempts ]; do
    if nc -z "${DATABASE_SERVER_IP}" "${DATABASE_SERVER_PORT}" 2>/dev/null; then
      echo "Database Server ready at ${DATABASE_SERVER_IP}:${DATABASE_SERVER_PORT}"
      break
    fi
    sleep 2
    attempt=$((attempt + 1))
    # 检查进程是否还在运行
    if ! kill -0 $DB_SERVER_PID 2>/dev/null; then
      echo "Warning: Database Server process died. Attempt: $attempt"
    fi
  done

  if [ $attempt -eq $max_attempts ]; then
    echo "Error: Database Server failed to start after $max_attempts attempts" >&2
    exit 1
  fi
else
  echo "Database Server already running at ${DATABASE_SERVER_IP}:${DATABASE_SERVER_PORT}"
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 4
  --sglang-mem-fraction-static 0.5
  # 使用外部 router（Agent B 自己的 router）
  --sglang-router-ip "${SUMMARY_AGENT_HOST}"
  --sglang-router-port "${SUMMARY_AGENT_PORT}"
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

PRECISE_ARGS=(
   --transformer-impl transformer_engine
   --bf16
)

CUSTOM_ARGS=(
  --custom-generate-function-path agent_b_summary.generate
  --custom-rm-path agent_b_summary.reward_func
)

# Ray runtime env
# 注意：现在同时支持 router 模式（实时通信）和数据库模式（offline 训练）
RUNTIME_ENV_JSON="{
  \"working_dir\": \"${SLIME_DIR}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${EXAMPLE_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:2048\",
    \"KEY_SUFFIX\": \"${KEY_SUFFIX}\",
    \"DATABASE_SERVER_IP\": \"${DATABASE_SERVER_IP}\",
    \"SUMMARY_AGENT_HOST\": \"${SUMMARY_AGENT_HOST}\",
    \"SUMMARY_AGENT_PORT\": \"${SUMMARY_AGENT_PORT}\",
    \"JUDGE_MODEL_API_KEY\": \"${JUDGE_MODEL_API_KEY}\",
    \"JUDGE_MODEL_BASE_URL\": \"${JUDGE_MODEL_BASE_URL}\",
    \"JUDGE_MODEL_NAME\": \"${JUDGE_MODEL_NAME}\",
    \"http_proxy\": \"\",
    \"https_proxy\": \"\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"no_proxy\": \"${NO_PROXY}\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
    \"TORCH_SHOW_CPP_STACKTRACES\": \"1\",
    \"TRITON_DEBUG\": \"1\",
    \"WANDB_MODE\": \"offline\",
    \"WANDB_DIR\": \"/home/projects/polyullm/zhouqi/slime_kj/wandb/\"
  }
}"

ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-${SLURM_JOB_NUM_NODES:-2}}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-${SLURM_GPUS_PER_NODE:-8}}

ray job submit --address="${RAY_ADDRESS}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${DISTRIBUTED_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]} \
  ${PRECISE_ARGS[@]} \
  ${CUSTOM_ARGS[@]}

