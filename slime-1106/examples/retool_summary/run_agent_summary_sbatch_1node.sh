#!/bin/bash
set -euxo pipefail

# Ensure Ray head address is available (explicit local default)
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_ADDR


# echo master addr
echo "MASTER_ADDR: ${MASTER_ADDR}"


RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
export RAY_ADDRESS=${RAY_ADDRESS:-"http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}"}

# will prevent ray from buffering stdout/stderr
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

# # ====== Summary Router for sglang =====
# export SUMMARY_AGENT_HOST=${SUMMARY_AGENT_HOST:-${MASTER_ADDR}}
# export SUMMARY_AGENT_IP=${SUMMARY_AGENT_IP:-${SUMMARY_AGENT_HOST}}
# export SUMMARY_AGENT_PORT=${SUMMARY_AGENT_PORT:-3333}


# ====== 训练/数据 ======

export KEY_SUFFIX=${KEY_SUFFIX:-exp_summary_$(date +%s)}

HF_CKPT_PATH=${HF_CKPT_PATH:-"/lustre/projects/polyullm/caishuo/cs_models/Qwen3-4B"}
DIST_CKPT_PATH=${DIST_CKPT_PATH:-"/lustre/projects/polyullm/caishuo/cs_models/Qwen3-4B_torch_dist"}

# checkpoint path
LOAD_DIR=${LOAD_DIR:-/home/projects/polyullm/zhouqi/slime_kj/ckpt/summary-agent_test}
SAVE_DIR=${SAVE_DIR:-/home/projects/polyullm/zhouqi/slime_kj/ckpt/summary-agent_test}


# dummy example, the summary agent fetch data from db
PROMPT_DATA_PATH=${PROMPT_DATA_PATH:-"${EXAMPLE_DIR}/example_data.jsonl"}


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

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA_PATH}"
  --input-key problem
  --label-key answer
  --rollout-shuffle
  --num-rollout 100
  --rollout-batch-size 8
  --n-samples-per-prompt 8
  --rollout-max-response-len 4096
  --rollout-temperature 0.8
  --global-batch-size 64
  --balance-data
)

PERF_ARGS=(
  --tensor-model-parallel-size 1
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --no-check-for-nan-in-loss-and-grad
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
   --wandb-group slime_agent_summary
   --wandb-mode offline
   --wandb-dir /home/projects/polyullm/zhouqi/slime_kj/wandb/
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.3
  # --sglang-router-ip "${SUMMARY_AGENT_HOST}"
  # --sglang-router-port "${SUMMARY_AGENT_PORT}"
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
   #--fp8-format e4m3
   #--fp8-recipe blockwise
   #--fp8-param-gather
   # --direct-update-fp8-weight
)


CUSTOM_ARGS=(
  --custom-generate-function-path agent_b_summary.generate
  --custom-rm-path agent_b_summary.reward_func
)

cleanup() {
  if [ -n "${DB_PID:-}" ] && kill -0 "${DB_PID}" 2>/dev/null; then
    kill "${DB_PID}" || true
    wait "${DB_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ====== 启动 FastAPI DB，绑定到 hostname 或 0.0.0.0 ======
# Prefer hostname IP if available; otherwise bind all interfaces
DB_BIND_IP=${DATABASE_SERVER_BIND_IP:-${HOST_IP}}
if [ -z "${DB_BIND_IP}" ]; then
  DB_BIND_IP="0.0.0.0"
fi
pushd "${EXAMPLE_DIR}" >/dev/null
DATABASE_SERVER_IP="${DB_BIND_IP}" bash start_database_server.sh >/dev/null 2>&1 &
DB_PID=$!
popd >/dev/null

# Choose an address to check readiness: prefer HOST_IP, then MASTER_ADDR, then localhost
DB_CHECK_IP="${HOST_IP}"
if [ -z "${DB_CHECK_IP}" ]; then
  DB_CHECK_IP="${MASTER_ADDR}"
fi
if [ -z "${DB_CHECK_IP}" ]; then
  DB_CHECK_IP="127.0.0.1"
fi
until nc -z "${DB_CHECK_IP}" 18888; do
  echo "Waiting for Database at ${DB_CHECK_IP}:18888..."
  sleep 1
done

# Address exposed to remote clients: prefer HOST_IP, fallback MASTER_ADDR
CLIENT_DATABASE_IP=${CLIENT_DATABASE_IP:-${HOST_IP}}
if [ -z "${CLIENT_DATABASE_IP}" ]; then
  CLIENT_DATABASE_IP="${MASTER_ADDR}"
fi
export DATABASE_SERVER_IP="${CLIENT_DATABASE_IP}"

# ray runtime env

RUNTIME_ENV_JSON="{
  \"working_dir\": \"${SLIME_DIR}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${EXAMPLE_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"KEY_SUFFIX\": \"${KEY_SUFFIX}\",
    \"DATABASE_SERVER_IP\": \"${CLIENT_DATABASE_IP}\",
    \"JUDGE_MODEL_API_KEY\": \"${JUDGE_MODEL_API_KEY}\",
    \"JUDGE_MODEL_BASE_URL\": \"${JUDGE_MODEL_BASE_URL}\",
    \"JUDGE_MODEL_NAME\": \"${JUDGE_MODEL_NAME}\",
    \"http_proxy\": \"\",
    \"https_proxy\": \"\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"no_proxy\": \"${NO_PROXY}\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
    \"WANDB_MODE\": \"offline\",
    \"WANDB_DIR\": \"/home/projects/polyullm/zhouqi/slime_kj/wandb/\"
  }
}"


ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-${SLURM_JOB_NUM_NODES:-1}}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-${SLURM_GPUS_PER_NODE:-8}}

ray job submit --address="${RAY_ADDRESS}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --colocate \
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
  ${CUSTOM_ARGS[@]}




