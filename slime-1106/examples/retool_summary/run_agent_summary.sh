#!/bin/bash
set -euxo pipefail

# Clean up existing processes to avoid port or resource conflicts
pkill -9 sglang || true
sleep 2
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 2

export PYTHONBUFFERED=16


SLIME_DIR=${SLIME_DIR:-/home/projects/polyullm/zhouqi/slime_kj/slime/}
EXAMPLE_DIR="${SLIME_DIR}/examples/retool_summary"
source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

# ====== proxy ======
export http_proxy=""
export https_proxy=""
# to avoid Ray/GCS connection timeout
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
HOST_IP=$(hostname -i | awk '{print $1}')
export NO_PROXY="localhost,127.0.0.1,${MASTER_ADDR},${HOST_IP},klb-dgx-*"
export no_proxy="${NO_PROXY}"

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}

# ====== Summary Router（SGLang，用于生成摘要）=====
export SUMMARY_AGENT_HOST=${SUMMARY_AGENT_HOST:-${MASTER_ADDR}}
export SUMMARY_AGENT_PORT=${SUMMARY_AGENT_PORT:-3333}

# reward model for llm-as-judge
export JUDGE_MODEL_BASE_URL=${JUDGE_MODEL_BASE_URL:-http://klb-dgx-049:8035/v1}
export JUDGE_MODEL_NAME=${JUDGE_MODEL_NAME:-qwen3-30b-a3b-thinking}
export JUDGE_MODEL_API_KEY=${JUDGE_MODEL_API_KEY:-EMPTY}

# ====== 训练/数据 ======
export KEY_SUFFIX=${KEY_SUFFIX:-exp_summary_$(date +%s)}


HF_CKPT_PATH=${HF_CKPT_PATH:-"/lustre/projects/polyullm/caishuo/cs_models/Qwen3-4B"}
DIST_CKPT_PATH=${DIST_CKPT_PATH:-"/lustre/projects/polyullm/caishuo/cs_models/Qwen3-4B_torch_dist"}


LOAD_DIR=/home/projects/polyullm/zhouqi/slime_kj/ckpt/summary-agent_test
SAVE_DIR=/home/projects/polyullm/zhouqi/slime_kj/ckpt/summary-agent_test


PROMPT_DATA_PATH=${PROMPT_DATA_PATH:-"${EXAMPLE_DIR}/example_data.jsonl"}

CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT_PATH}"
  --ref-load "${DIST_CKPT_PATH}"
  --load "${LOAD_DIR}"
  --save "${SAVE_DIR}"
  --save-interval 10
)

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA_PATH}"
  # --input-key prompt
  # --apply-chat-template
  --input-key problem
  --label-key answer
  --rollout-shuffle
  --num-rollout 200
  --rollout-batch-size 32
  --n-samples-per-prompt 8
  --rollout-max-response-len 4096
  --rollout-temperature 0.8
  --global-batch-size 32
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
  --adam-beta2 0.999
)


WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-agent_kj_dev
   --wandb-group slime_agent_summary
   --wandb-mode offline
   --wandb-dir /home/projects/polyullm/zhouqi/slime_kj/wandb
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.7
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

CUSTOM_ARGS=(
  --custom-generate-function-path agent_b_summary.generate
  --custom-rm-path agent_b_summary.reward_func
)

# starting ray
ray stop --force || true
ray start --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus 8 \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

export RAY_ADDRESS="http://${MASTER_ADDR}:8265"

# ====== 启动 SGLang Router（在 SUMMARY_AGENT_HOST:SUMMARY_AGENT_PORT）======
python -m sglang_router.launch_router \
  --host "${SUMMARY_AGENT_HOST}" \
  --port "${SUMMARY_AGENT_PORT}" \
  >/dev/null 2>&1 &

until nc -z "${SUMMARY_AGENT_HOST}" "${SUMMARY_AGENT_PORT}"; do
  echo "Waiting for SGLang router at ${SUMMARY_AGENT_HOST}:${SUMMARY_AGENT_PORT}..."
  sleep 1
done

# ====== 启动 FastAPI DB，绑定到 0.0.0.0方便retool agent能够访问 ======
cd "${EXAMPLE_DIR}"
export DATABASE_SERVER_IP=${DATABASE_SERVER_IP:-0.0.0.0}
bash start_database_server.sh >/dev/null 2>&1 &
cd "${SLIME_DIR}"
until nc -z "${MASTER_ADDR}" 18888; do
  echo "Waiting for Database at ${MASTER_ADDR}:18888..."
  sleep 1
done

# ray runtime env

RUNTIME_ENV_JSON="{
  \"working_dir\": \"${SLIME_DIR}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${EXAMPLE_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"KEY_SUFFIX\": \"${KEY_SUFFIX}\",
    \"DATABASE_SERVER_IP\": \"${MASTER_ADDR}\",
    \"JUDGE_MODEL_API_KEY\": \"${JUDGE_MODEL_API_KEY}\",
    \"JUDGE_MODEL_BASE_URL\": \"${JUDGE_MODEL_BASE_URL}\",
    \"JUDGE_MODEL_NAME\": \"${JUDGE_MODEL_NAME}\",
    \"SUMMARY_AGENT_PORT\": \"${SUMMARY_AGENT_PORT}\",
    \"http_proxy\": \"\",
    \"https_proxy\": \"\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"no_proxy\": \"${NO_PROXY}\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
    \"WANDB_MODE\": \"offline\",
    \"WANDB_DIR\": \"/home/projects/polyullm/zhouqi/slime_kj/wandb/\"
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
  ${PERF_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]} \
  ${CUSTOM_ARGS[@]}

