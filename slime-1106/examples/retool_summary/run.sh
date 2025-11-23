#!/usr/bin/env bash
# Main launch script for retool-summary dual-agent training

set -euo pipefail

# ---------- User-friendly defaults (can be overridden by env) ---------- #
# You can change the DEFAULT_* values below once, then run this script directly
DEFAULT_SLIME_DIR=${DEFAULT_SLIME_DIR:-"/home/projects/polyullm/zhouqi/slime_kj/slime/"}
DEFAULT_KEY_SUFFIX=${DEFAULT_KEY_SUFFIX:-"exp_summary_$(date +%s)"}
DEFAULT_JUDGE_MODEL_BASE_URL=${DEFAULT_JUDGE_MODEL_BASE_URL:-"http://klb-dgx-049:8035/v1"}
DEFAULT_JUDGE_MODEL_NAME=${DEFAULT_JUDGE_MODEL_NAME:-"qwen3-30b-a3b-thinking"}
DEFAULT_JUDGE_MODEL_API_KEY=${DEFAULT_JUDGE_MODEL_API_KEY:-"EMPTY"}
DEFAULT_TRAIN_MODE=${DEFAULT_TRAIN_MODE:-"local"}

# Export final values (env overrides defaults)
export SLIME_DIR=${SLIME_DIR:-"$DEFAULT_SLIME_DIR"}
export KEY_SUFFIX=${KEY_SUFFIX:-"$DEFAULT_KEY_SUFFIX"}
export JUDGE_MODEL_BASE_URL=${JUDGE_MODEL_BASE_URL:-"$DEFAULT_JUDGE_MODEL_BASE_URL"}
export JUDGE_MODEL_NAME=${JUDGE_MODEL_NAME:-"$DEFAULT_JUDGE_MODEL_NAME"}
export JUDGE_MODEL_API_KEY=${JUDGE_MODEL_API_KEY:-"$DEFAULT_JUDGE_MODEL_API_KEY"}
export TRAIN_MODE=${TRAIN_MODE:-"$DEFAULT_TRAIN_MODE"}


# print all environment variables
echo "Environment variables:"
echo "SLIME_DIR: $SLIME_DIR"
echo "KEY_SUFFIX: $KEY_SUFFIX"
echo "JUDGE_MODEL_BASE_URL: $JUDGE_MODEL_BASE_URL"
echo "JUDGE_MODEL_NAME: $JUDGE_MODEL_NAME"
echo "JUDGE_MODEL_API_KEY: $JUDGE_MODEL_API_KEY"
echo "TRAIN_MODE: $TRAIN_MODE"

if [[ -z "${1:-}" ]]; then
  echo "Usage: $0 {agent-retool|agent-summary}" >&2
  exit 1
fi

run_type=$1

cd "$SLIME_DIR"

# Validate required environment variables (DATABASE_SERVER_IP can be set here or in summary branch we default to local IP)
if [[ -z "${DATABASE_SERVER_IP:-}" ]]; then
  echo "Warning: DATABASE_SERVER_IP is not set at entry. It will be set to local IP when starting agent-summary."
fi

if [[ -z "${KEY_SUFFIX:-}" ]]; then
  echo "KEY_SUFFIX environment variable must be set." >&2
  exit 1
fi

# Prepare log directory
LOG_DATE="$(date +%m%d)"
LOG_DIR="/home/projects/polyullm/zhouqi/slime_kj/logs/${LOG_DATE}"
mkdir -p "$LOG_DIR"

case "$run_type" in
  agent-retool|agent-a)
    echo "Starting agent-retool (Retool Agent) training..."
    # Retool agent now launches its own SGLang router; only DB/IP and KEY_SUFFIX need to align
    # Run agent-retool training
    bash examples/retool_summary/run_agent_retool.sh 2>&1 | tee "$LOG_DIR/${KEY_SUFFIX}_agent_retool.log"
    ;;

  agent-summary|agent-b)
    echo "Starting agent-summary (Summary Agent) training..."

    if [[ -z "${JUDGE_MODEL_API_KEY:-}" ]]; then
      echo "JUDGE_MODEL_API_KEY environment variable must be set for Agent Summary." >&2
      exit 1
    fi

    # Use loopback in local mode to align with run_agent_summary.sh (MASTER_ADDR=127.0.0.1)
    export DATABASE_SERVER_IP=${DATABASE_SERVER_IP:-127.0.0.1}
    export SUMMARY_AGENT_PORT=${SUMMARY_AGENT_PORT:-3333}
    # Let inner script manage DB + Ray head startup and readiness
    bash examples/retool_summary/run_agent_summary.sh 2>&1 | tee "$LOG_DIR/${KEY_SUFFIX}_agent_summary.log"
    ;;

  *)
    echo "Unknown type: $run_type" >&2
    echo "Usage: $0 {agent-retool|agent-summary}" >&2
    exit 1
    ;;
esac

echo "Training for $run_type completed!"


