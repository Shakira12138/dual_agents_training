#!/bin/bash
# 获取 Agent B 的节点 IP 的辅助脚本
# 使用方法：GET_AGENT_B_IP.sh <agent_b_job_id>

if [ $# -lt 1 ]; then
  echo "Usage: $0 <agent_b_job_id>"
  echo "Example: $0 12345"
  exit 1
fi

JOB_ID=$1

# 方法 1: 从 SLURM 获取节点名
NODELIST=$(squeue -j ${JOB_ID} -h -o "%N" 2>/dev/null)
if [ -z "${NODELIST}" ]; then
  # 如果作业已完成，从 sacct 获取
  NODELIST=$(sacct -j ${JOB_ID} --format=NodeList -n --noheader | head -n 1 | awk '{print $1}')
fi

if [ -n "${NODELIST}" ]; then
  # 获取第一个节点名
  FIRST_NODE=$(scontrol show hostnames "${NODELIST}" | head -n 1)

  if [ -n "${FIRST_NODE}" ]; then
    # 检查是否是 IP 地址格式
    if [[ "${FIRST_NODE}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      echo "${FIRST_NODE}"
      exit 0
    else
      # 节点名是主机名，需要获取 IP
      # 在计算节点上获取 IP（如果有访问权限）
      NODE_IP=$(srun --nodes=1 --ntasks=1 -w "${FIRST_NODE}" hostname -I | awk '{print $1}' 2>/dev/null)
      if [ -n "${NODE_IP}" ]; then
        echo "${NODE_IP}"
        exit 0
      fi
    fi
  fi
fi

# 方法 2: 从日志中提取（如果 Agent B 打印了 IP）
LOG_FILE="/home/projects/polyullm/zhouqi/slime_kj/logs/slime-rl-4node-${JOB_ID}.out"
if [ -f "${LOG_FILE}" ]; then
  # 从日志中提取 "Summary Agent Router will be at: <ip>:<port>"
  EXTRACTED_IP=$(grep "Summary Agent Router will be at:" "${LOG_FILE}" | head -n 1 | sed 's/.*at: \([0-9.]*\):.*/\1/')
  if [ -n "${EXTRACTED_IP}" ]; then
    echo "${EXTRACTED_IP}"
    exit 0
  fi
fi

echo "Error: Could not determine Agent B IP address" >&2
exit 1


