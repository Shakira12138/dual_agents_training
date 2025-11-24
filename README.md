<h1 align="center">
  <br>
  DualAgent Training Framework for Long-Context summarization
  <br>
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2508.05731"><img src="https://img.shields.io/badge/arXiv-Preprint-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv Paper"></a>
</p>

---

## 🌟 Overview

A fundamental challenge for tool-using language agents is handling **multi-step reasoning** over rich interaction histories, where the agent repeatedly calls tools and produces intermediate outputs before a final answer. Training such systems with **reinforcement learning (RL)** is difficult: trajectories are long, rewards are sparse and delayed, and full interaction histories can easily exceed the model’s context window, causing truncation, noisy credit assignment, and expensive evaluation of overlong outputs.

To address this problem, we propose **DualAgent**, a **dual-agent training framework** that separates *solving the task* from *summarizing the task so far*. An **Answering Agent** carries out multi-round reasoning and tool use, while a **Summarizing Agent** periodically compresses the ongoing interaction into a concise semantic summary once it becomes overlong. The Answering Agent then continues conditioned on this summary rather than the full history. For RL, we use only the final summary and its subsequent continuation as the conditioning context, yielding a **compact high-level representation** of long tool-use trajectories, alleviating context-length constraints, and **focusing the reward signal on semantically important decisions**.


---

## ⚙️ Setup

This section explains how to **prepare experiments, data and model** to start the code.

### 1. Model and Data Setup


### 2. Environment Setup

Recommended versions（按实际环境稍作调整即可）：

- Python ≥ 3.9  
- PyTorch ≥ 2.2  
- CUDA (optional, for GPU training)  

```bash
# 可选：创建虚拟环境
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

# 克隆仓库
git clone https://github.com/Shakira12138/dual_agents_training.git
cd dual_agents_training
sbatch --nodes=2  --job-name=summary  submit_2nodes.sh slime-1106/examples/retool_summary/run_agent_summary_sbatch_router.sh
sbatch --nodes=2  --job-name=retool  submit_2nodes.sh slime-1106/examples/retool_summary/run_agent_retool_sbatch_router.sh
```


---
##  📊 Results

