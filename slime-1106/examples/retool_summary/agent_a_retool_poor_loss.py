"""
Module: agent_a_retool_poor_loss
---------------------------------
穷人版 Loss 实现：直接用 Summary 替换早期推理

核心思想：
- 用 summary 直接替换早期的 retool 步骤
- 减少训练的 token 数量，节省计算

Rollout 序列：
prompt + summary + summary + retool + answer
  0    +    1     +    1     +   1    +   1

对比完整版：
- 完整版：prompt + retool + summary + retool + summary + retool + answer
- 穷人版：prompt + summary + summary + retool + answer （省略早期 retool）

优点：
1. 计算量更小（训练 token 更少）
2. 可以处理更长的推理链
3. 适合资源受限场景

缺点：
1. 可能丢失早期推理细节
2. Summary 质量要求更高
"""

import re
import uuid
import asyncio
from typing import Any, Dict, List, Tuple

from jinja2 import Template

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

try:
    from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError:
    raise ImportError("MathDapo is not installed")

from config import global_config
from tool_sandbox import TOOL_CONFIGS, tool_registry
from database_utils import commit_summary_data, fetch_completed_summary

# Tool template (same as before)
TOOL_TEMPLATE = """<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful assistant.
{%- endif %}
{%- if tools %}
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{- tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{- message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""


def format_conversation_with_tools(
    prompt: str,
    tools: List[Dict[str, Any]] = None,
    system_prompt: str = None,
    messages: List[Dict[str, Any]] = None
) -> str:
    """Format conversation using Jinja2 template with tool support"""
    template = Template(TOOL_TEMPLATE)
    
    messages_to_render = []
    
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = (
            "You are a helpful assistant that can use Python "
            "tools to solve mathematical problems. When you need "
            "to perform calculations, use the code_interpreter "
            "tool to execute code and get results."
        )
    
    messages_to_render.append({"role": "system", "content": system_content})
    
    if prompt:
        messages_to_render.append({"role": "user", "content": prompt})
    
    if messages:
        messages_to_render.extend(messages)
    
    formatted_text = template.render(messages=messages_to_render, tools=tools or [])
    return formatted_text


def postprocess_predictions(prediction: str):
    """Extract action and content from prediction"""
    answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    answer_match = re.search(answer_pattern, prediction, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        return "answer", content
    
    tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, prediction, re.DOTALL)
    if tool_call_match:
        try:
            import json
            json_str = tool_call_match.group(1).replace("\n", "\\n")
            tool_call_data = json.loads(json_str)
            tool_name = tool_call_data.get("name")
            arguments = tool_call_data.get("arguments", {})
            
            if tool_name == "code_interpreter":
                code = arguments.get("code", "")
                if code.strip():
                    return "code", code
        except (json.JSONDecodeError, KeyError, AttributeError):
            pass
    
    return None, ""


def postprocess_responses(resp: str) -> str:
    """Post-process response to ensure tag completeness"""
    if "<tool_call>" in resp:
        tool_call_pattern = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
        matches = list(re.finditer(tool_call_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]
    
    if "Answer:" in resp and "\\boxed{" in resp:
        answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
        matches = list(re.finditer(answer_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]
    
    return resp


async def execute_predictions(prediction: str) -> tuple[str, bool]:
    """Execute predictions and return results"""
    action, content = postprocess_predictions(prediction)
    
    if action == "code":
        code = content.strip()
        if code:
            result = await tool_registry.execute_tool("code_interpreter", {"code": code})
            next_obs = f"\n\n<interpreter>\n{result}\n</interpreter>\n\n"
            done = False
        else:
            next_obs = "\n\n<interpreter>\nError: No Python code found\n</interpreter>\n\n"
            done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = (
            "\nMy previous action is invalid. "
            "If I want to execute code, I should use <tool_call> tags. "
            "If I want to give the final answer, I should use the format "
            "'Answer: \\boxed{answer}'. Let me try again.\n"
        )
        done = False
    
    return next_obs, done


async def request_and_replace_with_summary(
    prompt: str,
    response: str,
    tokenizer,
    timeout: int = 60
) -> Tuple[str, str, List[int], List[int], bool]:
    """
    穷人版：用 Summary 完全替换早期推理
    
    策略：
    1. 请求 summary
    2. 丢弃当前的 response
    3. 用 summary 作为新的起点
    
    Returns:
        (new_prompt, new_response, new_response_token_ids, new_loss_masks, summarized)
    """
    full_text = prompt + response
    token_count = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
    
    if token_count < global_config.CONTEXT_LENGTH_THRESHOLD:
        return prompt, response, None, None, False
    
    print(f"[Agent A Poor Loss] Context {token_count} exceeds {global_config.CONTEXT_LENGTH_THRESHOLD}")
    
    # Submit summary request
    task_id = str(uuid.uuid4())
    summary_data = {
        "original_prompt": prompt,
        "conversation_history": [{"role": "assistant", "content": response}],
        "token_count": token_count,
    }
    commit_summary_data(task_id, summary_data)
    print(f"[Agent A Poor Loss] Submitted summary request: {task_id}")
    
    # Wait for summary
    summary_text = fetch_completed_summary(task_id, timeout=timeout)
    
    if not summary_text:
        print(f"[Agent A Poor Loss] Timeout, continuing without summary")
        return prompt, response, None, None, False
    
    # ✨ 关键：完全替换，而不是追加
    new_response = (
        f"\n[Previous reasoning summarized by AI]\n"
        f"{summary_text}\n\n"
        f"[Now let me solve this step by step]\n"
    )
    
    # Tokenize new response
    new_response_token_ids = tokenizer(new_response, add_special_tokens=False)["input_ids"]
    
    # ✅ Summary 部分全部参与训练
    new_loss_masks = [1] * len(new_response_token_ids)
    
    new_token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) + len(new_response_token_ids)
    compression_ratio = token_count / max(new_token_count, 1)
    
    print(f"[Agent A Poor Loss] Replaced with summary: {token_count} -> {new_token_count} tokens ({compression_ratio:.2f}x)")
    
    return prompt, new_response, new_response_token_ids, new_loss_masks, True


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """
    穷人版生成函数：用 Summary 替换早期推理
    
    策略：
    1. 生成 5 轮推理
    2. 如果超过阈值，请求 summary 并替换
    3. 从 summary 继续生成
    4. 重复直到完成
    
    Loss mask 策略：
    - Prompt: 0（不训练）
    - Summary: 1（训练）✅
    - New retool: 1（训练）✅
    - Tool output: 0（不训练）
    - Final answer: 1（训练）✅
    """
    assert not args.partial_rollout, "Partial rollout is not supported"
    
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    
    # Initialize
    tool_specs = tool_registry.get_tool_specs()
    prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)
    
    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0
    summarization_count = 0
    
    for turn in range(TOOL_CONFIGS["max_turns"]):
        # Check if we should replace with summary (every 5 turns)
        if turn > 0 and turn % 5 == 0:
            new_prompt, new_response, new_response_token_ids, new_loss_masks, summarized = \
                await request_and_replace_with_summary(
                    prompt, response, state.tokenizer, timeout=60
                )
            
            if summarized:
                summarization_count += 1
                # ✨ 完全替换
                response = new_response
                response_token_ids = new_response_token_ids
                loss_masks = new_loss_masks
                
                print(f"[Agent A Poor Loss] Summary {summarization_count} applied, restarting from compressed context...")
        
        # Generate response
        payload = {
            "text": prompt + response,
            "sampling_params": sampling_params,
        }
        
        output = await post(url, payload)
        
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            sample.reward = global_config.DEFAULT_SCORE
            return sample
        
        cur_response = output["text"]
        cur_response = postprocess_responses(cur_response)
        
        # Tokenize new generation
        cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]
        response += cur_response
        response_token_ids += cur_response_token_ids
        
        # ✅ 新生成的内容参与训练
        loss_masks += [1] * len(cur_response_token_ids)
        
        if output["meta_info"]["finish_reason"]["type"] == "length":
            sample.status = Sample.Status.TRUNCATED
            break
        
        # Execute prediction
        next_obs, done = await execute_predictions(cur_response)
        if done:
            sample.status = Sample.Status.COMPLETED
            break
        
        if "<interpreter>" in next_obs:
            tool_call_count += 1
        
        # Add tool output
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        
        # ❌ Tool output 不参与训练
        loss_masks += [0] * len(obs_tokens_ids)
        
        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            sample.status = Sample.Status.COMPLETED
            break
    
    # Set sample attributes
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks
    sample.tool_call_count = tool_call_count
    sample.summarization_count = summarization_count
    
    # Validation
    assert len(sample.tokens) == len(prompt_tokens_ids) + len(response_token_ids), "Token count mismatch"
    assert len(sample.loss_mask) == len(response_token_ids), f"Loss mask mismatch: {len(sample.loss_mask)} vs {len(response_token_ids)}"
    
    # Statistics
    total_trainable = sum(loss_masks)
    total_tokens = len(loss_masks)
    trainable_ratio = total_trainable / max(total_tokens, 1)
    
    print(f"[Agent A Poor Loss] Generation complete:")
    print(f"  - Total tokens: {total_tokens}")
    print(f"  - Trainable tokens: {total_trainable} ({trainable_ratio:.1%})")
    print(f"  - Tool calls: {tool_call_count}")
    print(f"  - Summarizations: {summarization_count}")
    print(f"  - Compression ratio: {trainable_ratio:.1%} of original")
    
    if not hasattr(sample, 'status') or sample.status is None:
        sample.status = Sample.Status.COMPLETED
    
    return sample


async def reward_func(args, sample, **kwargs):
    """Reward function (same as before)"""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")
    
    solution_str = sample.prompt + sample.response
    ground_truth = sample.label if sample.label is not None else ""
    num_turns = getattr(sample, "tool_call_count", 0)
    
    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)
    
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)
    
    # Bonus for efficient summarization
    if hasattr(sample, 'summarization_count') and sample.summarization_count > 0:
        result["score"] += 0.1 * sample.summarization_count  # Higher bonus for poor version
    
    if result["pred"] is None:
        result["pred"] = ""
    
    return result

