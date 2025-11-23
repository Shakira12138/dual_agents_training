"""
Module: agent_a_retool_full_loss
---------------------------------
完整版 Loss 实现：让 Retool Agent 学习如何使用 Summary

核心思想：
- Retool 部分：训练（loss_mask=1）
- Summary 部分：也训练（loss_mask=1）✨ 关键改进
- Tool outputs：不训练（loss_mask=0）

Rollout 序列：
prompt + retool + summary + retool + summary + retool + answer
  0   +   1    +    1     +   1    +    1     +   1    +   1

这样 Retool Agent 会学会：
1. 如何生成有效的推理步骤
2. 如何理解和使用 summary
3. 如何在 summary 后继续推理
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

# Tool template
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


async def request_and_apply_summary_full_loss(
    prompt: str,
    response: str,
    response_token_ids: List[int],
    loss_masks: List[int],
    tokenizer,
    timeout: int = 400,
    max_summary_rounds: int = 8,
) -> Tuple[str, str, str, List[int], List[int], bool]:

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    summarized = False
    round_idx = 0

    # 初始 token 数
    full_text = prompt + response
    token_count = count_tokens(full_text)
    final_summary_text = ""

    # 如果太长，就开始循环 summary
    while token_count > global_config.CONTEXT_LENGTH_THRESHOLD and round_idx < max_summary_rounds:
        round_idx += 1
        summarized = True

        print(f"[Agent A Full Loss] Round {round_idx}: Context {token_count} > {global_config.CONTEXT_LENGTH_THRESHOLD}, submitting summary")
        #print("before summary:", response)

        # --- 提交 summary 请求 ---
        task_id = str(uuid.uuid4())
        summary_data = {
            "original_prompt": prompt,
            "conversation_history": [{"role": "assistant", "content": response}],
            "token_count": token_count,
            "round": round_idx,
        }
        commit_summary_data(task_id, summary_data)

        # --- 等待 summary ---
        summary_text = fetch_completed_summary(task_id, timeout=timeout)
        if not summary_text:
            print(f"[Agent A Full Loss] Round {round_idx} timeout or empty summary, stop summarizing")
            break

        # 生成新的 summary 包装文本
        summary_prefix = f"\n\n[AI Summary - Round {round_idx}]\n"
        summary_suffix = "\n\n[Continuing Reasoning]\n"
        #print("summary result: ",summary_text)
        final_summary_text = summary_prefix + summary_text + summary_suffix

        # 把 summary 当作新的“response”进行下一轮 summary
        response = final_summary_text
        full_text = prompt + response
        token_count = count_tokens(full_text)

        print(f"[Agent A Full Loss] After round {round_idx}: token_count = {token_count}")

    # --- 计算最终拼接结果 ---
    if summarized:
        # 最终的模型输入：原 response + 最终 summary
        combined_response = response_token_ids + tokenizer(final_summary_text, add_special_tokens=False)["input_ids"]
        combined_loss_masks = loss_masks + [0] * len(tokenizer(final_summary_text, add_special_tokens=False)["input_ids"])

        new_response = response + final_summary_text
        new_response_token_ids = combined_response
        new_loss_masks = combined_loss_masks
    else:
        # 不需要 summary 的情况
        new_response = response
        new_response_token_ids = response_token_ids
        new_loss_masks = loss_masks

    print(f"[Agent A Full Loss] Summary rounds: {round_idx}, summarized={summarized}")

    return prompt, final_summary_text, new_response, new_response_token_ids, new_loss_masks, summarized


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """
    完整版生成函数：所有部分都参与训练

    Loss mask 策略：
    - Prompt: 0（不训练）
    - Retool generation: 1（训练）✅
    - Tool output: 0（不训练）
    - Summary: 0
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
    summary = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0
    summarization_count = 0

    for turn in range(TOOL_CONFIGS["max_turns"]):
        if turn > 0 and turn % 1 == 0:
            new_prompt, full_summary_text, new_response, new_response_token_ids, new_loss_masks, summarized = \
                await request_and_apply_summary_full_loss(
                    prompt, response, response_token_ids, loss_masks,
                    state.tokenizer, timeout=400
                )

            if summarized:
                summarization_count += 1
                response = new_response
                response_token_ids = new_response_token_ids
                loss_masks = new_loss_masks
                summary = full_summary_text
                print(f"[Agent A Full Loss] Summary {summarization_count} applied, continuing generation...")

        payload = {
            "text": prompt + summary,
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

    print(f"[Agent A Full Loss] Generation complete:")
    print(f"  - Total tokens: {total_tokens}")
    print(f"  - Trainable tokens: {total_trainable} ({trainable_ratio:.1%})")
    print(f"  - Tool calls: {tool_call_count}")
    print(f"  - Summarizations: {summarization_count}")

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
    #todo: 考虑一下是否真的需要这一项
    if hasattr(sample, 'summarization_count') and sample.summarization_count > 0:
        result["score"] += 0.05 * sample.summarization_count

    if result["pred"] is None:
        result["pred"] = ""

    return result


