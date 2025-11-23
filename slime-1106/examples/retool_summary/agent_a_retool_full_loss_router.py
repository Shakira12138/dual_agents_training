import re
import os
from typing import Any, Dict, List, Tuple, Optional

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
from agent_b_summary import construct_summary_prompt, SUMMARY_SYSTEM_PROMPT
from database_utils import commit_retool_training_data
from retool_summary_converter import convert_retool_to_summary_sample
import uuid

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


async def request_summary_via_router(
    prompt: str,
    conversation_history: List[Dict],
    tokenizer,
    timeout: int = 60,
) -> Optional[str]:
    summary_host = os.getenv("SUMMARY_AGENT_HOST", "127.0.0.1")
    summary_port = os.getenv("SUMMARY_AGENT_PORT", "3333")
    summary_url = f"http://{summary_host}:{summary_port}/generate"

    print(f"[Agent A Router] Requesting summary from Agent B router at {summary_url}")

    summary_prompt = construct_summary_prompt(prompt, conversation_history)


    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": summary_prompt},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # ✅ 增量summary策略：只summary新增部分，通常不会太长
    # 但如果真的超长（比如新增部分本身就有6000+ tokens），仍然需要保护
    MAX_FORMATTED_PROMPT_TOKENS = 6000  # 提高到6000，给新增部分足够的空间
    prompt_token_ids = tokenizer(formatted_prompt, add_special_tokens=False)["input_ids"]
    prompt_token_length = len(prompt_token_ids)

    if prompt_token_length > MAX_FORMATTED_PROMPT_TOKENS:
        print(f"[Agent A Router]  Formatted prompt too long ({prompt_token_length} tokens), truncating to {MAX_FORMATTED_PROMPT_TOKENS}")
        print(f"[Agent A Router]  This should be rare with incremental summary strategy (only summarizing new parts)")
        # 截断策略：保留开头（system prompt和重要上下文）+ 结尾（最近的response内容）
        front_tokens = MAX_FORMATTED_PROMPT_TOKENS // 2
        back_tokens = MAX_FORMATTED_PROMPT_TOKENS - front_tokens

        truncated_tokens = prompt_token_ids[:front_tokens] + prompt_token_ids[-back_tokens:]
        formatted_prompt = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
        prompt_token_length = len(truncated_tokens)
        print(f"[Agent A Router] Truncated formatted prompt to {prompt_token_length} tokens")
        print(f"[Agent A Router]  Note: Only formatted prompt was truncated, original response content was NOT truncated")

    payload = {
        "text": formatted_prompt,
        "sampling_params": {
            "temperature": 0.7,
            "max_new_tokens": 2048,
            "top_p": 0.95,
        },
    }

    try:
        output = await post(summary_url, payload)

        if output["meta_info"]["finish_reason"]["type"] == "abort":
            print(f"[Agent A Router] Summary request aborted")
            return None

        summary_text = output["text"].strip()
        #print(f"[Agent A Router] Received summary from Agent B router: {len(summary_text)} chars")
        #print("***************original text is: ", formatted_prompt)
        #print("***************summary text is: ", summary_text)
        return summary_text

    except Exception as e:
        print(f"[Agent A Router] Error requesting summary from router: {e}")
        import traceback
        traceback.print_exc()
        return None


async def request_and_apply_summary_full_loss(
    prompt: str,
    tokenizer,
    res_summary: str = "",  # res_summary：用于生成上下文的summary变量（每次替换）
    timeout: int = 200,
) -> Tuple[str, str, bool]:

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    context_for_generation = prompt + res_summary  # 生成时使用的上下文

    token_count = count_tokens(context_for_generation)

    if token_count > global_config.CONTEXT_LENGTH_THRESHOLD:
        summarized = True
        print(f"[Agent A Router] Context {token_count} > {global_config.CONTEXT_LENGTH_THRESHOLD}, requesting summary for res_summary")
        print(f"[Agent A Router] prompt + res_summary: {token_count} tokens (res_summary: {count_tokens(res_summary)} tokens)")

        summary_content = res_summary if res_summary else ""

        summary_text = await request_summary_via_router(
            prompt=prompt,
            conversation_history=[{"role": "assistant", "content": summary_content}],  # summary(res_summary)
            tokenizer=tokenizer,
            timeout=timeout,
        )

        if not summary_text:
            print(f"[Agent A Router] Failed to get summary, continue without summary")
            summarized = False
            new_res_summary = res_summary  # 保持原有summary
        else:
            # 包装新的summary
            summary_prefix = f"\n\n[AI Summary]\n"
            summary_suffix = "\n\n[Continuing Reasoning]\n"
            new_res_summary = summary_prefix + summary_text + summary_suffix
            print(f"[Agent A Router] Old res_summary: {count_tokens(res_summary)} tokens  New res_summary: {count_tokens(new_res_summary)} tokens")
    else:
        # 不需要summary，保持原有summary
        summarized = False
        new_res_summary = res_summary

    print(f"[Agent A Router] Summary applied: summarized={summarized}, context tokens: {count_tokens(prompt + new_res_summary)}")

    return prompt, new_res_summary, summarized


async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported"

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Initialize
    tool_specs = tool_registry.get_tool_specs()
    prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)

    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response = ""
    tool_call_count = 0
    summarization_count = 0
    res_summary = ""
    res_summary_token_ids = []
    res_summary_loss_mask = []

    for turn in range(TOOL_CONFIGS["max_turns"]):
        if turn > 0:
            prompt, new_res_summary, summarized = \
                await request_and_apply_summary_full_loss(
                    prompt,
                    state.tokenizer,
                    res_summary=res_summary,
                    timeout=200
                )

            res_summary = new_res_summary

            if summarized:
                res_summary_token_ids = state.tokenizer(res_summary, add_special_tokens=False)["input_ids"]
                res_summary_loss_mask = [0] * len(res_summary_token_ids)
                summarization_count += 1
                print(f"[Agent A Router] Summary {summarization_count} applied, res_summary: {len(res_summary)} chars")

        # 生成时使用 prompt + res_summary（压缩后的上下文）
        # 而不是 prompt + response（完整的、可能很长的上下文）
        payload = {
            "text": prompt + res_summary,
            "sampling_params": sampling_params,
        }

        output = await post(url, payload)

        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            sample.reward = global_config.DEFAULT_SCORE
            return sample

        cur_response = output["text"]
        cur_response = postprocess_responses(cur_response)

        # Tokenize response
        cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

        # response：累积用于训练（评估模型质量）
        response += cur_response

        # res_summary：累积用于下一轮生成上下文（模型需要知道之前说了什么）
        res_summary += cur_response
        res_summary_token_ids += cur_response_token_ids
        res_summary_loss_mask += [1] * len(cur_response_token_ids)

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

        # response：累积tool输出（用于训练）
        response += next_obs
        # res_summary：累积tool输出（用于下一轮生成上下文）
        res_summary += next_obs
        res_summary_token_ids += obs_tokens_ids
        res_summary_loss_mask += [0] * len(obs_tokens_ids)

        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            sample.status = Sample.Status.COMPLETED
            break

    summary_token_ids = res_summary_token_ids
    summary_loss_mask = res_summary_loss_mask

    MAX_RESPONSE_TOKENS = 28672
    if len(summary_token_ids) >= MAX_RESPONSE_TOKENS:
        sample.status = Sample.Status.ABORTED
        sample.reward = {
        "score": global_config.DEFAULT_SCORE,
         "pred": "",
         "label": "",
         "acc": 0.0,
        }
        sample.tokens = prompt_tokens_ids  # 只保留 prompt
        sample.response_length = 0
        sample.full_response = response
        sample.response = ""
        sample.loss_mask = []
        sample.tool_call_count = tool_call_count
        sample.summarization_count = summarization_count
        print(f"[Agent A Router] Discard sample: summary tokens {len(summary_token_ids)} > {MAX_RESPONSE_TOKENS}")
        return sample
    sample.tokens = prompt_tokens_ids + summary_token_ids
    sample.response_length = len(summary_token_ids)
    sample.full_response = response
    sample.response = res_summary
    sample.loss_mask = summary_loss_mask
    sample.tool_call_count = tool_call_count
    sample.summarization_count = summarization_count

    assert len(sample.tokens) == len(prompt_tokens_ids) + len(summary_token_ids), "Token count mismatch"
    assert len(sample.loss_mask) == len(summary_token_ids), f"Loss mask mismatch: {len(sample.loss_mask)} vs {len(summary_token_ids)}"


    # Statistics
    total_trainable = sum(sample.loss_mask)
    total_tokens = len(sample.loss_mask)
    trainable_ratio = total_trainable / max(total_tokens, 1)

    print(f"[Agent A Router] Generation complete:")
    print(f"  - Total tokens: {total_tokens}")
    print(f"  - Trainable tokens: {total_trainable} ({trainable_ratio:.1%})")
    print(f"  - Tool calls: {tool_call_count}")
    print(f"  - Summarizations: {summarization_count}")

    if not hasattr(sample, 'status') or sample.status is None:
        sample.status = Sample.Status.COMPLETED

    # Store conversation data to database for Agent B offline training
    # Similar to MrlX-TakesTwo: commit patient data after doctor generation
    summary_sample = convert_retool_to_summary_sample(sample)
    if summary_sample:
        task_id = str(uuid.uuid4())
        try:
            commit_retool_training_data(task_id, summary_sample)
            print(f"[Agent A] Stored training data to database: {task_id}")
        except Exception as e:
            print(f"[Agent A] Failed to store training data to database: {e}")

    return sample


async def reward_func(args, sample, **kwargs):
    """Reward function (same as before)"""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    solution_response = getattr(sample, "full_response", sample.response)
    solution_str = sample.prompt + solution_response

    ground_truth = sample.label if sample.label is not None else ""
    num_turns = getattr(sample, "tool_call_count", 0)

    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)

    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if hasattr(sample, 'summarization_count') and sample.summarization_count > 0:
        result["score"] += 0.05 * sample.summarization_count

    if result["pred"] is None:
        result["pred"] = ""

    return result



