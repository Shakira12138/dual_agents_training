"""
Module: agent_a_retool_v2
-------------------------
Enhanced version of Agent A with synchronous summary retrieval and context replacement.

Key improvements:
1. Synchronous summary waiting and retrieval
2. Context replacement with summary
3. Continued generation after summarization

Functions:
    generate(...) -> Sample
    reward_func(...) -> float
"""

import re
import uuid
import asyncio
from typing import Any, Dict, List

from jinja2 import Template

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Import retool reward
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


async def request_and_apply_summary(
    prompt: str, 
    response: str, 
    tokenizer,
    timeout: int = 60
) -> tuple[str, str, bool]:
    """
    Request summary from Agent B and apply it to compress context.
    
    Args:
        prompt: Current full prompt
        response: Current full response
        tokenizer: Tokenizer for counting tokens
        timeout: Maximum time to wait for summary (seconds)
        
    Returns:
        tuple: (new_prompt, new_response, summarized)
            - new_prompt: Compressed prompt (or original if no summary)
            - new_response: Compressed response (or original if no summary)
            - summarized: Whether summarization was applied
    """
    full_text = prompt + response
    token_count = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
    
    # Check if we need to summarize
    if token_count < global_config.CONTEXT_LENGTH_THRESHOLD:
        return prompt, response, False
    
    print(f"[Agent A] Context length {token_count} exceeds threshold {global_config.CONTEXT_LENGTH_THRESHOLD}")
    print(f"[Agent A] Requesting summary from Agent B...")
    
    # Submit summary request to database
    task_id = str(uuid.uuid4())
    summary_data = {
        "original_prompt": prompt,
        "conversation_history": [{"role": "assistant", "content": response}],
        "token_count": token_count,
        "metadata": {"task_id": task_id},
    }
    
    commit_summary_data(task_id, summary_data)
    print(f"[Agent A] Summary request submitted: {task_id}")
    
    # Wait for summary (synchronous)
    summary_text = fetch_completed_summary(task_id, timeout=timeout)
    
    if summary_text:
        print(f"[Agent A] Summary received: {len(summary_text)} chars")
        
        # Strategy 1: Replace entire conversation with summary
        # Keep only the original problem statement + summary
        new_prompt = prompt  # Keep original prompt
        new_response = f"\n\n[Previous conversation summarized by AI]\n{summary_text}\n\n[Continuing with the problem...]\n"
        
        new_token_count = len(tokenizer(new_prompt + new_response, add_special_tokens=False)["input_ids"])
        compression_ratio = token_count / max(new_token_count, 1)
        
        print(f"[Agent A] Context compressed: {token_count} -> {new_token_count} tokens ({compression_ratio:.2f}x)")
        
        return new_prompt, new_response, True
    else:
        print(f"[Agent A] Summary timeout after {timeout}s, continuing without compression")
        return prompt, response, False


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """
    Enhanced Agent A generation with context compression.
    
    Workflow:
    1. Generate response with tool calls
    2. Monitor context length
    3. If exceeds threshold:
       - Request summary from Agent B
       - Wait for summary (with timeout)
       - Replace context with summary
       - Continue generation
    4. Compute reward
    
    Args:
        args: Training arguments
        sample: Sample object containing prompt
        sampling_params: Sampling parameters
        
    Returns:
        Completed Sample with response and metadata
    """
    assert not args.partial_rollout, "Partial rollout is not supported"
    
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    
    # Set up initial prompt with tools
    tool_specs = tool_registry.get_tool_specs()
    prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)
    
    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0
    summarization_count = 0
    
    for turn in range(TOOL_CONFIGS["max_turns"]):
        # Check if we need to summarize BEFORE generating
        if turn > 0 and turn % 5 == 0:  # Check every 5 turns
            new_prompt, new_response, summarized = await request_and_apply_summary(
                prompt, response, state.tokenizer, timeout=60
            )
            
            if summarized:
                summarization_count += 1
                # Update prompt and response with compressed versions
                prompt = new_prompt
                response = new_response
                
                # Re-tokenize after compression
                prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
                response_token_ids = state.tokenizer(response, add_special_tokens=False)["input_ids"]
                loss_masks = [0] * len(response_token_ids)  # Summary part not trained
        
        # Generate response
        payload = {
            "text": prompt + response,
            "sampling_params": sampling_params,
        }
        
        output = await post(url, payload)
        
        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            sample.reward = global_config.DEFAULT_SCORE
            return sample
        
        cur_response = output["text"]
        cur_response = postprocess_responses(cur_response)
        
        # Record response tokens
        cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]
        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)  # Train on new generations
        
        # Check length limit
        if output["meta_info"]["finish_reason"]["type"] == "length":
            sample.status = Sample.Status.TRUNCATED
            break
        
        # Execute prediction
        next_obs, done = await execute_predictions(cur_response)
        if done:
            sample.status = Sample.Status.COMPLETED
            break
        
        # Count tool calls
        if "<interpreter>" in next_obs:
            tool_call_count += 1
        
        # Add observation to response
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)  # Don't train on tool outputs
        
        # Check max tool calls
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
    
    # Store metadata
    sample.metadata["summarization_count"] = summarization_count
    sample.metadata["final_context_length"] = len(sample.tokens)
    
    # Set default status if not set
    if not hasattr(sample, 'status') or sample.status is None:
        sample.status = Sample.Status.COMPLETED
    
    print(f"[Agent A] Generation complete: {tool_call_count} tool calls, {summarization_count} summarizations")
    
    return sample


async def reward_func(args, sample, **kwargs):
    """
    Retool reward function using math_dapo as primary reward model.
    
    Args:
        args: Training arguments
        sample: Sample object
        
    Returns:
        Reward score dict
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")
    
    # Build complete solution string
    solution_str = sample.prompt + sample.response
    
    # Get ground truth answer
    ground_truth = sample.label if sample.label is not None else ""
    
    # Get tool call count
    num_turns = getattr(sample, "tool_call_count", 0)
    
    # Use math_dapo reward
    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)
    
    # Encourage model to call tools
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)
    
    # Small bonus for successful summarization (shows good context management)
    if hasattr(sample, 'summarization_count') and sample.summarization_count > 0:
        result["score"] += 0.05 * sample.summarization_count
    
    if result["pred"] is None:
        result["pred"] = ""
    
    return result

