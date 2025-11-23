"""
Module: summary_reward
----------------------
LLM-as-judge reward function for evaluating summary quality.
"""

import json
import asyncio
from typing import Dict, List, Any
from openai import AsyncOpenAI

from config import global_config


# Judge prompt template
JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing the quality of a summary.

**Original Query:**
{original_prompt}

**Original Conversation:**
{conversation_text}

**Generated Summary:**
{summary}

**Evaluation Criteria:**
    1) Completeness: captures all critical facts/decisions/constraints and key reasoning.
    - 9-10: covers all critical facts/decisions/constraints; no key omissions
    - 7-8: minor omissions only
    - 4-6: misses at least one key point
    - 0-3: largely off or missing

    2) Conciseness: high information density with minimal redundancy.
    - 9-10: very high information density; minimal redundancy
    - 7-8: small redundancy; acceptable
    - 4-6: noticeable verbosity or under-compressed
    - 0-3: long paraphrase or irrelevant content

    3) Coherence: clear structure and logical flow.
    - 9-10: clear structure and logical flow
    - 7-8: mostly clear with minor jumps
    - 4-6: disorganized; hard to follow
    - 0-3: incoherent or contradictory

    4) Faithfulness: strictly faithful to the conversation; no invented content.
    - 9-10: strictly faithful; no hallucinations
    - 7-8: minor phrasing shifts; meaning intact
    - 4-6: some distortions/unsupported claims
    - 0-3: major hallucinations or contradictions

**Compression Ratio:**
- Original tokens: {original_token_count}
- Summary tokens: {summary_token_count}
- Compression ratio: {compression_ratio:.2f}x

Please provide your evaluation in the following JSON format:
```json
{{
  "completeness": <0-10>,
  "conciseness": <0-10>,
  "coherence": <0-10>,
  "faithfulness": <0-10>,
  "reasoning": "<brief justification citing 1-3 concrete issues or strengths (<=200 chars)>"
}}
```

Your evaluation:"""


def format_conversation_for_judge(conversation_history: List[Dict]) -> str:
    """Format conversation history for judge evaluation"""
    conversation_text = ""
    for turn in conversation_history:
        role = turn.get("role", "assistant")
        content = turn.get("content", "")
        conversation_text += f"\n\n[{role.upper()}]\n{content}"
    return conversation_text


async def call_judge_model(
    client: AsyncOpenAI,
    prompt: str,
    temperature: float = 0.3,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Call LLM judge model to evaluate summary.

    Args:
        client: OpenAI async client
        prompt: Evaluation prompt
        temperature: Sampling temperature
        max_retries: Maximum retry attempts

    Returns:
        Evaluation scores dictionary
    """
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=global_config.JUDGE_MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format={"type":"json_object"},
                temperature=temperature,
                max_tokens=1000,
            )

            content = response.choices[0].message.content

            # Extract JSON from response
            json_match = content
            if "```json" in content:
                json_match = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_match = content.split("```")[1].split("```")[0].strip()

            # Parse JSON
            evaluation = json.loads(json_match)

            # Ensure evaluation is a dictionary (not string, list, number, etc.)
            if not isinstance(evaluation, dict):
                raise ValueError(f"Expected dict, got {type(evaluation).__name__}: {evaluation}")

            return evaluation

        except (json.JSONDecodeError, IndexError, KeyError, ValueError) as e:
            print(f"[Judge] Attempt {attempt + 1} failed to parse response: {e}")
            if attempt == max_retries - 1:
                # Return default scores on final failure
                return {
                    "completeness": 5.0,
                    "conciseness": 5.0,
                    "coherence": 5.0,
                    "faithfulness": 5.0,
                    "reasoning": "Failed to get valid judge response"
                }
        except Exception as e:
            print(f"[Judge] Attempt {attempt + 1} failed with error: {e}")
            if attempt == max_retries - 1:
                return {
                    "completeness": 5.0,
                    "conciseness": 5.0,
                    "coherence": 5.0,
                    "faithfulness": 5.0,
                    "reasoning": f"Error: {str(e)}"
                }

        # Wait before retry
        await asyncio.sleep(1)

    return {
        "completeness": 5.0,
        "conciseness": 5.0,
        "coherence": 5.0,
        "faithfulness": 5.0,
        "reasoning": "Max retries exceeded"
    }


async def compute_summary_score(
    original_prompt: str,
    conversation_history: List[Dict],
    summary: str,
    original_token_count: int,
    summary_token_count: int,
) -> Dict[str, Any]:
    """
    Compute summary quality score using LLM-as-judge.

    Args:
        original_prompt: Original user query
        conversation_history: List of conversation turns
        summary: Generated summary text
        original_token_count: Token count of original conversation
        summary_token_count: Token count of summary

    Returns:
        Dictionary with scores and total_score
    """
    # Calculate compression ratio
    compression_ratio = original_token_count / max(summary_token_count, 1)

    # Format conversation for judge
    conversation_text = format_conversation_for_judge(conversation_history)

    # Construct judge prompt
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        original_prompt=original_prompt,
        conversation_text=conversation_text,
        summary=summary,
        original_token_count=original_token_count,
        summary_token_count=summary_token_count,
        compression_ratio=compression_ratio,
    )

    # Create judge client
    judge_client = AsyncOpenAI(
        api_key=global_config.JUDGE_MODEL_API_KEY,
        base_url=global_config.JUDGE_MODEL_BASE_URL,
    )

    # Get evaluation from judge
    evaluation = await call_judge_model(judge_client, judge_prompt)

    # Extract scores
    completeness = float(evaluation.get("completeness", 5.0))
    conciseness = float(evaluation.get("conciseness", 5.0))
    coherence = float(evaluation.get("coherence", 5.0))
    faithfulness = float(evaluation.get("faithfulness", 5.0)) 
    reasoning = evaluation.get("reasoning", "")

    # Compute weighted total score (normalized to 0-1 range)
    # Weights: completeness=0.4, conciseness=0.3, coherence=0.3
    total_score = (completeness * 0.4 + conciseness * 0.3 + coherence * 0.3 + faithfulness * 0.5) / 15.0

    # Bonus for good compression ratio (2-5x is ideal)
    if 2.0 <= compression_ratio <= 5.0:
        compression_bonus = 0.1
    elif compression_ratio > 1.5:
        compression_bonus = 0.05
    else:
        compression_bonus = 0.0

    total_score = min(1.0, total_score + compression_bonus)

    return {
        "completeness": completeness,
        "conciseness": conciseness,
        "coherence": coherence,
        "faithfulness": faithfulness,
        "compression_ratio": compression_ratio,
        "reasoning": reasoning,
        "total_score": total_score,
    }

