"""
Module: retool_summary_converter
---------------------------------
Converts a retool agent conversation sample (Sample) into summary agent
training format suitable for offline training.

Similar to change_patient_tool_to_sample in MrlX-TakesTwo, this converts
Agent A's conversation data into format for Agent B's training.
"""

import logging
from typing import Dict, Optional
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def convert_retool_to_summary_sample(retool_sample: Sample) -> Optional[Dict]:
    """
    Convert a retool agent conversation Sample into summary agent training format.
    
    This will:
        1. Extract the full conversation (prompt + response)
        2. Format it for summary training
        3. Prepare tokens and loss mask
    
    Args:
        retool_sample: Retool agent's conversation sample
        
    Returns:
        dict: Processed summary training sample, or None if invalid
    """
    try:
        # Extract data from retool sample
        prompt = retool_sample.prompt
        response = retool_sample.response
        tokens = retool_sample.tokens if hasattr(retool_sample, 'tokens') else []
        loss_mask = retool_sample.loss_mask if hasattr(retool_sample, 'loss_mask') else []
        status = retool_sample.status if hasattr(retool_sample, 'status') else None
        
        # Build messages for summary training
        # The full conversation (prompt + response) will be used as the content to summarize
        full_conversation = prompt + response
        
        # For summary training, we treat the full conversation as the content to summarize
        # The original prompt is the problem statement, and the response is the conversation
        conversation_history = [
            {"role": "assistant", "content": full_conversation}
        ]
        
        # Build summary training sample
        # Note: This format matches what get_retool_training_data expects
        summary_sample = {
            "messages": conversation_history,
            "tokens": tokens,
            "response_length": len(tokens) if isinstance(tokens, list) else 0,
            "loss_mask": loss_mask,
            "response": response,
            "prompt": prompt,
            "full_conversation": full_conversation,  # Store full conversation for summary
            "label": retool_sample.label if hasattr(retool_sample, 'label') else "",
            "status": status,
            "tool_call_count": getattr(retool_sample, 'tool_call_count', 0),
            "summarization_count": getattr(retool_sample, 'summarization_count', 0),
        }
        
        return summary_sample
        
    except Exception as e:
        logger.error(f"Error converting retool sample to summary sample: {e}")
        return None

