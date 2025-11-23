"""
Module: database_utils
-----------------------
Utilities for interacting with the database server.
Handles data submission and retrieval for the dual-agent system.
"""

import json
import requests
from typing import Dict, Optional

from config import global_config


def get_database_url():
    """Get database server URL"""
    return f"http://{global_config.DATABASE_SERVER_IP}:18888"


def commit_summary_data(task_id: str, sample_data: Dict):
    """
    Commit summary task data to database server.
    
    Args:
        task_id: Unique task identifier
        sample_data: Dictionary containing sample data to be summarized
    """
    list_key = f"summary_queue_{global_config.KEY_SUFFIX}"
    
    # Prepare task data
    task_data = {
        "taskId": task_id,
        "originalPrompt": sample_data.get("original_prompt", ""),
        "conversationHistory": sample_data.get("conversation_history", []),
        "tokenCount": sample_data.get("token_count", 0),
        "metadata": sample_data.get("metadata", {}),
    }
    
    url = f"{get_database_url()}/taskCommit"
    payload = {
        "listKey": list_key,
        "taskData": json.dumps(task_data),
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        if result.get("success"):
            print(f"[Database] Successfully committed summary task: {task_id}")
        else:
            print(f"[Database] Failed to commit summary task: {result}")
    except Exception as e:
        print(f"[Database] Error committing summary task: {e}")


def get_summary_data() -> Optional[Dict]:
    """
    Fetch summary task data from database server.
    
    Returns:
        Dictionary containing task data, or None if queue is empty
    """
    list_key = f"summary_queue_{global_config.KEY_SUFFIX}"
    
    url = f"{get_database_url()}/taskFetch"
    payload = {"listKey": list_key}
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        if result.get("success") and result.get("data", {}).get("success"):
            task_data_str = result["data"]["taskData"]
            task_data = json.loads(task_data_str)
            print(f"[Database] Fetched summary task: {task_data.get('taskId')}")
            return task_data
        else:
            return None
            
    except Exception as e:
        print(f"[Database] Error fetching summary task: {e}")
        return None


def commit_completed_summary(task_id: str, summary_text: str):
    """
    Commit completed summary back to database for Agent A to retrieve.
    
    Args:
        task_id: Task identifier
        summary_text: Generated summary text
    """
    list_key = f"completed_summary_{global_config.KEY_SUFFIX}"
    
    task_data = {
        "taskId": task_id,
        "summary": summary_text,
    }
    
    url = f"{get_database_url()}/taskCommit"
    payload = {
        "listKey": list_key,
        "taskData": json.dumps(task_data),
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        if result.get("success"):
            print(f"[Database] Successfully committed completed summary: {task_id}")
    except Exception as e:
        print(f"[Database] Error committing completed summary: {e}")


def fetch_completed_summary(task_id: str, timeout: int = 60) -> Optional[str]:
    """
    Fetch completed summary from database (blocking with timeout).
    
    Args:
        task_id: Task identifier
        timeout: Maximum time to wait in seconds
        
    Returns:
        Summary text or None if timeout
    """
    import time
    
    list_key = f"completed_summary_{global_config.KEY_SUFFIX}"
    url = f"{get_database_url()}/taskFetch"
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            payload = {"listKey": list_key}
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()
            
            if result.get("success") and result.get("data", {}).get("success"):
                task_data_str = result["data"]["taskData"]
                task_data = json.loads(task_data_str)
                
                if task_data.get("taskId") == task_id:
                    return task_data.get("summary")
                    
        except Exception as e:
            print(f"[Database] Error fetching completed summary: {e}")
        
        time.sleep(1)
    
    print(f"[Database] Timeout waiting for summary: {task_id}")
    return None


def commit_retool_training_data(task_id: str, retool_data: Dict, max_retries: int = 10):
    """
    Commit retool agent's training data to database server for Agent B offline training.
    
    Similar to commit_patient_data in MrlX-TakesTwo, this stores complete conversation
    data from Agent A for Agent B to use in offline training.
    
    Args:
        task_id: Unique task identifier
        retool_data: Dictionary containing retool conversation data
        max_retries: Maximum retry attempts
    """
    import time
    from datetime import datetime
    from slime.utils.types import Sample
    
    list_key = f"retool_training_queue_{global_config.KEY_SUFFIX}"
    
    # Skip commit if sample status is aborted
    status = retool_data.get("status")
    if hasattr(Sample, "Status") and status == Sample.Status.ABORTED:
        print("[Database] Sample status is ABORTED, data will not be saved.")
        return None
    
    # Convert Enum status to string if needed
    status_name = status.name if hasattr(status, 'name') else str(status)
    
    # Build task payload for server
    task_data = {
        "id": task_id,
        "response": retool_data.get("response", ""),
        "responseLength": retool_data.get("response_length", 0),
        "status": status_name,
        "tokens": retool_data.get("tokens", []),
        "lossMask": retool_data.get("loss_mask", []),
        "messages": retool_data.get("messages", []),
        "prompt": retool_data.get("prompt", ""),
        "fullConversation": retool_data.get("full_conversation", ""),  # Store full conversation
        "label": retool_data.get("label", ""),
        "toolCallCount": retool_data.get("tool_call_count", 0),
        "summarizationCount": retool_data.get("summarization_count", 0),
        "createdDate": datetime.now().isoformat(),
    }
    
    url = f"{get_database_url()}/taskCommit"
    payload = {
        "listKey": list_key,
        "taskData": json.dumps(task_data, ensure_ascii=False),
    }
    
    # Retry sending data
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result.get("success") and result.get("data", {}).get("success", False):
                print(f"[Database] Task {task_id} submitted successfully after {attempt + 1} attempts.")
                return result
            else:
                print(f"[Database] Task {task_id} submission failed (attempt {attempt + 1}): {result}")
                
        except requests.exceptions.RequestException as e:
            print(f"[Database] Task {task_id} submission failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    
    print(f"[Database] Task {task_id} submission failed after {max_retries} attempts.")
    return None


def get_retool_training_data(max_retries: int = 100, retry_delay: int = 5) -> Optional[Dict]:
    """
    Retrieve retool training data from the remote service for Agent B offline training.
    
    Similar to get_patient_data in MrlX-TakesTwo, this fetches conversation data
    from Agent A for Agent B to use in training.
    
    Args:
        max_retries: Maximum retry attempts before giving up
        retry_delay: Delay in seconds between retries
        
    Returns:
        Dictionary containing retool training data, or None if queue is empty
    """
    import time
    
    list_key = f"retool_training_queue_{global_config.KEY_SUFFIX}"
    url = f"{get_database_url()}/taskFetch"
    payload = {"listKey": list_key}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            # Check if queue is empty
            if not result.get("success"):
                inner_data = result.get("data", {})
                if inner_data and not inner_data.get("success", True):
                    error_msg = inner_data.get("errorMsg", "")
                    if error_msg == "Queue is empty":
                        print("[Database] Queue is empty. Waiting...")
                        time.sleep(retry_delay)
                        continue
                    elif error_msg == "System error":
                        print(f"[Database] System error encountered. Retrying (attempt {attempt + 1}/{max_retries}).")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"[Database] Unexpected error: {error_msg}")
                        return None
            
            # Successful retrieval
            if result.get("success") and result.get("data", {}).get("success"):
                task_data_str = result["data"].get("taskData")
                if task_data_str:
                    try:
                        task_data = json.loads(task_data_str)
                        print(f"[Database] Fetched retool training data: {task_data.get('id')}")
                        return task_data
                    except json.JSONDecodeError:
                        print("[Database] Error decoding taskData JSON.")
                        return None
                else:
                    print("[Database] taskData is missing or null.")
                    return None
                    
        except requests.exceptions.RequestException as e:
            print(f"[Database] HTTP request failed: {e}")
            time.sleep(retry_delay)
    
    print("[Database] Max retries reached. Giving up.")
    return None

