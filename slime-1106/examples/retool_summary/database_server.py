"""
Module: database_server
-----------------------
In-memory task queue service for retool-summary workflow.

Provides:
    - /taskFetch  : Fetch the next task from a queue (FIFO order).
    - /taskCommit : Commit/append a new task to a queue.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import threading
import json
import uvicorn

from typing import Dict, List
from config import global_config

# Application setup & in-memory storage
app = FastAPI()

# Dictionary storing task queues
storage: Dict[str, List[str]] = {}

# Lock for thread-safe access
lock = threading.Lock()


# Request Models
class TaskFetchRequest(BaseModel):
    """Request body for /taskFetch API"""
    listKey: str


class TaskCommitRequest(BaseModel):
    """Request body for /taskCommit API"""
    listKey: str
    taskData: str


# API Endpoints
@app.post("/taskFetch")
def task_fetch(payload: TaskFetchRequest):
    """Fetch and remove the first task in the specified queue (FIFO order)"""
    list_key = payload.listKey
    
    with lock:
        if list_key not in storage or not storage[list_key]:
            return JSONResponse(
                status_code=200,
                content={"success": False, "data": {"success": False, "errorMsg": "Queue is empty"}},
            )
        
        try:
            task_data = storage[list_key].pop(0)
            return {"success": True, "data": {"success": True, "taskData": task_data}}
        except Exception:
            return JSONResponse(
                status_code=500,
                content={"success": False, "data": {"success": False, "errorMsg": "System error"}},
            )


@app.post("/taskCommit")
def task_commit(payload: TaskCommitRequest):
    """Commit/append a new task to the specified queue"""
    list_key = payload.listKey
    task_data = payload.taskData
    
    with lock:
        if list_key not in storage:
            storage[list_key] = []
        
        try:
            # Validate JSON format
            json.loads(task_data)
            
            # Append the task
            storage[list_key].append(task_data)
            return {"success": True, "data": {"success": True}}
            
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"success": False, "data": {"success": False, "errorMsg": "Invalid taskData JSON"}},
            )
        except Exception:
            return JSONResponse(
                status_code=500,
                content={"success": False, "data": {"success": False, "errorMsg": "Server error"}},
            )


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "queues": list(storage.keys())}


# Main entrypoint
if __name__ == "__main__":
    DATABASE_SERVER_IP = global_config.DATABASE_SERVER_IP
    if not DATABASE_SERVER_IP:
        DATABASE_SERVER_IP = "0.0.0.0"
    
    print(f"Starting database server on {DATABASE_SERVER_IP}:18888")
    uvicorn.run("database_server:app", host=DATABASE_SERVER_IP, port=18888, reload=False)

