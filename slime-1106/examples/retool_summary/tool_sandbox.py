"""
Tool sandbox module for safe code execution and tool management.
Adapted from examples/retool/tool_sandbox.py
"""

import asyncio
import gc
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, List

import psutil

from config import global_config

# Configuration for tool execution
TOOL_CONFIGS = {
    "max_turns": global_config.TOOL_MAX_TURNS,
    "max_tool_calls": global_config.TOOL_MAX_CALLS,
    "tool_concurrency": 32,
    "python_timeout": global_config.TOOL_TIMEOUT,
    "python_memory_limit": "4GB",
    "python_cpu_limit": 1,
    "max_memory_usage": 12288,
    "cleanup_threshold": 6144,
    "aggressive_cleanup_threshold": 3072,
    "force_cleanup_threshold": 9216,
}

# Global semaphore for controlling concurrent tool executions
SEMAPHORE = asyncio.Semaphore(TOOL_CONFIGS["tool_concurrency"])


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()


def check_and_cleanup_memory():
    """Check memory usage and perform appropriate cleanup"""
    current_memory = get_memory_usage()
    
    if current_memory > TOOL_CONFIGS["force_cleanup_threshold"]:
        gc.collect()
        return f"Warning: High memory usage ({current_memory:.1f}MB), performed cleanup"
    elif current_memory > TOOL_CONFIGS["cleanup_threshold"]:
        gc.collect()
        return f"Info: Memory usage ({current_memory:.1f}MB), performed cleanup"
    
    return None


class PythonSandbox:
    """Python code sandbox for safe code execution"""

    def __init__(self, timeout: int = 120, memory_limit: str = "4GB"):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.allowed_modules = {
            "math", "random", "datetime", "collections", "itertools",
            "functools", "operator", "statistics", "decimal", "fractions",
        }

    def _check_code_safety(self, code: str) -> tuple[bool, str]:
        """Check code safety by scanning for dangerous patterns"""
        dangerous_patterns = [
            r"import\s+os", r"import\s+sys", r"import\s+subprocess",
            r"__import__", r"eval\s*\(", r"exec\s*\(",
            r"open\s*\(", r"input\s*\(", r"compile\s*\(",
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Code contains dangerous pattern: {pattern}"
        
        return True, "Code is safe"

    @contextmanager
    def _create_safe_environment(self):
        """Create safe execution environment with temporary directory"""
        temp_dir = tempfile.mkdtemp(prefix="python_sandbox_")
        
        try:
            script_path = os.path.join(temp_dir, "code.py")
            env = os.environ.copy()
            env["PYTHONPATH"] = temp_dir
            env["PYTHONUNBUFFERED"] = "1"
            
            yield script_path, env, temp_dir
        finally:
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception:
                pass

    async def execute_code(self, code: str) -> str:
        """Execute Python code in sandbox with safety checks"""
        current_memory = get_memory_usage()
        if current_memory > TOOL_CONFIGS["max_memory_usage"]:
            cleanup_memory()
            return "Error: Memory usage too high, please try again"

        is_safe, message = self._check_code_safety(code)
        if not is_safe:
            return f"Error: {message}"

        indented_code = "\n".join("    " + line for line in code.split("\n"))
        
        wrapped_code = f"""import sys
import traceback
from io import StringIO

old_stdout = sys.stdout
old_stderr = sys.stderr
stdout_capture = StringIO()
stderr_capture = StringIO()
sys.stdout = stdout_capture
sys.stderr = stderr_capture

try:
{indented_code}
    
    stdout_output = stdout_capture.getvalue()
    stderr_output = stderr_capture.getvalue()
    
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    result = ""
    if stdout_output:
        result += f"Output:\\n{{stdout_output}}"
    if stderr_output:
        result += f"\\nErrors:\\n{{stderr_output}}"
    
    print(result if result else "Code executed successfully with no output")
    
except Exception as e:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    error_msg = f"Error: {{str(e)}}\\nTraceback:\\n{{traceback.format_exc()}}"
    print(error_msg)"""

        with self._create_safe_environment() as (script_path, env, temp_dir):
            with open(script_path, "w") as f:
                f.write(wrapped_code)

            try:
                process = subprocess.Popen(
                    ["python3", script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=temp_dir,
                    text=True,
                )

                try:
                    stdout, stderr = process.communicate(timeout=self.timeout)
                    
                    if process.returncode == 0:
                        result = stdout.strip()
                    else:
                        result = f"Error: Process exited with code {process.returncode}\n{stderr}"

                except subprocess.TimeoutExpired:
                    process.kill()
                    result = f"Error: Code execution timed out after {self.timeout} seconds"

            except Exception as e:
                result = f"Error: Failed to execute code: {str(e)}"

            cleanup_message = check_and_cleanup_memory()
            if cleanup_message:
                print(f"[Memory] {cleanup_message}")

            return result


class ToolRegistry:
    """Tool registry for managing available tools"""

    def __init__(self):
        self.tools = {}
        self.python_sandbox = PythonSandbox(
            timeout=TOOL_CONFIGS["python_timeout"],
            memory_limit=TOOL_CONFIGS["python_memory_limit"]
        )
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools"""
        self.register_tool(
            "code_interpreter",
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "description": "Execute Python code in a safe sandbox environment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code to execute"
                            }
                        },
                        "required": ["code"],
                    },
                },
            },
        )

    def register_tool(self, name: str, tool_spec: Dict[str, Any]):
        """Register a new tool"""
        self.tools[name] = tool_spec

    def get_tool_specs(self) -> List[Dict[str, Any]]:
        """Get all tool specifications"""
        return list(self.tools.values())

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"

        if tool_name == "code_interpreter":
            return await self._execute_python(arguments)
        else:
            return f"Error: Tool '{tool_name}' not implemented"

    async def _execute_python(self, arguments: Dict[str, Any]) -> str:
        """Execute Python code using the sandbox"""
        code = arguments.get("code", "")
        if not code.strip():
            return "Error: No code provided"

        result = await self.python_sandbox.execute_code(code)
        return result


# Global tool registry instance
tool_registry = ToolRegistry()

