import os
import subprocess
from typing import Any, Dict, List, Optional
from harness.guard import WorkspaceGuard
from harness.planner import Planner
from harness.memory import MemoryStore


class ToolRegistry:
    """
    Registry for tool primitives used by the agent, enforcing workspace security rules.
    """

    def __init__(
        self,
        guard: Optional[WorkspaceGuard] = None,
        planner: Optional[Planner] = None,
        memory: Optional[MemoryStore] = None,
    ):
        self.guard = guard or WorkspaceGuard()
        self.planner = planner or Planner()
        self.memory = memory or MemoryStore()

    def read_file(self, path: str, offset: int = 0, limit: Optional[int] = None) -> Dict[str, Any]:
        """Reads content from a file within the workspace boundary."""
        try:
            safe_path = self.guard.validate_path(path)
            if not safe_path.exists():
                return {"error": f"File '{path}' does not exist."}

            with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            selected_lines = lines[offset:]
            if limit is not None:
                selected_lines = selected_lines[:limit]

            return {
                "path": path,
                "content": "".join(selected_lines),
                "total_lines": total_lines,
                "offset": offset,
                "limit": limit,
            }
        except Exception as e:
            return {"error": str(e)}

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Writes content to a file within the workspace boundary."""
        try:
            safe_path = self.guard.validate_path(path)
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            with open(safe_path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"status": "success", "path": path, "bytes_written": len(content)}
        except Exception as e:
            return {"error": str(e)}

    def apply_patch(self, path: str, old_str: str, new_str: str) -> Dict[str, Any]:
        """Replaces exact occurrence of old_str with new_str in file."""
        try:
            safe_path = self.guard.validate_path(path)
            if not safe_path.exists():
                return {"error": f"File '{path}' does not exist."}

            with open(safe_path, "r", encoding="utf-8") as f:
                file_content = f.read()

            if old_str not in file_content:
                return {"error": f"Target old_str not found in '{path}'."}

            updated_content = file_content.replace(old_str, new_str, 1)
            with open(safe_path, "w", encoding="utf-8") as f:
                f.write(updated_content)

            return {"status": "success", "path": path}
        except Exception as e:
            return {"error": str(e)}

    def execute_command(self, command: str, timeout_sec: int = 30) -> Dict[str, Any]:
        """Executes bash command within workspace boundary."""
        try:
            # Basic validation to disallow escaping workspace directory via path traversal in command if needed
            if ".." in command.split():
                return {"error": "Path traversal detected in command string."}

            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.guard.workspace_root),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            return {
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Command '{command}' timed out after {timeout_sec} seconds."}
        except Exception as e:
            return {"error": str(e)}

    def update_plan(
        self,
        summary: Optional[str] = None,
        completed_tasks: Optional[List[str]] = None,
        current_task: Optional[str] = None,
        next_tasks: Optional[List[str]] = None,
        content: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delegates plan updates to the Planner state manager."""
        return self.planner.update_plan(
            summary=summary,
            completed_tasks=completed_tasks,
            current_task=current_task,
            next_tasks=next_tasks,
            content=content,
        )

    def search_memory(self, query: str) -> Dict[str, Any]:
        """Searches long-term RAG memory using query string."""
        results = self.memory.retrieve_relevant(query=query, top_k=3)
        return {"query": query, "results": results}

    def get_openai_tools_schema(self) -> List[Dict[str, Any]]:
        """Returns JSON schema definitions for OpenAI function calls."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents within workspace boundary.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative file path"},
                            "offset": {"type": "integer", "description": "Line offset to start reading from"},
                            "limit": {"type": "integer", "description": "Number of lines to read"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write text content to file within workspace boundary.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative file path"},
                            "content": {"type": "string", "description": "File content to write"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "description": "Replace exact string match in file with new string.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative file path"},
                            "old_str": {"type": "string", "description": "Exact text to replace"},
                            "new_str": {"type": "string", "description": "Replacement text"},
                        },
                        "required": ["path", "old_str", "new_str"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Execute a bash command within workspace directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Bash command string"},
                            "timeout_sec": {"type": "integer", "description": "Execution timeout in seconds"},
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_plan",
                    "description": "Overwrites PLAN.md with new structured state or raw Markdown content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string", "description": "Summary of overall plan"},
                            "completed_tasks": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of completed tasks",
                            },
                            "current_task": {"type": "string", "description": "Task currently active"},
                            "next_tasks": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of upcoming tasks",
                            },
                            "content": {"type": "string", "description": "Raw Markdown plan content"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "Search embedded memory for relevant information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query string"},
                        },
                        "required": ["query"],
                    },
                },
            },
        ]
