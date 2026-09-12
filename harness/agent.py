import json
import os
from typing import Any, Dict, List, Optional
from openai import OpenAI
from harness.guard import WorkspaceGuard
from harness.planner import Planner
from harness.memory import MemoryStore
from harness.tools import ToolRegistry


class Agent:
    """
    Antigravity-style agent harness executing a Plan -> Act -> Observe -> Reflect loop
    against OpenAI-compatible API endpoints using the OpenAI Python SDK.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "default-model",
        workspace_dir: str = ".",
        db_path: str = ":memory:",
        max_turns: int = 20,
    ):
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://host.containers.internal:8080/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "dummy-key")
        self.model = model
        self.max_turns = max_turns

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        self.guard = WorkspaceGuard(workspace_root=workspace_dir)
        self.planner = Planner(plan_file_path=str(self.guard.workspace_root / "PLAN.md"))
        self.memory = MemoryStore(db_path=db_path)
        self.tool_registry = ToolRegistry(
            guard=self.guard,
            planner=self.planner,
            memory=self.memory,
        )

    def execute_tool_call(self, tool_call) -> Dict[str, Any]:
        fn_name = tool_call.function.name
        try:
            kwargs = json.loads(tool_call.function.arguments or "{}")
        except Exception as e:
            return {"error": f"Invalid JSON arguments for {fn_name}: {e}"}

        if fn_name == "read_file":
            return self.tool_registry.read_file(**kwargs)
        elif fn_name == "write_file":
            return self.tool_registry.write_file(**kwargs)
        elif fn_name == "apply_patch":
            return self.tool_registry.apply_patch(**kwargs)
        elif fn_name == "execute_command":
            return self.tool_registry.execute_command(**kwargs)
        elif fn_name == "update_plan":
            return self.tool_registry.update_plan(**kwargs)
        elif fn_name == "search_memory":
            return self.tool_registry.search_memory(**kwargs)
        else:
            return {"error": f"Unknown tool name: {fn_name}"}

    def run(self, user_prompt: str) -> str:
        """
        Executes the main multi-turn agent loop.
        """
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are an autonomous agent operating within a restricted workspace. "
                    "Maintain state using update_plan tool calls. Always execute steps systematically."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        tools_schema = self.tool_registry.get_openai_tools_schema()

        for turn in range(1, self.max_turns + 1):
            # Enforce context truncation via Planner context guard
            messages = self.planner.truncate_history_context(messages)

            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
            )

            choice = response.choices[0]
            message = choice.message

            # Build assistant message dict
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if message.content:
                assistant_msg["content"] = message.content

            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]

            messages.append(assistant_msg)

            # If no tool calls, model provided final answer or halt
            if not message.tool_calls:
                return message.content or ""

            # Process tool calls
            for tc in message.tool_calls:
                result = self.execute_tool_call(tc)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": json.dumps(result),
                    }
                )

        return "Max turns reached without final text output."
