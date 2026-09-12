import json
import pytest
import httpx
from harness.agent import Agent


class MockHTTPXTransport(httpx.BaseTransport):
    """
    Simulates /v1/chat/completions endpoint for 4 distinct turns:
    Turn 1: update_plan
    Turn 2: write_file (calc.py)
    Turn 3: execute_command (python calc.py)
    Turn 4: Final text response & halt
    """

    def __init__(self):
        self.turn_count = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.turn_count += 1

        if self.turn_count == 1:
            # Turn 1: Call update_plan
            response_payload = {
                "id": "chatcmpl-turn1",
                "object": "chat.completion",
                "created": 123456789,
                "model": "mock-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "update_plan",
                                        "arguments": json.dumps({
                                            "summary": "Calculator project plan",
                                            "completed_tasks": [],
                                            "current_task": "Create calc.py",
                                            "next_tasks": ["Run calc.py"],
                                        }),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        elif self.turn_count == 2:
            # Turn 2: Call write_file (calc.py)
            response_payload = {
                "id": "chatcmpl-turn2",
                "object": "chat.completion",
                "created": 123456789,
                "model": "mock-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": json.dumps({
                                            "path": "calc.py",
                                            "content": "def add(a, b):\n    return a + b\n\nif __name__ == '__main__':\n    print(f'2 + 3 = {add(2, 3)}')\n",
                                        }),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        elif self.turn_count == 3:
            # Turn 3: Call execute_command (python calc.py)
            response_payload = {
                "id": "chatcmpl-turn3",
                "object": "chat.completion",
                "created": 123456789,
                "model": "mock-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_3",
                                    "type": "function",
                                    "function": {
                                        "name": "execute_command",
                                        "arguments": json.dumps({
                                            "command": "python3 calc.py",
                                        }),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        else:
            # Turn 4: Final summary output text and halt
            response_payload = {
                "id": "chatcmpl-turn4",
                "object": "chat.completion",
                "created": 123456789,
                "model": "mock-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Calculator implementation and execution completed successfully.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }

        return httpx.Response(200, json=response_payload)


def test_agent_loop_e2e_mock(tmp_path):
    mock_transport = MockHTTPXTransport()

    agent = Agent(
        base_url="http://mock-llm-server/v1",
        api_key="mock-key",
        workspace_dir=str(tmp_path),
        max_turns=10,
    )

    # Monkeypatch the OpenAI client's httpx client transport
    agent.client._client = httpx.Client(transport=mock_transport)

    final_result = agent.run("Please build and test calc.py")

    assert "completed successfully" in final_result

    # Verification assertions:
    # 1. calc.py exists and contains expected code
    calc_file = tmp_path / "calc.py"
    assert calc_file.exists()
    assert "def add(a, b):" in calc_file.read_text(encoding="utf-8")

    # 2. PLAN.md exists and matches final state only (no duplicated log entries)
    plan_file = tmp_path / "PLAN.md"
    assert plan_file.exists()
    plan_content = plan_file.read_text(encoding="utf-8")
    assert "# Plan Overview" in plan_content
    assert "Calculator project plan" in plan_content
    # Assert PLAN.md contains single overview header, not multiple appended logs
    assert plan_content.count("# Plan Overview") == 1
