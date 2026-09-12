import os
from typing import List, Optional, Union, Dict, Any


class Planner:
    """
    Explicit state manager for planning instead of an append-only log.
    Overwrites PLAN.md with a structured, condensed Markdown representation
    and enforces anti-repetition / context guard policies.
    """

    def __init__(self, plan_file_path: str = "PLAN.md"):
        self.plan_file_path = plan_file_path
        self.last_plan_state: Optional[str] = None

    def format_plan_markdown(
        self,
        summary: str,
        completed_tasks: List[str],
        current_task: str,
        next_tasks: List[str],
    ) -> str:
        completed_str = "\n".join(f"- [x] {task}" for task in completed_tasks) if completed_tasks else "- None"
        next_str = "\n".join(f"- [ ] {task}" for task in next_tasks) if next_tasks else "- None"
        current_str = f"- [ ] {current_task}" if current_task else "- None"

        md = f"""# Plan Overview

## Summary
{summary}

## Current Task
{current_str}

## Completed Tasks
{completed_str}

## Next Tasks
{next_str}
"""
        return md.strip() + "\n"

    def update_plan(
        self,
        summary: Optional[str] = None,
        completed_tasks: Optional[List[str]] = None,
        current_task: Optional[str] = None,
        next_tasks: Optional[List[str]] = None,
        content: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Updates PLAN.md by atomic overwrite. Accepts either structured parameters
        or raw content string.
        """
        if content is not None and not summary and not current_task:
            new_plan_content = content.strip() + "\n"
        else:
            summary = summary or "No summary provided."
            completed_tasks = completed_tasks or []
            current_task = current_task or "None"
            next_tasks = next_tasks or []
            new_plan_content = self.format_plan_markdown(
                summary=summary,
                completed_tasks=completed_tasks,
                current_task=current_task,
                next_tasks=next_tasks,
            )

        # Anti-repetition check
        if self.last_plan_state is not None and self.last_plan_state.strip() == new_plan_content.strip():
            return {
                "status": "rejected",
                "message": "Plan state is identical to the previous state. No update performed.",
                "updated": False,
            }

        # Write atomically to file
        temp_file = f"{self.plan_file_path}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(new_plan_content)
        os.replace(temp_file, self.plan_file_path)

        self.last_plan_state = new_plan_content
        return {
            "status": "success",
            "message": "PLAN.md updated successfully.",
            "updated": True,
            "content": new_plan_content,
        }

    def truncate_history_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Context guard: keeps system message, user prompt, and recent history,
        compacting/truncating previous update_plan tool calls and their matching tool responses.
        """
        # Find all assistant messages that called update_plan
        plan_tool_call_ids = set()

        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    fn_name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
                    if fn_name == "update_plan":
                        tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                        if tc_id:
                            plan_tool_call_ids.add(tc_id)

        if not plan_tool_call_ids:
            return messages

        # Identify the latest update_plan tool call ID (if any) to preserve
        latest_plan_id = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    fn_name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
                    if fn_name == "update_plan":
                        latest_plan_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                        break
            if latest_plan_id:
                break

        # Filter out earlier update_plan assistant tool calls and their matching tool responses
        pruned = []
        for msg in messages:
            role = msg.get("role")

            if role == "assistant" and msg.get("tool_calls"):
                filtered_tcs = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)

                    if fn_name == "update_plan" and tc_id != latest_plan_id:
                        continue
                    filtered_tcs.append(tc)

                if not filtered_tcs and not msg.get("content"):
                    continue  # Prune entire assistant msg if it only contained old update_plan tool calls

                msg_copy = dict(msg)
                if filtered_tcs:
                    msg_copy["tool_calls"] = filtered_tcs
                else:
                    msg_copy.pop("tool_calls", None)
                pruned.append(msg_copy)
                continue

            if role == "tool":
                call_id = msg.get("tool_call_id")
                # Prune if this is a response to an old update_plan tool call
                if call_id in plan_tool_call_ids and call_id != latest_plan_id:
                    continue

            pruned.append(msg)

        return pruned
