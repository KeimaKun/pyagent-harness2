import os
import pytest
from harness.planner import Planner
from harness.guard import WorkspaceGuard
from harness.tools import ToolRegistry


def test_planner_update_plan_structured(tmp_path):
    plan_file = tmp_path / "PLAN.md"
    planner = Planner(plan_file_path=str(plan_file))

    res = planner.update_plan(
        summary="Building calculator app",
        completed_tasks=["Initial setup"],
        current_task="Implement calc.py",
        next_tasks=["Run tests"],
    )

    assert res["status"] == "success"
    assert res["updated"] is True
    assert plan_file.exists()

    content = plan_file.read_text(encoding="utf-8")
    assert "# Plan Overview" in content
    assert "Building calculator app" in content
    assert "- [x] Initial setup" in content
    assert "- [ ] Implement calc.py" in content
    assert "- [ ] Run tests" in content


def test_planner_update_plan_raw_content(tmp_path):
    plan_file = tmp_path / "PLAN.md"
    planner = Planner(plan_file_path=str(plan_file))

    res = planner.update_plan(content="# Custom Raw Plan\n- Task 1\n- Task 2\n")
    assert res["status"] == "success"
    assert plan_file.read_text(encoding="utf-8") == "# Custom Raw Plan\n- Task 1\n- Task 2\n"


def test_planner_anti_repetition(tmp_path):
    plan_file = tmp_path / "PLAN.md"
    planner = Planner(plan_file_path=str(plan_file))

    res1 = planner.update_plan(summary="Same plan", current_task="Task A")
    assert res1["updated"] is True

    res2 = planner.update_plan(summary="Same plan", current_task="Task A")
    assert res2["status"] == "rejected"
    assert res2["updated"] is False


def test_workspace_guard_security(tmp_path):
    guard = WorkspaceGuard(workspace_root=str(tmp_path))

    # Safe path inside workspace
    safe_file = guard.validate_path("sub/file.txt")
    assert str(safe_file).startswith(str(tmp_path.resolve()))

    # Directory traversal attempts
    with pytest.raises(PermissionError):
        guard.validate_path("../outside.txt")

    with pytest.raises(PermissionError):
        guard.validate_path("../../etc/passwd")


def test_tool_registry_file_operations(tmp_path):
    guard = WorkspaceGuard(workspace_root=str(tmp_path))
    tools = ToolRegistry(guard=guard)

    # Write file
    w_res = tools.write_file("test.txt", "line1\nline2\nline3\n")
    assert w_res["status"] == "success"

    # Read file
    r_res = tools.read_file("test.txt", offset=1, limit=2)
    assert r_res["content"] == "line2\nline3\n"

    # Apply patch
    p_res = tools.apply_patch("test.txt", "line2", "line2_modified")
    assert p_res["status"] == "success"

    r_res2 = tools.read_file("test.txt")
    assert "line2_modified" in r_res2["content"]

    # Traversal in tool registry rejected
    bad_read = tools.read_file("../secret.txt")
    assert "error" in bad_read
