"""Render a validated structured summary as readable Markdown."""

from __future__ import annotations
from typing import Any


def _value(value: Any) -> str:
    if value is None:
        return "未提及"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "未提及"
    return str(value)


def render_summary(payload: dict[str, Any]) -> str:
    lines = ["# 会议纪要", "", "## 摘要", "", _value(payload.get("summary")), "", "## 讨论主题", ""]
    topics = payload.get("topics") or ["未提及"]
    lines.extend(f"- {_value(item.get('topic') if isinstance(item, dict) else item)}" for item in topics)
    lines.extend(["", "## 决策", ""])
    for item in payload.get("decisions") or ["未提及"]:
        if isinstance(item, dict):
            evidence = ", ".join(item.get("evidence_segment_ids") or []) or "未提及"
            lines.append(f"- {_value(item.get('decision'))}（证据: {evidence}）")
        else:
            lines.append(f"- {_value(item)}")
    lines.extend(["", "## 行动项", ""])
    for item in payload.get("action_items") or ["未提及"]:
        if isinstance(item, dict):
            evidence = ", ".join(item.get("evidence_segment_ids") or []) or "未提及"
            lines.append(f"- {_value(item.get('action'))}；负责人: {_value(item.get('owner'))}；截止日期: {_value(item.get('due_date'))}；状态: {_value(item.get('status'))}；证据: {evidence}")
        else:
            lines.append(f"- {_value(item)}")
    lines.extend(["", "## 未解决问题", ""])
    lines.extend(f"- {_value(item)}" for item in (payload.get("unresolved_questions") or ["未提及"]))
    lines.extend(["", "## 模型推断（需确认）", ""])
    lines.extend(f"- {_value(item)}" for item in (payload.get("inferences") or ["未提及"]))
    lines.extend(["", "## 缺失信息", ""])
    lines.extend(f"- {_value(item)}" for item in (payload.get("missing_information") or ["未提及"]))
    return "\n".join(lines).rstrip() + "\n"
