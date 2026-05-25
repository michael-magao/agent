from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from pkg.agentic.model import llm
from pkg.agentic.skill_engine.loader import SkillLoader
from pkg.agentic.state import AgentState
from pkg.agentic.tools.manager import list_tools

DEFAULT_SKILLS_DIR = str(Path(__file__).resolve().parent / "skills")


def _format_skill_data(skill_data: Dict[str, Any]) -> str:
    parts = [f"【{skill_data.get('name', '')}】"]
    if skill_data.get("description"):
        parts.append(f"描述: {skill_data['description']}")
    if skill_data.get("capabilities"):
        parts.append("能力: " + "; ".join(skill_data["capabilities"]))
    if skill_data.get("sub_skills"):
        parts.append("子技能: " + ", ".join(skill_data["sub_skills"]))
    if skill_data.get("parameters"):
        parts.append("参数: " + str(skill_data["parameters"]))
    if skill_data.get("content_snippet"):
        parts.append(str(skill_data["content_snippet"])[:1200])
    return "\n".join(parts)


@lru_cache(maxsize=4)
def _gather_skill_docs(skills_dir: str = DEFAULT_SKILLS_DIR) -> str:
    """预加载 skill 文档摘要，结果缓存，避免每轮规划都重新扫文件。"""
    parts: List[str] = []

    try:
        loader = SkillLoader(skills_dir)
        skill_names = loader.list_skills()
    except (FileNotFoundError, OSError, KeyError):
        skill_names = []
        loader = None

    if loader and skill_names:
        for name in skill_names:
            try:
                parts.append(_format_skill_data(loader.load_skill(name)))
                for sub in loader.load_sub_skills(name):
                    parts.append(_format_skill_data(sub))
            except Exception as e:
                parts.append(f"[Skill '{name}' 加载失败: {e}]")

    return "\n\n---\n\n".join(parts)


def _extract_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _parse_plan(content: str) -> List[str]:
    try:
        data = _extract_json(content)
        if isinstance(data, dict):
            steps = data.get("steps") or data.get("plan") or []
        else:
            steps = data
        if isinstance(steps, list):
            parsed = []
            for step in steps:
                if isinstance(step, dict):
                    tool = step.get("tool")
                    action = step.get("action") or step.get("step") or step.get("description")
                    params = step.get("params") or step.get("arguments") or step.get("args")
                    text = f"{action or ''}".strip()
                    if tool:
                        text = f"{text}；工具：{tool}" if text else f"工具：{tool}"
                    if params:
                        text = f"{text}；参数意图：{params}"
                    if text:
                        parsed.append(text)
                elif isinstance(step, str) and step.strip():
                    parsed.append(step.strip())
            if parsed:
                return parsed[:5]
    except (TypeError, ValueError, json.JSONDecodeError):
        pass

    lines = []
    for line in content.splitlines():
        text = line.strip()
        if re.match(r"^(\d+[\.\)]|[-*])\s+", text):
            lines.append(re.sub(r"^(\d+[\.\)]|[-*])\s+", "", text).strip())
    return [line for line in lines if line][:5]


def plan_node(state: AgentState) -> Dict[str, Any]:
    """规划节点：使用结构化输出制定执行计划。"""
    skill_docs = _gather_skill_docs()
    tool_names = [t.name for t in list_tools()]
    skill_section = f"\n已预加载的 Skill 文档摘要：\n{skill_docs}\n" if skill_docs else ""

    prompt = f"""
你是一个任务规划器。请基于目标、历史反思和可用工具制定可执行计划。

目标：{state['current_goal']}
历史反思：{state.get('reflections', [])[-3:]}
已执行步骤数：{len(state.get('tool_results', []))}
可用工具：{tool_names}
{skill_section}

请只输出 JSON，不要输出 Markdown。格式如下：
{{
  "steps": [
    {{"tool": "工具名", "action": "要做什么", "params": "关键参数或参数意图"}}
  ]
}}
要求：3-5 个步骤；每一步必须对应一个可用工具；如果需要参考 Skill 文档，请体现在 action 中。
"""

    response = llm.invoke([("human", prompt)])
    plan_lines = _parse_plan(response.content)

    return {
        "plan": plan_lines,
        "reflections": state.get("reflections", []) + [f"制定计划：{len(plan_lines)}个步骤"],
    }
