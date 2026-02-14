from pathlib import Path
from typing import Dict, Any, List, Optional

from pkg.agentic.skill_engine.loader import SkillLoader

# 默认 skills 目录：pkg/agentic/skills（与 skill_engine 同层）
_DEFAULT_SKILLS_DIR = str(Path(__file__).resolve().parent.parent / "skills")
_loader: Optional[SkillLoader] = None


def _get_loader() -> SkillLoader:
    global _loader
    if _loader is None:
        _loader = SkillLoader(skills_dir=_DEFAULT_SKILLS_DIR)
    return _loader


def list_skills() -> List[str]:
    """列出可用的技能名称（供 plan 阶段预加载用）"""
    return _get_loader().list_skills()


def load_skill(skill_name: str) -> Dict[str, Any]:
    """加载指定技能文档（描述、能力、子技能、内容摘要等）"""
    return _get_loader().load_skill(skill_name)


def load_sub_skill(skill_name: str) -> List[Dict[str, Any]]:
    """加载指定技能的全体子技能文档"""
    return _get_loader().load_sub_skills(skill_name)