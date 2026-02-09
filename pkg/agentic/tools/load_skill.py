from typing import Dict, Any, List

from pkg.agentic.skill_engine.loader import SkillLoader

def load_skill(skill_name: str):
    """加载指定技能"""
    load = SkillLoader
    return load.load_skill(skill_name)

def load_sub_skill(skill_name: str) -> List[Dict[str, Any]]:
    """加载指定技能"""
    load = SkillLoader
    return load.load_sub_skills(skill_name)