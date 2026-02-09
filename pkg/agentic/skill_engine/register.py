import hashlib
import json
import os
from typing import Dict, Any

from pkg.agentic.skill_engine.metadata import SkillDefinition


class SkillRegistry:
    """技能注册中心 - 管理所有技能"""
    def __init__(self, skills_root: str = "./skills"):
        self.skills_root = skills_root
        self.skills: Dict[str, SkillDefinition] = {}
        self.graph_cache: Dict[str, Any] = {}  # 缓存编译后的图
        self._discover_skills()

    def _discover_skills(self):
        """发现并加载所有技能"""
        if not os.path.exists(self.skills_root):
            os.makedirs(self.skills_root)

        for skill_dir in os.listdir(self.skills_root):
            skill_path = os.path.join(self.skills_root, skill_dir)
            if os.path.isdir(skill_path) and os.path.exists(os.path.join(skill_path, "SKILL.md")):
                try:
                    skill = SkillDefinition(skill_path)
                    self.skills[skill.metadata.name] = skill
                    print(f"加载技能: {skill.metadata.name}")
                except Exception as e:
                    print(f"加载技能 {skill_dir} 失败: {e}")

    def get_skill_graph(self, skill_name: str, available_tools: Dict) -> Any:
        """获取技能对应的图（优先从缓存读取）"""
        # 生成缓存键
        cache_key = f"{skill_name}_{hashlib.md5(json.dumps(list(available_tools.keys())).encode()).hexdigest()}"

        if cache_key not in self.graph_cache:
            if skill_name not in self.skills:
                raise ValueError(f"技能 '{skill_name}' 未找到")

            skill = self.skills[skill_name]
            graph = skill.build_graph(available_tools)
            self.graph_cache[cache_key] = graph

        return self.graph_cache[cache_key]


# skill_registry.py
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
from langchain.tools import StructuredTool
from langchain_core.tools import BaseTool

class SkillRegistry2:
    def __init__(self, skills_dir: str = "./skills"):
        self.skills_dir = Path(skills_dir)
        self.registry: Dict[str, Dict] = {} # 技能元数据缓存
        self.tools: Dict[str, BaseTool] = {} # 实例化工具
        self._load_all_skills()

    def _parse_skill_md(self, skill_path: Path) -> Dict[str, Any]:
        """解析SKILL.md中的YAML元数据头"""
        content = skill_path.read_text(encoding='utf-8')
        # 假设YAML头在 --- 之间
        if content.startswith('---'):
            parts = content.split('---', 2)
            metadata = yaml.safe_load(parts[1])
            return metadata
        return {}

    def _load_skill(self, skill_name: str):
        """加载单个技能并注册为LangChain Tool"""
        skill_dir = self.skills_dir / skill_name
        tool_def_path = skill_dir / "tool_definition.json"

        with open(tool_def_path, 'r') as f:
            tool_def = json.load(f)

        # 动态创建工具函数
        def skill_function(**kwargs):
            # 这里可以动态执行技能包内的脚本
            # 例如: 调用 analyze_script.py 并传入 kwargs
            result = self._execute_skill_internal(skill_name, kwargs)
            return result

        # 创建结构化工具
        tool = StructuredTool.from_function(
            name=tool_def["name"],
            description=tool_def["description"],
            args_schema=self._create_args_schema(tool_def["args_schema"]),
            func=skill_function
        )

        # 存入注册表
        self.registry[tool_def["name"]] = {
            "skill_dir": skill_name,
            "metadata": tool_def,
            "tool": tool
        }
        self.tools[tool_def["name"]] = tool

    def get_tool_names(self) -> List[str]:
        return list(self.tools.keys())

    def get_tool_descriptions(self) -> str:
        """生成给Planner看的技能描述列表"""
        desc = []
        for name, info in self.registry.items():
            desc.append(f"- {name}: {info['metadata']['description']}")
        return "\n".join(desc)

    # todo 真实的调用tool
    def _execute_skill_internal(self, skill_name: str, args: dict):
        """实际执行技能的逻辑（可根据需要扩展）"""
        # 这里可以实现：调用子进程、导入模块、运行脚本等
        skill_dir = self.skills_dir / skill_name
        # 示例：简单返回
        return f"执行技能 {skill_name}，参数：{args}"