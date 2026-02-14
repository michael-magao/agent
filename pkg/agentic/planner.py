# planner_node.py
from pathlib import Path
from typing import Dict, Any, List

from pkg.agentic.model import llm
from pkg.agentic.skill_engine.loader import SkillLoader
from pkg.agentic.skill_engine.register import SkillRegistry2
from pkg.agentic.state import AgentState
from pkg.agentic.tools.manager import list_tools
from pkg.agentic.tools.load_skill import list_skills, load_skill, load_sub_skill


# class PlannerNode:
#     def __init__(self, skill_registry: SkillRegistry2):
#         self.registry = skill_registry
#         self.llm = ChatOpenAI(model="gpt-4", temperature=0)
#
#         self.prompt = ChatPromptTemplate.from_messages([
#             SystemMessage(content=f"""你是一个任务规划器。根据用户请求，决定是否需要调用技能工具。
#
#             可用的技能工具：
#             {self.registry.get_tool_descriptions()}
#
#             请按以下JSON格式回复：
#             {{
#                 "reasoning": "你的思考过程",
#                 "needs_tool": true/false,
#                 "next_action": {{
#                     "tool": "工具名或null",
#                     "input": {{"arg1": "value1"}} 或 null
#                 }},
#                 "response_to_user": "如果不需要工具，直接回复用户的话"
#             }}
#             如果不需要工具，"next_action"设为null，并在"response_to_user"中直接回复。"""),
#             MessagesPlaceholder(variable_name="messages"),
#         ])
#
#     async def __call__(self, state: AgentState) -> AgentState:
#         """Planner节点的核心逻辑：分析状态，决定下一步"""
#         chain = self.prompt | self.llm
#         response = await chain.ainvoke({"messages": state["messages"]})
#
#         try:
#             decision = json.loads(response.content)
#         except:
#             decision = {"needs_tool": False, "response_to_user": response.content}
#
#         new_state = {**state}
#
#         if decision.get("needs_tool", False):
#             # 需要调用工具
#             new_state["plan"] = decision.get("reasoning", "")
#             new_state["next_action"] = decision["next_action"]
#             new_state["final_output"] = None
#         else:
#             # 直接回复
#             new_state["final_output"] = decision.get("response_to_user", response.content)
#             new_state["next_action"] = None
#
#         return new_state

# todo 后续集成Skill可以加上这句话：“如果你需要获取更多的Skill，强烈建议先通过工具: {load_skill} 来获取相关的技能说明文档，来更好的制定计划。”

# 计划阶段预加载 skill 时使用的技能目录（可与 load_skill 工具一致）
DEFAULT_SKILLS_DIR = "pkg/agentic/skills"

def _format_skill_data(skill_data: Dict[str, Any]) -> str:
    """将 load_skill 返回的字典格式化为可读文本"""
    parts = [f"【{skill_data.get('name', '')}】"]
    if skill_data.get("description"):
        parts.append(f"描述: {skill_data['description']}")
    if skill_data.get("capabilities"):
        parts.append("能力: " + "; ".join(skill_data["capabilities"]))
    if skill_data.get("sub_skills"):
        parts.append("子技能: " + ", ".join(skill_data["sub_skills"]))
    if skill_data.get("parameters"):
        parts.append("参数: " + str(skill_data["parameters"]))
    return "\n".join(parts)


def _gather_skill_docs(skills_dir: str = DEFAULT_SKILLS_DIR) -> str:
    """在 plan 阶段预加载 skill 文档：尝试用 load_skill / load_sub_skill 获取技能说明与可用工具信息。
    若存在 skill_metadata.json 则通过 SkillLoader 加载；否则从目录扫描 Skill.md 作为回退。
    返回格式化后的文档字符串，供制定计划时参考。
    """
    parts: List[str] = []

    try:
        loader = SkillLoader(skills_dir)
        skill_names = list(loader.metadata.get("skills", {}).keys())
    except (FileNotFoundError, OSError, KeyError):
        skill_names = []
        loader = None

    if loader and skill_names:
        for name in skill_names:
            try:
                skill_data = loader.load_skill(name)
                parts.append(_format_skill_data(skill_data))
                try:
                    sub_list = loader.load_sub_skills(name)
                    for sub in sub_list:
                        parts.append(_format_skill_data(sub))
                except Exception:
                    pass
            except Exception as e:
                parts.append(f"[Skill '{name}' 加载失败: {e}]")
    else:
        # 回退：从目录扫描 Skill.md
        base = Path(skills_dir)
        if not base.exists():
            return ""
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            skill_md = d / "Skill.md"
            if skill_md.exists():
                try:
                    content = skill_md.read_text(encoding="utf-8")
                    parts.append(f"【{d.name}】\n{content[:2000]}" + ("..." if len(content) > 2000 else ""))
                except Exception as e:
                    parts.append(f"[{d.name} 读取失败: {e}]")
            for sub in sorted(d.iterdir()):
                if not sub.is_dir():
                    continue
                sub_md = sub / "Skill.md"
                if sub_md.exists():
                    try:
                        content = sub_md.read_text(encoding="utf-8")
                        parts.append(f"【{d.name}/{sub.name}】\n{content[:1500]}" + ("..." if len(content) > 1500 else ""))
                    except Exception as e:
                        parts.append(f"[{d.name}/{sub.name} 读取失败: {e}]")

    if not parts:
        return ""
    return "\n\n---\n\n".join(parts)


def plan_node(state: AgentState) -> Dict[str, Any]:
    """规划节点：制定执行计划。会先预加载 skill 文档，再基于目标与可用工具生成计划。"""
    # 计划阶段先尝试用 load_skill / load_sub_skill 获取 skill 文档
    skill_docs = _gather_skill_docs()

    tools = list_tools()
    tool_names = [t.name for t in tools]
    skill_section = ""
    if skill_docs:
        skill_section = f"""
        已预加载的 Skill 文档（执行任务所需信息与可参考工具说明，请优先参考后再制定计划）：
        {skill_docs}
        """

    prompt = f"""
        你是一个优秀的计划制定者。
        基于以下目标制定执行计划：
        目标：{state['current_goal']}
        历史反思：{state.get('reflections', [])[-3:]}
        已执行步骤：{len(state.get('tool_results', []))}

        可用工具：{tool_names}
        {skill_section}
        计划中的每一步可以通过调用上述工具之一来完成（例如：用 query_cluster_detail 查集群、query_monitor_detail 查监控、query_log_info 查日志、search_sop 查SOP）。
        若已预加载的 Skill 文档中有与目标相关的流程、参考文档或工具说明，请在计划中体现。
        
        请提供3-5个步骤的详细计划，每步明确要调用的工具和参数意图：
        1. 
        2. 
        3. 
        """

    response = llm.invoke([("human", prompt)])

    # 解析计划步骤
    plan_lines = [line.strip() for line in response.content.split('\n') if line.strip().startswith(('1.', '2.', '3.', '4.', '5.'))]

    # print("plan_node info:", plan_lines)
    return {
        "plan": plan_lines,
        "reflections": state.get("reflections", []) + [f"制定计划：{len(plan_lines)}个步骤"]
    }