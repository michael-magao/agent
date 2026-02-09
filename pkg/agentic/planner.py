# planner_node.py
from typing import Dict, Any


from pkg.agentic.model import llm
from pkg.agentic.skill_engine.register import SkillRegistry2
from pkg.agentic.state import AgentState
from pkg.agentic.tools.manager import list_tools


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


def plan_node(state: AgentState) -> Dict[str, Any]:
    """规划节点：制定执行计划"""
    tools = list_tools()
    tool_names = [t.name for t in tools]
    prompt = f"""
        你是一个优秀的计划制定者。
        基于以下目标制定执行计划：
        目标：{state['current_goal']}
        历史反思：{state.get('reflections', [])[-3:]}
        已执行步骤：{len(state.get('tool_results', []))}

        可用工具：{tool_names}
    
        如果你需要获取更多的Skill，强烈建议先通过工具: "load_skill" 和 "load_sub_skill" 来获取相关的技能说明文档，来更好的制定计划。
        
        计划中的每一步可以通过调用上述工具之一来完成（例如：用 query_cluster_detail 查集群、query_monitor_detail 查监控、query_log_info 查日志、search_sop 查SOP）。
        
        请提供3-5个步骤的详细计划，每步明确要调用的工具和参数意图：
        1. 
        2. 
        3. 
        """

    # todo 生成计划的过程中，可以渐进式加载skill信息，获取指导手册信息和可用工具

    # print("plan_node prompt:", prompt)
    response = llm.invoke([("human", prompt)])

    # 解析计划步骤
    plan_lines = [line.strip() for line in response.content.split('\n') if line.strip().startswith(('1.', '2.', '3.', '4.', '5.'))]

    # print("plan_node info:", plan_lines)
    return {
        "plan": plan_lines,
        "reflections": state.get("reflections", []) + [f"制定计划：{len(plan_lines)}个步骤"]
    }