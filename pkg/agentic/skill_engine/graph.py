# skill_graph.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from typing import Dict, Any

class SkillAgentGraph:
    def __init__(self, skill_registry: SkillRegistry):
        self.registry = skill_registry
        self.tool_executor = ToolExecutor([tool for tool in skill_registry.tools.values()])

        # 构建图
        self.graph = self._build_graph()
        self.compiled = self.graph.compile()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # 添加节点
        planner = PlannerNode(self.registry)
        workflow.add_node("planner", planner)
        workflow.add_node("execute_tool", self._execute_tool)
        workflow.add_node("process_result", self._process_result)
        workflow.add_node("finalize", self._finalize)

        # 设置入口
        workflow.set_entry_point("planner")

        # 添加条件边
        workflow.add_conditional_edges(
            "planner",
            self._route_after_plan,
            {
                "use_tool": "execute_tool",
                "respond": "finalize",
                "continue": "planner"  # 可能继续规划
            }
        )

        workflow.add_edge("execute_tool", "process_result")
        workflow.add_edge("process_result", "planner")  # 执行后重新规划
        workflow.add_edge("finalize", END)

        return workflow

    def _route_after_plan(self, state: AgentState) -> str:
        """根据Planner的决策路由"""
        if state.get("final_output"):
            return "respond"
        elif state.get("next_action"):
            return "use_tool"
        else:
            return "continue"  # 继续规划

    async def _execute_tool(self, state: AgentState) -> AgentState:
        """执行技能工具"""
        action = state["next_action"]
        tool_name = action["tool"]

        # 创建工具调用
        tool_invocation = ToolInvocation(
            tool=tool_name,
            tool_input=action["input"]
        )

        # 执行
        observation = await self.tool_executor.ainvoke(tool_invocation)

        return {
            **state,
            "observation": str(observation),
            "next_action": None  # 清空，等待下次规划
        }

    async def _process_result(self, state: AgentState) -> AgentState:
        """处理工具执行结果，可以在这里添加日志或后处理"""
        # 将观察结果添加到消息历史，供下一轮规划参考
        messages = state["messages"]
        messages.append(AIMessage(content=f"工具执行结果：{state['observation']}"))

        return {
            **state,
            "messages": messages
        }

    async def _finalize(self, state: AgentState) -> AgentState:
        """生成最终回复"""
        messages = state["messages"]
        messages.append(AIMessage(content=state["final_output"]))

        return {
            **state,
            "messages": messages
        }