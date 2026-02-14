from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Any, Optional

from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from pkg.agentic.checkpoint.file import FileCheckpointSaver
from pkg.agentic.memory.rag import setup_knowledge_base
from pkg.agentic.model import llm, llm_tools
from pkg.agentic.planner import plan_node
from pkg.agentic.reason import reason_with_knowledge
from pkg.agentic.state import AgentState

from pkg.agentic.tools.manager import list_tools, set_approval_callback

class ReflectiveAgent:
    def __init__(self, model_name="gpt-4", max_iterations=5):
        # 初始化模型
        self.llm = llm  # 使用高逻辑性模型（deepseek-reasoner）
        self.llm_tools = llm_tools  # 工具调用用 deepseek-chat，避免 reasoning_content 400 报错

        # 定义工具集
        self.tools = list_tools()

        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="output"
        )
        # 知识库支持
        self.knowledge_base = setup_knowledge_base()

        # 构建Agent
        self.agent = self.build_agent()

    def build_agent(self):
        """构建LangGraph Agent"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("reason", reason_with_knowledge) # 正常情况下推理应该只有最初一次，后续推理都在 plan 里体现
        workflow.add_node("plan", plan_node)
        workflow.add_node("execute", self.execute_node)
        workflow.add_node("reflect", self.reflect_node)
        workflow.add_node("adjust", self.adjust_node)

        # 设置入口点
        workflow.set_entry_point("reason")

        # 定义边（状态流转）
        workflow.add_edge("reason", "plan")
        workflow.add_edge("plan", "execute")
        workflow.add_edge("execute", "reflect")

        # 条件边：根据反思结果决定下一步
        workflow.add_conditional_edges(
            "reflect",
            self._should_continue,
            {
                "continue": "adjust",
                "complete": END
            }
        )
        workflow.add_edge("adjust", "plan")

        # 添加记忆与人类审核支持（敏感工具内 interrupt 需子图也 checkpoint）
        file_saver = FileCheckpointSaver()
        self.app = workflow.compile(checkpointer=file_saver)

        # 子图 react_agent 使用同一 checkpointer，执行敏感工具时 interrupt 可被主图 resume
        from langgraph.prebuilt import create_react_agent
        # 不设 interrupt_before=["tools"]，工具会直接执行；需要人工审核的工具在 Tool 内部通过 interrupt() 等待 run_resume
        self.react_agent = create_react_agent(
            self.llm_tools,
            self.tools,
            checkpointer=file_saver,
        )

        return self.app

    def execute_node(self, state: AgentState) -> Dict[str, Any]:
        """执行节点：使用工具执行计划（LangGraph tool-calling agent，无需 ReAct 文本格式）。

        工具真实调用链（均在 react_agent.invoke 内部完成）：
        1. react_agent 图包含 agent 节点 + ToolNode(tools) 节点；
        2. LLM（bind_tools 后）根据「执行步骤」决定是否返回 AIMessage(tool_calls=[...])；
        3. 若有 tool_calls，图会路由到 ToolNode，对每个 call 执行：
           tools_by_name[call["name"]].invoke(call_args)  # 即你的 Tool.func 被真实调用
        4. 敏感工具（见 TOOLS_REQUIRING_APPROVAL）在 Tool.func 内会先 interrupt(payload)，
           主图需用 run_resume(config, approved=True/False) 恢复；传入 config 与 thread_id 一致时 resume 会传到该 interrupt。
        """
        # 与主图共用 thread_id，使子图内 interrupt 能被 run_resume 正确恢复
        thread_id = state.get("thread_id") or "thread_1"
        config = {"configurable": {"thread_id": thread_id}}

        # 执行当前步骤（使用带 checkpointer 的 react_agent，以便敏感工具 interrupt 可 resume）
        current_step = state["plan"][0] if state["plan"] else state["current_goal"]
        tool_names = ", ".join(t.name for t in self.tools)
        execute_prompt = f"""执行以下步骤，必须通过调用工具完成，不要仅用文字回答。可用工具：{tool_names}

步骤：{current_step}"""

        agent_result = self.react_agent.invoke(
            {"messages": [HumanMessage(content=execute_prompt)]},
            config=config,
        )

        # 取最后一条 AI 回复作为输出（若中间调用了工具，messages 中会有 AIMessage + ToolMessage 交替）
        messages = agent_result.get("messages", [])
        output = messages[-1].content if messages else ""

        # 可选：打印本轮所有消息，便于确认工具是否被调用及返回（ToolMessage 的 content 即工具执行结果）
        if __debug__:
            for i, m in enumerate(messages):
                kind = type(m).__name__
                if kind == "AIMessage" and getattr(m, "tool_calls", None):
                    print(f"  [{i}] {kind} tool_calls={[tc.get('name') for tc in m.tool_calls]}")
                elif kind == "ToolMessage":
                    print(f"  [{i}] {kind} name={getattr(m, 'name', '')} content={str(getattr(m, 'content', ''))[:200]}")
                else:
                    print(f"  [{i}] {kind} content={str(getattr(m, 'content', ''))[:200]}")
        print("execute_node output", output)

        # 更新状态
        new_results = state.get("tool_results", []) + [{
            "step": current_step,
            "result": output,
            "timestamp": str(datetime.now())
        }]

        # 移除已完成的步骤
        updated_plan = state["plan"][1:] if len(state["plan"]) > 1 else []

        return {
            "tool_results": new_results,
            "plan": updated_plan,
            "iteration": state.get("iteration", 0) + 1
        }

    def reflect_node(self, state: AgentState) -> Dict[str, Any]:
        """反思节点：评估执行效果"""
        prompt = f"""
        请进行反思评估：
        
        目标：{state['current_goal']}
        已执行步骤：{len(state['tool_results'])}
        执行结果：{state['tool_results'][-1]['result'] if state['tool_results'] else '无'}
        
        请回答以下问题：
        1. 当前进度如何？目标完成度？
        2. 遇到了什么问题？
        3. 计划需要调整吗？
        4. 是否需要更多信息？
        
        反思总结：
        """

        response = self.llm.invoke([("human", prompt)])

        reflections = state.get("reflections", []) + [
            f"第{state.get('iteration', 0)}次反思：{response.content[:100]}..."
        ]

        return {
            "reflections": reflections
        }

    def enhanced_reflect_node(self, state: AgentState):
        """增强的反思节点"""
        reflection_types = [
            "效果评估",
            "错误分析",
            "效率评估",
            "改进建议"
        ]

        reflections = []
        for r_type in reflection_types:
            prompt = f"""进行{r_type}：
            目标：{state['current_goal']}
            执行历史：{state['tool_results']}
            """
            response = self.llm.invoke([("human", prompt)])
            reflections.append(f"{r_type}: {response.content}")

        return {"reflections": state["reflections"] + reflections}

    def adjust_node(self, state: AgentState) -> Dict[str, Any]:
        """调整节点：基于反思调整策略"""
        last_reflection = state["reflections"][-1] if state["reflections"] else ""

        prompt = f"""
        基于反思进行调整：
        反思：{last_reflection}
        当前目标：{state['current_goal']}
        
        请提出具体的调整建议：
        1. 计划调整：
        2. 策略优化：
        3. 需要补充的信息：
        """

        response = self.llm.invoke([("human", prompt)])

        return {
            "reflections": state["reflections"] + [f"调整建议：{response.content}"],
            "plan": []  # 清空计划，让plan节点重新制定
        }

    def _should_continue(self, state: AgentState) -> str:
        """判断是否继续执行"""
        # 检查迭代次数限制
        if state.get("iteration", 0) >= state.get("max_iterations", 5):
            return "complete"

        # 检查目标是否完成
        last_result = state.get("tool_results", [{}])[-1]
        if self._is_goal_achieved(state["current_goal"], last_result.get("result", "")):
            return "complete"

        # 检查计划是否为空（表示需要重新规划）
        if not state.get("plan"):
            return "continue"

        return "continue"

    def _is_goal_achieved(self, goal: str, result: str) -> bool:
        """简单判断目标是否达成"""
        prompt = f"""
        目标：{goal}
        当前结果：{result}
        
        目标是否已达成？回答 '是' 或 '否'
        """

        response = self.llm.invoke([("human", prompt)])
        return "是" in response.content

    # ========== 运行方法 ==========

    def run(
        self,
        input_text: str,
        config: Optional[Dict] = None,
        approval_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """运行Agent。
        approval_callback: 若提供，敏感工具会同步调用 callback(payload) 获取是否批准，不再阻塞在 interrupt，便于 CLI/前端做人工审核入口。
        若不提供，敏感工具内会 interrupt(payload)，需在别处调用 run_resume(config, approved=...) 恢复。"""
        config = config or {"configurable": {"thread_id": "thread_1"}}
        thread_id = config.get("configurable", {}).get("thread_id", "thread_1")

        if approval_callback is not None:
            set_approval_callback(approval_callback)
        try:
            initial_state = {
                "messages": [("human", input_text)],
                "current_goal": input_text,
                "plan": [],
                "reflections": [],
                "tool_results": [],
                "iteration": 0,
                "max_iterations": 10,
                "is_complete": False,
                "thread_id": thread_id,
            }
            # 用 stream(stream_mode=["updates", "values"]) 既拿到最终状态，又保留每步进度输出
            # 最后一个 mode=="values" 的 payload 即为最终状态（与 invoke() 一致）
            final_state = None
            for chunk in self.agent.stream(
                initial_state, config, stream_mode=["updates", "values"]
            ):
                # chunk 为 (mode, payload) 或 (ns, mode, payload)
                if len(chunk) == 2:
                    mode, payload = chunk
                else:
                    _, mode, payload = chunk
                if mode == "updates" and isinstance(payload, dict):
                    for node_name in payload:
                        if not node_name.startswith("_"):
                            print(f"节点 {node_name} 完成")
                elif mode == "values":
                    final_state = payload
            # 若因人类审核而暂停，stream 会提前结束并可能抛出/返回中断；需用 run_resume 恢复
            return final_state if final_state is not None else initial_state
        finally:
            if approval_callback is not None:
                set_approval_callback(None)

    def run_resume(self, config: Optional[Dict] = None, approved: bool = True, feedback: str = "") -> Dict[str, Any]:
        """人类审核后恢复执行。approved=True 继续执行该敏感工具，False 则取消该次工具调用。
        传入的 config 需与 run() 时一致（含相同 thread_id），否则 resume 可能无法传到子图。"""
        from langgraph.types import Command

        config = config or {"configurable": {"thread_id": "thread_1"}}
        resume_value = {"approved": approved, "feedback": feedback} if feedback else approved
        result = self.agent.invoke(Command(resume=resume_value), config=config)
        return result

    def parallel_execute(self, state: AgentState):
        """并行执行多个工具"""
        if len(state["plan"]) <= 1:
            return self.execute_node(state)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for step in state["plan"][:3]:  # 并行执行前3步
                future = executor.submit(self._execute_single_step, step, state)
                futures.append(future)

            results = [f.result() for f in futures]

        return {"tool_results": state["tool_results"] + results}