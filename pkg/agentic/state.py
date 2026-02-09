from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

# class AgentState:
#     def __init__(
#         self,
#         current_goal: str = "",
#         plan: Optional[List[str]] = None,
#         reflections: Optional[List[str]] = None,
#         tool_results: Optional[List[Dict[str, Any]]] = None,
#         iteration: int = 0,
#         max_iterations: int = 10,
#         is_complete: bool = False,
#         ) -> None:
#         self.current_goal = current_goal  # 当前目标
#         self.plan = plan if plan is not None else []  # 执行计划
#         self.reflections = reflections if reflections is not None else []  # 反思记录
#         self.tool_results = tool_results if tool_results is not None else []  # 工具执行结果
#         self.iteration = iteration  # 迭代次数
#         self.max_iterations = max_iterations  # 最大迭代次数
#         self.is_complete = is_complete  # 是否完成
#
#     # reflection implementation
#     def task_completed(self) -> bool:
#         if self.iteration >= self.max_iterations:
#             self.is_complete = True
#         return self.is_complete


class AgentState(TypedDict, total=False):
    """Agent状态定义"""
    messages: Annotated[List[BaseMessage], add_messages]  # 对话历史
    current_goal: str  # 当前目标
    plan: List[str]  # 执行计划
    reflections: List[str]  # 反思记录
    tool_results: List[Dict[str, Any]]  # 工具执行结果
    iteration: int  # 迭代次数
    max_iterations: int  # 最大迭代次数
    is_complete: bool  # 是否完成

    # 人类审核相关（可选）：工具执行前的审核
    # review_status: None 无需审核, "PENDING" 等待审核, "APPROVED" 已批准, "REJECTED" 已拒绝
    review_status: Optional[str]
    review_feedback: str  # 审核员反馈

    # 用于子图（react_agent）与主图共享 checkpoint，以便 interrupt/resume 生效
    thread_id: Optional[str]