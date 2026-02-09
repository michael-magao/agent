import contextvars
from typing import List, Callable, Any, Optional
from langchain_core.tools import Tool
from sympy import false

from pkg.agentic.tools.load_skill import load_skill, load_sub_skill
from pkg.agentic.tools.knots_query import query_cluster_detail
from pkg.agentic.tools.log_query import query_log_info
from pkg.agentic.tools.monitor import query_monitor_detail
from pkg.agentic.tools.rag_knowledge import search_sop_knowledge


# 需要人类审核的工具名（执行前会 interrupt 等待人类批准，批准后才真实执行）
# 新增敏感工具：把工具名加入此集合，或在该 Tool 的 metadata 中设置 "requires_approval": True
TOOLS_REQUIRING_APPROVAL: set = {"search_sop"}

# 运行时可选：若在 run() 时设置了审核回调，工具会先调用该回调（同步、可阻塞），不再走 LangGraph interrupt，从而有明确的人工审核入口
_approval_callback_ctx: contextvars.ContextVar[Optional[Callable[[dict], Any]]] = contextvars.ContextVar(
    "approval_callback", default=None
)


def set_approval_callback(callback: Optional[Callable[[dict], Any]]) -> None:
    """设置当前线程/上下文的审核回调。run() 中传入后，敏感工具会调用此回调获取批准结果，不再阻塞在 interrupt()。"""
    _approval_callback_ctx.set(callback)


def get_approval_callback() -> Optional[Callable[[dict], Any]]:
    return _approval_callback_ctx.get(None)


def _with_human_approval(tool_name: str, func: Callable[..., Any], description: str) -> Callable[..., str]:
    """包装工具：执行前等待人类审核。若已通过 set_approval_callback 设置回调则同步走回调；否则通过 LangGraph interrupt() 等待 run_resume。"""

    def wrapper(*args: Any, **kwargs: Any) -> str:
        # 供前端/审核端展示的 payload（JSON 可序列化）
        payload = {
            "tool": tool_name,
            "description": description,
            "args": kwargs if kwargs else (list(args) if args else []),
            "message": f"工具「{tool_name}」需要人工审核，是否批准执行？",
        }
        print("请求人工审核，信息:", payload)

        callback = get_approval_callback()
        if callback is not None:
            # 有审核回调：同步调用，不阻塞在 interrupt，审核入口明确
            approved = callback(payload)
            print("审核结果（回调）:", approved)
        else:
            # 无回调：走 LangGraph interrupt，需主流程在别处 run_resume 才能继续
            from langgraph.types import interrupt
            approved = interrupt(payload)
            print("审核结果:", approved)

        if approved is True or (isinstance(approved, dict) and approved.get("approved", False)):
            try:
                if kwargs:
                    result = func(**kwargs)
                else:
                    result = func(*args) if args else func()
                return str(result) if result is not None else "执行完成"
            except Exception as e:
                return f"执行失败: {e!s}"
        return "操作已由用户取消"

    return wrapper


def list_tools() -> List[Tool]:
    """定义 Agent 可用的工具。需人类审核的工具会先 interrupt，审核通过后才执行。

    如何增加「执行前人类校验」的敏感工具：
    - 在 TOOLS_REQUIRING_APPROVAL 中加入工具名（如 "my_risky_tool"），或
    - 定义 Tool 时设置 metadata={"requires_approval": True}
    二者满足其一即可走 _with_human_approval 包装（先 interrupt，再根据 resume 决定是否执行）。
    """
    raw_tools = [
        Tool(
            name="search_sop",
            func=search_sop_knowledge,
            description="搜索SOP最新信息",
            metadata={"requires_approval": False},
        ),
        Tool(
            name="calculator",
            func=lambda x: str(eval(x)),
            description="计算数学表达式",
            metadata={"requires_approval": False},
        ),
        Tool(
            name="query_cluster_detail",
            func=query_cluster_detail,
            description="查询集群详细信息",
            metadata={"requires_approval": True},
        ),
        Tool(
            name="query_log_info",
            func=query_log_info,
            description="查询日志信息",
            metadata={"requires_approval": True},
        ),
        Tool(
            name="query_monitor_detail",
            func=query_monitor_detail,
            description="查询监控信息",
            metadata={"requires_approval": True},
        ),
        Tool(
            name="load_skill",
            func=load_skill,
            description="加载Skill技能信息",
            metadata={"requires_approval": False},
        ),
        Tool(
            name="load_sub_skill",
            func=load_sub_skill,
            description="加载SubSkill技能信息",
            metadata={"requires_approval": False},
        )
    ]

    tools: List[Tool] = []
    for t in raw_tools:
        meta = t.metadata or {}
        if meta.get("requires_approval") or t.name in TOOLS_REQUIRING_APPROVAL:
            wrapped = _with_human_approval(t.name, t.func, t.description)
            tools.append(Tool(name=t.name, func=wrapped, description=t.description, metadata=t.metadata))
        else:
            tools.append(t)
    return tools