import ast
import contextvars
import math
import operator
from typing import List, Callable, Any, Optional

from langchain_core.tools import Tool

from pkg.agentic.tools.load_skill import load_skill, load_sub_skill
from pkg.agentic.tools.knots_query import query_cluster_detail
from pkg.agentic.tools.amazon_product import amazon_search_products, amazon_analyze_for_product_dev
from pkg.agentic.tools.log_query import query_log_info
from pkg.agentic.tools.monitor import query_monitor_detail
from pkg.agentic.tools.rag_knowledge import search_sop_knowledge
from pkg.agentic.tools.web_search import web_search


# 需要人类审核的工具名（执行前会 interrupt 等待人类批准，批准后才真实执行）
# 新增敏感工具：把工具名加入此集合，或在该 Tool 的 metadata 中设置 "requires_approval": True
TOOLS_REQUIRING_APPROVAL: set = {"search_sop"}

# 运行时可选：若在 run() 时设置了审核回调，工具会先调用该回调（同步、可阻塞），不再走 LangGraph interrupt，从而有明确的人工审核入口
_approval_callback_ctx: contextvars.ContextVar[Optional[Callable[[dict], Any]]] = contextvars.ContextVar(
    "approval_callback", default=None
)

_ALLOWED_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "max": max,
    "min": min,
    "pow": pow,
    "round": round,
    "sqrt": math.sqrt,
}
_ALLOWED_CONSTANTS = {
    "e": math.e,
    "pi": math.pi,
}


def calculate_expression(expression: str) -> str:
    """安全计算基础数学表达式，拒绝属性访问、导入和任意函数调用。"""
    if not isinstance(expression, str) or not expression.strip():
        return "表达式不能为空"
    if len(expression) > 500:
        return "表达式过长"

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY_OPS:
            return _ALLOWED_BINARY_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
            return _ALLOWED_UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Name) and node.id in _ALLOWED_CONSTANTS:
            return _ALLOWED_CONSTANTS[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCTIONS:
            args = [_eval(arg) for arg in node.args]
            if node.keywords:
                raise ValueError("不支持关键字参数")
            return _ALLOWED_FUNCTIONS[node.func.id](*args)
        raise ValueError(f"不支持的表达式: {ast.dump(node, include_attributes=False)}")

    try:
        result = _eval(ast.parse(expression, mode="eval"))
    except Exception as e:
        return f"计算失败: {e}"
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


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

# todo 对于工具的定义需要更加明确，最好有参数和返回值的规范，方便后续集成到技能系统中（例如通过 tool_definition.json 定义参数结构，或直接在 Tool 的 metadata 中加入参数说明）。目前先简单实现功能，后续再迭代完善。
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
            func=calculate_expression,
            description="安全计算基础数学表达式，支持 + - * / // % ** 和 sqrt/log/round 等常用函数",
            metadata={"requires_approval": False},
        ),
        Tool(
            name="query_cluster_detail",
            func=query_cluster_detail,
            description=(
                "查询 Knots dc-admin 集群详情。输入集群名，"
                "或 JSON: {\"cluster\":\"...\",\"service\":\"zookeeper|etcd|v2\"}"
            ),
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
            description=(
                "查询 Knots 集群监控。输入集群名，或 JSON: "
                "{\"cluster\":\"...\",\"metrics\":[\"cpu\",\"memory\"],\"regions\":[\"sg\",\"us\"]}"
            ),
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
        ),
        Tool(
            name="amazon_search_products",
            func=lambda x: amazon_search_products(x, "US"),
            description="搜索亚马逊商品。输入：搜索关键词（如 wireless earbuds）。基于 PA-API 5.0，需配置 AMAZON_PAAPI_* 环境变量",
            metadata={"requires_approval": False},
        ),
        Tool(
            name="amazon_analyze_for_product_dev",
            func=lambda x: amazon_analyze_for_product_dev(x, "US"),
            description="抓取亚马逊商品并分析，输出可开发投入市场的产品建议。输入：类目/产品关键词（如 蓝牙耳机）。输出：市场概览、竞品分析、用户痛点、产品开发建议、投入产出评估",
            metadata={"requires_approval": False},
        ),
        Tool(
            name="web_search",
            func=web_search,
            description="基于互联网的 Web 搜索。在需要实时/最新信息、公开事实、外部文档或市场信息时调用。输入：搜索关键词或问句（字符串）。返回：标题、链接与摘要。可选配置 TAVILY_API_KEY 使用 Tavily，否则使用免费 DDGS。",
            metadata={"requires_approval": False},
        ),
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
