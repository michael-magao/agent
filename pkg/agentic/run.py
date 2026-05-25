from __future__ import annotations

import argparse
from typing import Any, Dict, Optional


def _cli_approval(payload: dict) -> bool:
    """人工审核入口：敏感工具执行前会同步调用，在终端交互批准/拒绝。"""
    print("\n[人工审核] 工具请求:", payload.get("message", ""))
    print("  工具:", payload.get("tool"), "| 参数:", payload.get("args"))
    while True:
        ans = input("批准执行? (y/n): ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("请输入 y 或 n")


def run(
    query: str,
    user_name: str = "",
    *,
    approval_callback: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    max_iterations: int = 3,
) -> Dict[str, Any]:
    """运行 Agent，供 CLI、前端或测试代码复用。"""
    from pkg.agentic.agent import ReflectiveAgent

    agent = ReflectiveAgent(max_iterations=max_iterations)
    result = agent.run(
        query,
        config=config,
        approval_callback=approval_callback,
    )
    if result is None:
        return {
            "current_goal": query,
            "user_name": user_name,
            "tool_results": [],
            "reflections": [],
        }
    if user_name:
        result["user_name"] = user_name
    return result


def _print_result(result: Dict[str, Any]) -> None:
    print("=" * 50)
    print("最终结果：")
    print(f"目标：{result.get('current_goal', '')}")
    print(f"执行步骤数：{len(result.get('tool_results') or [])}")
    print(f"反思次数：{len(result.get('reflections') or [])}")
    tool_results = result.get("tool_results") or []
    print(f"最终答案：{tool_results[-1]['result'] if tool_results else '无'}")

    print("\n反思记录：")
    for i, reflection in enumerate(result.get("reflections") or [], 1):
        print(f"{i}. {(reflection or '')[:200]}...")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="运行 agentic ReflectiveAgent")
    parser.add_argument("query", nargs="?", help="要交给 Agent 处理的问题")
    parser.add_argument("--user-name", default="", help="用户名称，可选")
    parser.add_argument("--max-iterations", type=int, default=3, help="最大迭代次数")
    parser.add_argument("--no-approval-prompt", action="store_true", help="不启用 CLI 人工审核输入")
    args = parser.parse_args(argv)

    query = args.query or input("请输入任务: ").strip()
    result = run(
        query,
        user_name=args.user_name,
        approval_callback=None if args.no_approval_prompt else _cli_approval,
        max_iterations=args.max_iterations,
    )
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
