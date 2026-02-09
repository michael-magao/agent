# 创建Agent
from pkg.agentic.agent import ReflectiveAgent

agent = ReflectiveAgent(model_name="gpt-4", max_iterations=3)  # 使用deepseek模型


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


# 运行复杂任务（传入 approval_callback 后，敏感工具会走人工审核，不再阻塞在 interrupt）
result = agent.run(
    """
集群zk-ai-platform-ego-common-us-live-jwhmjs8p-cc-backup出现大量CPU使用率飙升的告警，请处理：
""",
    approval_callback=_cli_approval,
)

# 防御 result 为 None（例如执行异常时）
if result is None:
    result = {
        "current_goal": "",
        "tool_results": [],
        "reflections": [],
    }

print("=" * 50)
print("最终结果：")
print(f"目标：{result.get('current_goal', '')}")
print(f"执行步骤数：{len(result.get('tool_results') or [])}")
print(f"反思次数：{len(result.get('reflections') or [])}")
tool_results = result.get("tool_results") or []
print(f"最终答案：{tool_results[-1]['result'] if tool_results else '无'}")

# 查看反思过程
print("\n反思记录：")
for i, reflection in enumerate(result.get("reflections") or [], 1):
    print(f"{i}. {(reflection or '')[:200]}...")