# 1、获取告警信息

# 2、查询RAG获取文档

# 3、调用LLM简单生成分析报告

# 4、列出需要执行的动作
# 4.1 查询集群信息
# 4.2 查询监控信息
# 4.3 查询日志信息
# 4.4 制定执行计划

# 5、获取结果继续调用LLM生成分析报告

# 6、调用自动化执行引擎执行

# 7、反馈执行结果，继续调用LLM生成最终报告

# 8、发送通知相关人员

import os
from pathlib import Path
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from pkg.ai_ops.tools import search_sop_knowledge, get_cluster_info, get_system_metrics, run_ssh_command, get_cluster_metrics, get_log_summary

# 初始化LLM的api_key等参数
# 优先从环境变量读取，如果没有则提示用户设置
api_key = "sk-aac1fe572f2c47609bf60dea4f428126"

llm = ChatOpenAI(
    model="deepseek-reasoner",  # 使用 DeepSeek 推理模型
    base_url="https://api.deepseek.com",  # DeepSeek API 地址
    api_key=api_key,  # DeepSeek API Key
    temperature=0  # Agent 建议使用高逻辑性模型
)
tools = [
    search_sop_knowledge,
    get_cluster_info,
    get_cluster_metrics,
    get_system_metrics,
    get_log_summary,
    run_ssh_command
]

# 从本地文件加载 Prompt 模板
# 创建 ReAct 格式的 PromptTemplate
# ReAct agent 需要包含 {tools}, {tool_names}, {input}, {agent_scratchpad} 变量
# 将自定义的 prompt 内容作为系统指令，然后添加标准的 ReAct 格式
# 注意：{tools} 和 {tool_names} 会由 create_react_agent 自动填充
prompt_file = Path(__file__).parent / "prompt" / "system_prompt"
with open(prompt_file, "r", encoding="utf-8") as f:
    react_prompt_template = f.read()

prompt = PromptTemplate.from_template(react_prompt_template)

# 创建 Agent TODO 和create_tool_calling_agent的区别是什么？
agent = create_react_agent(llm, tools, prompt)

# 自定义错误处理函数，当解析失败时提供更友好的提示
def handle_parsing_error(error: Exception) -> str:
    """处理 LLM 输出解析错误"""
    return f"解析错误：LLM 的输出格式不符合 ReAct 格式要求。请确保按照以下格式输出：\nThought: [你的思考]\nAction: [工具名]\nAction Input: [工具参数]\nObservation: [工具返回结果]\n\n或者最终答案格式：\nFinal Answer: [你的答案]\n\n错误详情：{str(error)}"

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=handle_parsing_error  # 当解析失败时，将错误信息反馈给 agent 重试
)

# 执行任务
# input = """
# [active][error] 2025-12-31 14:40:27 UTC+08:00
# Name:[Middleware] [new metrics] zookeeper_global_sessions_increase_too_fast_error
# Deploy: live-backup-1
# Target: zk-content-intelligence-video-live
# target_host: 127.0.0.1
# alert_content:zk-content-intelligence-video-live zookeeper_global_sessions_increase_too_fast_error triggered, increase by 1085 /30s
# [View detail] https://i.shp.ee/pbxyasn
# """

input = "能否帮忙扫描下上面这连个zk集群直连的sdu？"

from langchain.callbacks.base import BaseCallbackHandler

class PlanCallbackHandler(BaseCallbackHandler):
    def on_agent_action(self, action, **kwargs):
        """当 Agent 决定执行某个动作（即完成了一次规划）时触发"""
        print(f"\n--- Planning 决策 ---")
        print(f"思考内容: {action.log}")
        print(f"选择工具: {action.tool}")
        print(f"输入参数: {action.tool_input}\n")

agent_executor.invoke(
    {"input": "现在出现下面的告警: " + input + "，请参考 SOP 帮我处理。"},
    {"callbacks": [PlanCallbackHandler()]}
)