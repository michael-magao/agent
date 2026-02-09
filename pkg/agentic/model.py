from langchain_openai import ChatOpenAI

api_key = "sk-aac1fe572f2c47609bf60dea4f428126"

llm = ChatOpenAI(
    model="deepseek-reasoner",  # 使用 DeepSeek 推理模型
    base_url="https://api.deepseek.com",  # DeepSeek API 地址
    api_key=api_key,  # DeepSeek API Key
    temperature=0  # Agent 建议使用高逻辑性模型
)

# 工具调用专用模型：deepseek-chat 无需 reasoning_content，避免 thinking 模式下多轮 tool call 的 400 报错
llm_tools = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=api_key,
    temperature=0,
)