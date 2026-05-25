import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def _api_key() -> str:
    for key in ("DEEPSEEK_API_KEY", "AGENT_LLM_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(key)
        if value:
            return value
    return "missing-api-key"


def _temperature() -> float:
    raw = os.getenv("AGENT_LLM_TEMPERATURE", "0")
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _chat_model(model_env: str, default_model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv(model_env, default_model),
        base_url=os.getenv("AGENT_LLM_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")),
        api_key=_api_key(),
        temperature=_temperature(),
    )


llm = _chat_model("AGENT_REASONING_MODEL", "deepseek-reasoner")

# 工具调用专用模型：deepseek-chat 无需 reasoning_content，避免 thinking 模式下多轮 tool call 的 400 报错
llm_tools = _chat_model("AGENT_TOOLS_MODEL", "deepseek-chat")
