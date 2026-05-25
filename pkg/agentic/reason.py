from typing import Dict, Any

from pkg.agentic.memory.rag import setup_knowledge_base
from pkg.agentic.model import llm
from pkg.agentic.state import AgentState


def _message_content(message: object) -> str:
    if hasattr(message, "content"):
        return str(getattr(message, "content"))
    if isinstance(message, (tuple, list)) and len(message) >= 2:
        return str(message[1])
    return str(message)


def reason_node(state: AgentState) -> Dict[str, Any]:
    """推理节点：分析问题，明确目标"""
    messages = [
        ("system", "你是一个善于推理的AI助手。分析用户的问题，明确核心目标。"),
        ("human", _message_content(state["messages"][-1]) if state.get("messages") else state.get("current_goal", ""))
    ]

    response = llm.invoke(messages)

    return {
        "current_goal": response.content,
        "reflections": [f"推理完成：{response.content}"]
    }

def reason_with_knowledge(state: AgentState) -> Dict[str, Any]:
    """带知识检索的推理"""
    try:
        knowledge_base = setup_knowledge_base()
        docs = knowledge_base.similarity_search(state["current_goal"], k=1)
        knowledge = "\n".join([getattr(doc, "page_content", str(doc)) for doc in docs])
    except Exception as exc:
        knowledge = f"[知识库暂不可用: {exc}]"

    # todo 需要补充意图识别的逻辑（用户在多轮对话中，每一个新的输入，都需要结合该session的前期多次对话和记忆信息综合推演用户意图）
    prompt = f"""
        基于以下知识和历史进行推理：
        
        相关知识：
        {knowledge}
        
        历史对话：
        {[_message_content(message) for message in state.get('messages', [])[-5:]]}
        
        用户当前问题：
        {state['current_goal']}
        
        请分析核心问题和所需工具：
        """

    response = llm.invoke([("human", prompt)]) # todo 目前看上去借助llm进行目标的分析
    # print("reason_with_knowledge prompt:", prompt)
    # print("reason_with_knowledge info:", response.content)
    return {
        "current_goal": response.content
    }

def intent_recognition(text: str) -> str:
    """Recognize the intent from the given text.

    Args:
        text (str): The input text to analyze.

    Returns:
        str: The recognized intent.
    """
    # Placeholder implementation
    # todo 需要根据记忆里的内容进行更复杂的意图识别
    return text
