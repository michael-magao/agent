import os, yaml
import re
from sre_parse import State
from typing import Optional, Dict, Any, List

from langgraph.constants import END
from langgraph.graph import StateGraph

from pkg.agentic.skill_engine.metadata import SkillMetadata


class SkillDefinition:
    """完整的技能定义"""
    def __init__(self, path: str):
        self.path = path
        self.metadata: SkillMetadata = None
        self.instructions: str = ""
        self.script_path: Optional[str] = None
        self.graph = None
        self._load()

    def _load(self):
        """解析SKILL.md文件"""
        with open(os.path.join(self.path, "SKILL.md"), 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析YAML Front Matter (---之间的部分)
        parts = re.split(r'^---\s*$', content, maxsplit=2, flags=re.MULTILINE)
        if len(parts) >= 3:
            yaml_content = parts[1].strip()
            self.instructions = parts[2].strip()
            metadata_dict = yaml.safe_load(yaml_content)
        else:
            metadata_dict = {}
            self.instructions = content

        self.metadata = SkillMetadata(**metadata_dict)

        # 查找可能的执行脚本
        for file in os.listdir(self.path):
            if file.endswith('.py') and file != '__init__.py':
                self.script_path = os.path.join(self.path, file)
                break

    def build_graph(self, available_tools: Dict[str, Any]) -> StateGraph:
        """根据技能指令和可用工具构建LangGraph"""
        # 定义图的状态
        class SkillState(State):
            input: Dict[str, Any]
            step_outputs: Dict[str, Any] = {}
            current_step: str = ""
            final_output: Any = None

        builder = StateGraph(SkillState)

        # **核心：解析指令，构建节点**
        # 方案A：使用LLM解析复杂指令（推荐用于生产环境）
        steps = self._parse_instructions_with_llm(self.instructions)

        # 方案B：使用启发式规则解析（用于简单技能或测试）
        if not steps:
            steps = self._parse_instructions_heuristic(self.instructions)

        # 为每个步骤创建节点
        for i, step in enumerate(steps):
            node_name = step.get("id", f"step_{i}")

            # 创建节点执行函数
            def create_step_func(step_info):
                async def step_node(state: SkillState):
                    print(f"执行步骤: {step_info['action']}")

                    # 处理不同类型的步骤
                    action_type = step_info.get("type", "instruction")

                    if action_type == "tool_call" and "tool" in step_info:
                        # 调用工具
                        tool_name = step_info["tool"]
                        if tool_name in available_tools:
                            tool_input = self._prepare_tool_input(step_info, state)
                            result = await available_tools[tool_name].ainvoke(tool_input)
                            state.step_outputs[node_name] = result
                    elif action_type == "llm_call":
                        # 调用LLM处理
                        state.step_outputs[node_name] = await self._call_llm_for_step(step_info, state)
                    elif self.script_path:
                        # 执行技能自带脚本
                        result = await self._execute_script(state.input)
                        state.step_outputs[node_name] = result

                    state.current_step = node_name
                    return state

                return step_node

            builder.add_node(node_name, create_step_func(step))

        # 添加边（顺序执行或条件执行）
        for i in range(len(steps)-1):
            current_node = steps[i].get("id", f"step_{i}")
            next_node = steps[i+1].get("id", f"step_{i+1}")
            builder.add_edge(current_node, next_node)

        if steps:
            first_node = steps[0].get("id", "step_0")
            last_node = steps[-1].get("id", f"step_{len(steps)-1}")
            builder.add_edge("__start__", first_node)
            builder.add_edge(last_node, END)

        # 设置入口和出口节点
        builder.add_node("__start__", lambda state: state)
        builder.add_node("__end__", lambda state: state)

        self.graph = builder.compile()
        return self.graph

    # todo 使用大模型来初步解析出skill文档的内容
    def _parse_instructions_with_llm(self, instructions: str) -> List[Dict]:
        """使用LLM解析技能指令为结构化步骤（生产环境推荐）"""
        # 简化的LLM调用示例
        system_prompt = """你是一个工作流解析专家。请将以下技能指令解析为结构化的工作流步骤。
        每个步骤应包含：id, action, type (tool_call|llm_call|condition), tool(可选), condition(可选)。
        以JSON格式返回步骤列表。"""

        # 实际实现中，这里会调用OpenAI/Claude等API
        # 示例返回
        return [
            {"id": "parse_query", "action": "解析用户查询意图", "type": "llm_call"},
            {"id": "search_data", "action": "搜索相关数据", "type": "tool_call", "tool": "finance_db_query"},
            {"id": "analyze", "action": "分析数据趋势", "type": "llm_call"},
            {"id": "generate_report", "action": "生成报告", "type": "llm_call"}
        ]