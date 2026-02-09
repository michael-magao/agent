import re
from typing import List, Tuple, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from pkg.agentic.skill_engine.loader import SkillLoader


class SkillRouter:
    def __init__(self, skill_loader: SkillLoader):
        self.skill_loader = skill_loader
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")

    def match_skill_by_keywords(self, user_input: str) -> List[Tuple[str, float]]:
        """通过关键词匹配技能"""
        matches = []

        for skill_name, skill_meta in self.skill_loader.metadata['skills'].items():
            if 'triggers' not in skill_meta:
                continue

            score = 0
            triggers = skill_meta['triggers']

            for trigger in triggers:
                if re.search(rf'\b{re.escape(trigger)}\b', user_input, re.IGNORECASE):
                    score += 1

            if score > 0:
                matches.append((skill_name, score))

        # 按分数排序
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    async def route_with_llm(self, user_input: str, context: Dict = None) -> Dict:
        """使用 LLM 进行智能路由"""
        available_skills = list(self.skill_loader.metadata['skills'].keys())

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个技能路由助手。根据用户输入选择合适的技能。
            
            可用技能:
            {skills_list}
            
            返回 JSON 格式:
            {{
                "primary_skill": "技能名称",
                "confidence": 0.0-1.0,
                "sub_skill": "子技能名称 或 null",
                "reasoning": "选择理由"
            }}"""),
            ("human", "用户输入: {user_input}")
        ])

        chain = prompt | self.llm

        response = await chain.ainvoke({
            "skills_list": "\n".join([f"- {skill}: {self.skill_loader.metadata['skills'][skill].get('description', '')}"
                                      for skill in available_skills]),
            "user_input": user_input,
            "context": context or {}
        })

        # 解析 LLM 响应
        return self._parse_llm_response(response.content)

    def _parse_llm_response(self, response: str) -> Dict:
        """解析 LLM 的 JSON 响应"""
        import json
        try:
            # 提取 JSON 部分
            json_str = re.search(r'\{.*\}', response, re.DOTALL)
            if json_str:
                return json.loads(json_str.group())
        except:
            pass

        return {"primary_skill": None, "confidence": 0, "reasoning": "解析失败"}