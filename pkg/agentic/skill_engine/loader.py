import json
from pathlib import Path
from typing import List, Dict, Any


class SkillLoader:
    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = Path(skills_dir)
        self.metadata = self._load_metadata()
        self.loaded_skills = {}

    def _load_metadata(self) -> Dict:
        """加载技能元数据"""
        metadata_path = self.skills_dir / "skill_metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def parse_skill_md(self, content: str) -> Dict[str, Any]:
        """解析 Markdown 格式的技能文件"""
        skill_data = {
            "description": "",
            "capabilities": [],
            "input_requirements": "",
            "output_format": "",
            "sub_skills": [],
            "parameters": {}
        }

        lines = content.split('\n')
        current_section = None

        for line in lines:
            # 检测标题
            if line.startswith('## '):
                section = line[3:].strip().lower()
                if 'description' in section:
                    current_section = 'description'
                elif 'capabilit' in section:
                    current_section = 'capabilities'
                elif 'input' in section:
                    current_section = 'input'
                elif 'output' in section:
                    current_section = 'output'
                elif 'sub-skill' in section:
                    current_section = 'sub_skills'
                elif 'parameter' in section:
                    current_section = 'parameters'
                else:
                    current_section = None

            # 处理各个部分
            elif current_section == 'description':
                skill_data['description'] += line.strip() + ' '
            elif current_section == 'capabilities' and line.startswith('- '):
                skill_data['capabilities'].append(line[2:].strip())
            elif current_section == 'sub_skills' and line.startswith('- '):
                skill_name = line[2:].strip().split(':')[0]
                skill_data['sub_skills'].append(skill_name)
            elif current_section == 'parameters':
                if ':' in line:
                    key, value = line.split(':', 1)
                    skill_data['parameters'][key.strip()] = value.strip()

            # 清理描述
        skill_data['description'] = skill_data['description'].strip()

        return skill_data

    def load_skill(self, skill_name: str) -> Dict[str, Any]:
        """加载指定技能"""
        if skill_name in self.loaded_skills:
            return self.loaded_skills[skill_name]

        if skill_name not in self.metadata['skills']:
            raise ValueError(f"Skill '{skill_name}' not found")

        skill_meta = self.metadata['skills'][skill_name]
        skill_path = self.skills_dir / skill_meta['file']

        with open(skill_path, 'r', encoding='utf-8') as f:
            content = f.read()

        skill_data = self.parse_skill_md(content)
        skill_data.update({
            'name': skill_name,
            'file_path': skill_meta['file'],
            'triggers': skill_meta.get('triggers', []),
            'parent': skill_meta.get('parent'),
            'metadata': skill_meta
        })

        self.loaded_skills[skill_name] = skill_data
        return skill_data

    def load_sub_skills(self, parent_skill: str) -> List[Dict[str, Any]]:
        """加载子技能"""
        parent_data = self.load_skill(parent_skill)
        sub_skills = []

        for sub_name in parent_data.get('sub_skills', []):
            try:
                sub_skill = self.load_skill(sub_name)
                sub_skills.append(sub_skill)
            except ValueError:
                print(f"Warning: Sub-skill '{sub_name}' not found")

        return sub_skills