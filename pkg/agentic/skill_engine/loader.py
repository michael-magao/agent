import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import yaml
except ImportError:
    yaml = None


class SkillLoader:
    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = Path(skills_dir)
        self.metadata = self._load_metadata()
        self.loaded_skills = {}

    def _load_metadata(self) -> Dict:
        """加载技能元数据；若文件不存在则返回空结构以支持目录发现"""
        metadata_path = self.skills_dir / "skill_metadata.json"
        if not metadata_path.exists():
            return {"skills": {}}
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_skills(self) -> List[str]:
        """列出可用技能名：优先用 metadata，否则扫描目录下含 Skill.md 的子目录"""
        if self.metadata.get("skills"):
            return list(self.metadata["skills"].keys())
        names = []
        if not self.skills_dir.exists():
            return names
        for p in self.skills_dir.iterdir():
            if p.is_dir() and (p / "Skill.md").exists():
                names.append(p.name)
        return sorted(names)

    def _parse_skill_md_from_dir(self, content: str, skill_name: str) -> Dict[str, Any]:
        """从目录下的 Skill.md 解析：支持 YAML front matter + 正文作为能力说明"""
        data = {
            "description": "",
            "capabilities": [],
            "sub_skills": [],
            "content_snippet": "",
        }
        if yaml and content.strip().startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 2:
                try:
                    meta = yaml.safe_load(parts[1])
                    if isinstance(meta, dict) and meta.get("description"):
                        data["description"] = meta["description"]
                except Exception:
                    pass
        # 取 ## 开头的段落作为能力/参考摘要（前 2000 字符）
        for block in re.split(r"\n##\s+", content):
            block = block.strip()
            if block and not block.startswith("---"):
                data["content_snippet"] += "\n## " + block[:800]
        data["content_snippet"] = (data["content_snippet"] or content)[:3000].strip()
        data["name"] = skill_name
        return data

    def _load_skill_from_dir(self, skill_name: str) -> Dict[str, Any]:
        """从 skills_dir/skill_name/Skill.md 加载（无 metadata 时使用）"""
        skill_path = self.skills_dir / skill_name / "Skill.md"
        if not skill_path.exists():
            raise ValueError(f"Skill '{skill_name}' not found")
        content = skill_path.read_text(encoding="utf-8")
        data = self._parse_skill_md_from_dir(content, skill_name)
        # 子技能：同目录下含 Skill.md 的子目录名
        parent_dir = self.skills_dir / skill_name
        if parent_dir.exists():
            for d in parent_dir.iterdir():
                if d.is_dir() and (d / "Skill.md").exists():
                    data["sub_skills"].append(d.name)
        self.loaded_skills[skill_name] = data
        return data

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
        """加载指定技能：优先用 metadata，否则从目录 Skill.md 加载"""
        if skill_name in self.loaded_skills:
            return self.loaded_skills[skill_name]

        if skill_name in self.metadata.get("skills", {}):
            skill_meta = self.metadata["skills"][skill_name]
            skill_path = self.skills_dir / skill_meta["file"]
            with open(skill_path, "r", encoding="utf-8") as f:
                content = f.read()
            skill_data = self.parse_skill_md(content)
            skill_data.update({
                "name": skill_name,
                "file_path": skill_meta["file"],
                "triggers": skill_meta.get("triggers", []),
                "parent": skill_meta.get("parent"),
                "metadata": skill_meta,
            })
            self.loaded_skills[skill_name] = skill_data
            return skill_data

        return self._load_skill_from_dir(skill_name)

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