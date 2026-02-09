from typing import Optional, List, Dict, Any
from pydantic import BaseModel

# 定义技能元数据模型
class SkillMetadata(BaseModel):
    name: str
    description: str
    version: str = "1.0"
    author: Optional[str] = None
    tags: List[str] = []
    inputs: Dict[str, Any] = {}
    outputs: Dict[str, Any] = {}

class SkillEntity:
    id: str
    metadata: SkillMetadata
    bins: List[str] = []
    os: List[str] = []
    formula: str
    package: str
    module: str
    url: str
    archive: str
    targetDir: str
    tools: List[str] = []
    reference: List[str] = []