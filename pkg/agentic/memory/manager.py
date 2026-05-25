from __future__ import annotations

from pathlib import Path
from typing import Iterable

"""
记忆分为3层：
1. 长期记忆：RAG 外部数据、agent profile、Skill 文档、静态文件。
2. 短期记忆：多轮对话归档，可压缩后沉淀到长期记忆。
3. 即时记忆：本轮任务上下文，处理完成后可丢弃。
"""

DEFAULT_MEMORY_FILE = Path(__file__).resolve().parent / "long_term_mem.txt"


def _memory_path(path: str | Path | None = None) -> Path:
    return Path(path).expanduser() if path else DEFAULT_MEMORY_FILE


def _read_entries(path: Path) -> list[str]:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    return [entry.strip() for entry in content.split("\n\n") if entry.strip()]


def _write_entries(path: Path, entries: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [entry.strip() for entry in entries if entry and entry.strip()]
    path.write_text("\n\n".join(normalized) + ("\n" if normalized else ""), encoding="utf-8")


def append_memory(content: str, path: str | Path | None = None) -> None:
    target = _memory_path(path)
    entries = _read_entries(target)
    entries.append(content)
    _write_entries(target, entries)


def search_memory(query: str = "", limit: int = 10, path: str | Path | None = None) -> list[str]:
    entries = _read_entries(_memory_path(path))
    if not query:
        return entries[-limit:]
    keyword = query.lower()
    matched = [entry for entry in entries if keyword in entry.lower()]
    return matched[:limit]


def summarize_memory(max_chars: int = 2000, path: str | Path | None = None) -> str:
    entries = _read_entries(_memory_path(path))
    if not entries:
        return ""
    summary = "\n\n".join(entries[-20:])
    return summary[-max_chars:]


def compact_memory(max_chars: int = 8000, path: str | Path | None = None) -> str:
    target = _memory_path(path)
    entries = _read_entries(target)
    compacted = "\n\n".join(entries)
    if len(compacted) > max_chars:
        compacted = compacted[-max_chars:]
    _write_entries(target, [compacted] if compacted else [])
    return compacted


def delete_memory(confirm: bool = False, path: str | Path | None = None) -> bool:
    if not confirm:
        raise ValueError("删除长期记忆需要 confirm=True")
    target = _memory_path(path)
    if target.exists():
        target.unlink()
        return True
    return False
