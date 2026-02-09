import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple


def _to_json_safe(obj: Any) -> Any:
    """递归过滤，只保留可被 json.dump 序列化的值；其余转为 None 或字符串。"""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    # 其它类型（Runtime、CallbackManager 等）转为字符串，避免 TypeError
    try:
        return str(obj)
    except Exception:
        return None


class FileCheckpointSaver(BaseCheckpointSaver):
    """Checkpoint saver that saves checkpoints to a file.
    实现 LangGraph BaseCheckpointSaver 接口：get_tuple 返回单个 CheckpointTuple | None，
    put 接受 (config, checkpoint, metadata, new_versions) 并返回 RunnableConfig。
    """

    def __init__(self, base_dir: str = "./checkpoints", **kwargs: Any):
        super().__init__(**kwargs)
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Dict[str, Any],
        metadata: Dict[str, Any],
        new_versions: Dict[str, Any],
    ) -> RunnableConfig:
        """
        保存检查点。符合 LangGraph 接口：返回带 checkpoint_id 的 RunnableConfig。
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        thread_dir = self.base_dir / thread_id
        if checkpoint_ns:
            thread_dir = thread_dir / checkpoint_ns.replace(".", "_")
        thread_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_id = f"ckpt_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # 只持久化 configurable，并过滤掉 Runtime 等不可 JSON 序列化对象
        configurable = _to_json_safe(config.get("configurable") or {})
        # 使用 serde 序列化 checkpoint/metadata，支持 Message、复杂对象等
        typ_c, data_c = self.serde.dumps_typed(checkpoint)
        typ_m, data_m = self.serde.dumps_typed(metadata) # todo 干嘛用的？
        save_data = {
            "configurable": configurable,
            # "checkpoint_serde": {"type": typ_c, "data": base64.b64encode(data_c).decode()},
            # "metadata_serde": {"type": typ_m, "data": base64.b64encode(data_m).decode()},
            "checkpoint_serde": {"type": typ_c, "data": data_c},
            "metadata_serde": {"type": typ_m, "data": data_m},
            "new_versions": _to_json_safe(new_versions),
            "timestamp": datetime.now().isoformat(),
        }

        json_path = thread_dir / f"{checkpoint_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        latest_path = thread_dir / "LATEST.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(
                {"checkpoint_id": checkpoint_id, "timestamp": datetime.now().isoformat()},
                f,
                ensure_ascii=False,
                indent=2,
            )

        next_config: RunnableConfig = {
            "configurable": {
                **config.get("configurable", {}),
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
        return next_config

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """
        获取单个检查点元组（当前 config 指定或该 thread 下最新一个）。
        返回 None 表示未找到；否则返回一个 CheckpointTuple，供 LangGraph 使用 saved.checkpoint。
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        thread_dir = self.base_dir / thread_id
        if checkpoint_ns:
            thread_dir = thread_dir / checkpoint_ns.replace(".", "_")

        if not thread_dir.exists():
            return None

        checkpoint_id = config["configurable"].get("checkpoint_id")
        if checkpoint_id:
            file_path = thread_dir / f"{checkpoint_id}.json"
            if not file_path.exists():
                return None
        else:
            latest_path = thread_dir / "LATEST.json"
            if not latest_path.exists():
                return None
            with open(latest_path, "r", encoding="utf-8") as f:
                latest_info = json.load(f)
            checkpoint_id = latest_info["checkpoint_id"]
            file_path = thread_dir / f"{checkpoint_id}.json"

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 从 serde 格式反序列化
        cs = data.get("checkpoint_serde") or data.get("checkpoint")
        if isinstance(cs, dict) and "type" in cs and "data" in cs:
            checkpoint = self.serde.loads_typed(
                (cs["type"], base64.b64decode(cs["data"]))
            )
        else:
            checkpoint = cs if isinstance(cs, dict) else {}

        ms = data.get("metadata_serde") or data.get("metadata", {})
        if isinstance(ms, dict) and "type" in ms and "data" in ms:
            metadata = self.serde.loads_typed(
                (ms["type"], base64.b64decode(ms["data"]))
            )
        else:
            metadata = ms if isinstance(ms, dict) else {}

        parent_config = None
        if data.get("parent_configurable"):
            parent_config = {"configurable": data["parent_configurable"]}

        return CheckpointTuple(
            config={
                "configurable": {
                    **config.get("configurable", {}),
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=None,
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """
        列出该 thread（及可选的 checkpoint_ns）下的检查点，返回 CheckpointTuple 迭代器。
        """
        if config is None:
            return
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        thread_dir = self.base_dir / thread_id
        if checkpoint_ns:
            thread_dir = thread_dir / checkpoint_ns.replace(".", "_")
        if not thread_dir.exists():
            return

        files = sorted(thread_dir.glob("ckpt_*.json"), key=lambda p: p.stat().st_mtime)
        if before and before.get("configurable", {}).get("checkpoint_id"):
            before_id = before["configurable"]["checkpoint_id"]
            try:
                idx = next(i for i, p in enumerate(files) if p.stem == before_id)
                files = files[:idx]
            except StopIteration:
                pass
        if limit is not None:
            files = files[-limit:] if limit else []

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error loading checkpoint {file_path}: {e}")
                continue
            checkpoint_id = file_path.stem
            cs = data.get("checkpoint_serde") or data.get("checkpoint")
            if isinstance(cs, dict) and "type" in cs and "data" in cs:
                checkpoint = self.serde.loads_typed(
                    (cs["type"], base64.b64decode(cs["data"]))
                )
            else:
                checkpoint = cs if isinstance(cs, dict) else {}
            ms = data.get("metadata_serde") or data.get("metadata", {})
            if isinstance(ms, dict) and "type" in ms and "data" in ms:
                metadata = self.serde.loads_typed(
                    (ms["type"], base64.b64decode(ms["data"]))
                )
            else:
                metadata = ms if isinstance(ms, dict) else {}
            parent_config = (
                {"configurable": data["parent_configurable"]}
                if data.get("parent_configurable")
                else None
            )
            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=None,
            )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """存储子图/任务的中间 writes。当前文件实现不做持久化，仅保证接口存在。"""
        pass

    def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """异步封装：直接调用同步 get_tuple。"""
        return self.get_tuple(config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """异步列出检查点，委托给同步 list。"""
        for c in self.list(config, filter=filter, before=before, limit=limit):
            yield c
