from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

DEFAULT_TIMEOUT_SECONDS = 8
DEFAULT_BASE_URL = "http://127.0.0.1:8080"


@dataclass
class KnotsResponse:
    endpoint: str
    status: int
    body: Any


@dataclass
class KnotsRequestError:
    endpoint: str
    error: str
    status: int | None = None
    body: str | None = None


def _base_url() -> str:
    return (
        os.getenv("KNOTS_DC_ADMIN_BASE_URL")
        or os.getenv("DC_ADMIN_BASE_URL")
        or os.getenv("KNOTS_BASE_URL")
        or DEFAULT_BASE_URL
    ).rstrip("/") + "/"


def _timeout() -> float:
    raw = os.getenv("KNOTS_QUERY_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    token = (
        os.getenv("KNOTS_AUTH_TOKEN")
        or os.getenv("DC_ADMIN_AUTH_TOKEN")
        or os.getenv("KNOTS_TOKEN")
    )
    if token:
        headers["Authorization"] = token if token.lower().startswith("bearer ") else f"Bearer {token}"

    header_name = os.getenv("KNOTS_AUTH_HEADER")
    header_value = os.getenv("KNOTS_AUTH_HEADER_VALUE")
    if header_name and header_value:
        headers[header_name] = header_value
    return headers


def _parse_tool_input(value: str) -> dict[str, Any]:
    """支持纯集群名，也支持 JSON: {"cluster": "...", "service": "zookeeper|etcd|v2"}。"""
    text = (value or "").strip()
    if not text:
        raise ValueError("cluster 不能为空")

    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 输入解析失败: {exc}") from exc
        cluster = (
            data.get("cluster")
            or data.get("cluster_name")
            or data.get("name")
            or data.get("cluster_id")
        )
        if not cluster:
            raise ValueError("JSON 输入必须包含 cluster/cluster_name/name/cluster_id")
        return {
            "cluster": str(cluster).strip(),
            "service": (data.get("service") or data.get("component") or "").strip().lower(),
        }

    return {"cluster": text, "service": ""}


def _candidate_paths(cluster: str, service: str = "") -> list[str]:
    encoded = quote(cluster, safe="")
    service = service.lower()
    if service in {"zk", "zookeeper"}:
        return _unique_paths(
            [
                f"/v1/zookeeper/clusters/{encoded}",
                f"/v1/clusters/{encoded}",
                f"/v2/clusters/{encoded}",
                f"/v2/clusters?name={encoded}",
                f"/v2/concrete-clusters/{encoded}",
                f"/v2/concrete-clusters?name={encoded}",
            ]
        )
    if service == "etcd":
        return _unique_paths(
            [
                f"/v1/etcd/clusters/{encoded}",
                f"/v1/clusters/{encoded}",
                f"/v2/clusters/{encoded}",
                f"/v2/clusters?name={encoded}",
                f"/v2/concrete-clusters/{encoded}",
                f"/v2/concrete-clusters?name={encoded}",
            ]
        )
    if service in {"v2", "cluster"}:
        return _unique_paths(
            [
                f"/v2/clusters/{encoded}",
                f"/v2/clusters?name={encoded}",
                f"/v2/concrete-clusters/{encoded}",
                f"/v2/concrete-clusters?name={encoded}",
            ]
        )
    if service in {"concrete", "concrete_cluster", "concrete-cluster"}:
        return _unique_paths(
            [
                f"/v2/concrete-clusters/{encoded}",
                f"/v2/concrete-clusters?name={encoded}",
                f"/v2/clusters/{encoded}",
                f"/v2/clusters?name={encoded}",
            ]
        )
    if service in {"cmdb", "metadata"}:
        return _unique_paths(
            [f"/v1/clusters/{encoded}", f"/v1/clusters:withNodes?name={encoded}"]
        )

    lower = cluster.lower()
    if lower.startswith("zk-") or lower.startswith("zookeeper-") or "-zk-" in lower:
        return _unique_paths(
            [
                f"/v1/zookeeper/clusters/{encoded}",
                f"/v1/clusters/{encoded}",
                f"/v2/clusters/{encoded}",
                f"/v2/clusters?name={encoded}",
                f"/v2/concrete-clusters/{encoded}",
                f"/v2/concrete-clusters?name={encoded}",
            ]
        )
    if lower.startswith("etcd-") or "-etcd-" in lower:
        return _unique_paths(
            [
                f"/v1/etcd/clusters/{encoded}",
                f"/v1/clusters/{encoded}",
                f"/v2/clusters/{encoded}",
                f"/v2/clusters?name={encoded}",
                f"/v2/concrete-clusters/{encoded}",
                f"/v2/concrete-clusters?name={encoded}",
            ]
        )
    return _unique_paths(
        [
            f"/v2/clusters/{encoded}",
            f"/v2/clusters?name={encoded}",
            f"/v2/concrete-clusters/{encoded}",
            f"/v2/concrete-clusters?name={encoded}",
            f"/v1/clusters/{encoded}",
            f"/v1/etcd/clusters/{encoded}",
            f"/v1/zookeeper/clusters/{encoded}",
        ]
    )


def _unique_paths(paths: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _request_json(path: str) -> KnotsResponse:
    url = urljoin(_base_url(), path.lstrip("/"))
    request = Request(url, headers=_headers(), method="GET")
    with urlopen(request, timeout=_timeout()) as response:
        raw = response.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = raw
        return KnotsResponse(endpoint=path, status=response.status, body=body)


def _try_paths(paths: Iterable[str]) -> KnotsResponse | list[KnotsRequestError]:
    errors: list[KnotsRequestError] = []
    for path in paths:
        try:
            response = _request_json(path)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            errors.append(
                KnotsRequestError(endpoint=path, status=exc.code, error=exc.reason, body=body)
            )
            continue
        except (URLError, TimeoutError, socket.timeout) as exc:
            errors.append(KnotsRequestError(endpoint=path, error=str(exc)))
            continue

        if _is_success_payload(response.body):
            return response
        errors.append(
            KnotsRequestError(
                endpoint=path,
                status=response.status,
                error="non-success payload",
                body=_compact_json(response.body),
            )
        )
    return errors


def _is_success_payload(body: Any) -> bool:
    if not isinstance(body, dict):
        return bool(body)
    code = body.get("code")
    if code not in (None, 0, 200, "0", "200"):
        return False

    payload_keys = (
        "result",
        "cluster",
        "clusters",
        "concreteCluster",
        "concrete_cluster",
        "concreteClusters",
        "concrete_clusters",
        "data",
    )
    for key in payload_keys:
        value = body.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return len(value) > 0
        return True

    if code is None:
        detail_like_keys = {"id", "name", "nodes", "component", "mainCluster", "main_cluster"}
        return any(key in body for key in detail_like_keys)
    return False


def _compact_json(data: Any, max_chars: int = 8000) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...<truncated>"
    return text


def _format_success(cluster: str, response: KnotsResponse) -> str:
    return _compact_json(
        {
            "cluster": cluster,
            "source": "knots dc-admin",
            "endpoint": response.endpoint,
            "status": response.status,
            "data": response.body,
        }
    )


def _format_errors(cluster: str, errors: list[KnotsRequestError]) -> str:
    return _compact_json(
        {
            "cluster": cluster,
            "source": "knots dc-admin",
            "error": "failed to query cluster detail",
            "base_url": _base_url().rstrip("/"),
            "attempts": [error.__dict__ for error in errors],
            "hint": "请确认 KNOTS_DC_ADMIN_BASE_URL/DC_ADMIN_BASE_URL、认证头和 dc-admin 服务是否可用。",
        }
    )


def query_cluster_detail(cluster: str) -> str:
    """查询 Knots dc-admin 中的集群详情。

    输入可以是集群名，也可以是 JSON：
    {"cluster": "zk-xxx", "service": "zookeeper"}
    """
    try:
        parsed = _parse_tool_input(cluster)
    except ValueError as exc:
        return f"参数错误: {exc}"

    cluster_name = parsed["cluster"]
    result = _try_paths(_candidate_paths(cluster_name, parsed.get("service", "")))
    if isinstance(result, KnotsResponse):
        return _format_success(cluster_name, result)
    return _format_errors(cluster_name, result)
