from __future__ import annotations

import json
import os
import re
import socket
import time
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin
from urllib.request import Request, urlopen

DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_LIVE_MONITOR_URL = "https://monitoring.infra.sz.shopee.io"
DEFAULT_NONLIVE_MONITOR_URL = "https://monitoring.test.shopee.io"
DEFAULT_DC_ADMIN_URL = "http://127.0.0.1:8080"
DEFAULT_SG_PROM_URL = (
    "https://monitoring.infra.sz.shopee.io"
    "/vmselect/select/10083/prometheus/api/v1/query"
)
DEFAULT_US_PROM_URL = (
    "https://monitoring-us2.sz.shopee.io"
    "/vmselect/select/10095/prometheus/api/v1/query"
)

DATASOURCES = (
    "middleware-coordination-finance",
    "middleware-coordination-finance-us",
    "middleware_consul",
)
DATASOURCE_IDS = {
    "22": "middleware_consul",
    "703": "middleware-coordination-finance",
    "10562": "middleware-coordination-finance-us",
    "25": "nonlive_middleware_consul",
}
CPU_PROMQL = (
    'instance:node_cpu_utilisation:rate5m{bu="shopee",segment="General",'
    'platform="Distributed Coordination"}'
)
MEM_PROMQL = (
    '1 - node_memory_MemAvailable_bytes{bu="shopee",segment="General",'
    'platform="Distributed Coordination"} / '
    'node_memory_MemTotal_bytes{bu="shopee",segment="General",'
    'platform="Distributed Coordination"}'
)


@dataclass
class MonitorError:
    source: str
    error: str
    detail: str | None = None


def _timeout() -> float:
    raw = os.getenv("KNOTS_MONITOR_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def _monitor_base_url(env: str) -> str:
    env = env.lower()
    if env == "nonlive":
        return (
            os.getenv("KNOTS_MONITOR_NONLIVE_BASE_URL")
            or os.getenv("MONITOR_NONLIVE_BASE_URL")
            or DEFAULT_NONLIVE_MONITOR_URL
        ).rstrip("/") + "/"
    return (
        os.getenv("KNOTS_MONITOR_LIVE_BASE_URL")
        or os.getenv("KNOTS_MONITOR_BASE_URL")
        or os.getenv("MONITOR_BASE_URL")
        or DEFAULT_LIVE_MONITOR_URL
    ).rstrip("/") + "/"


def _dc_admin_base_url() -> str:
    return (
        os.getenv("KNOTS_DC_ADMIN_BASE_URL")
        or os.getenv("DC_ADMIN_BASE_URL")
        or os.getenv("KNOTS_BASE_URL")
        or DEFAULT_DC_ADMIN_URL
    ).rstrip("/") + "/"


def _prom_url(region: str) -> str:
    if region.lower() == "us":
        return os.getenv("KNOTS_MONITOR_PROM_US_URL") or DEFAULT_US_PROM_URL
    return os.getenv("KNOTS_MONITOR_PROM_SG_URL") or DEFAULT_SG_PROM_URL


def _http_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: Any | None = None,
) -> Any:
    data = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=req_headers, method=method)
    with urlopen(request, timeout=_timeout()) as response:
        raw = response.read().decode("utf-8", errors="replace")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw


def _http_form(url: str, data: dict[str, Any]) -> Any:
    encoded = urlencode({k: v for k, v in data.items() if v is not None}).encode("utf-8")
    request = Request(
        url,
        data=encoded,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urlopen(request, timeout=_timeout()) as response:
        raw = response.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw else {}


def _parse_tool_input(value: str) -> dict[str, Any]:
    text = (value or "").strip()
    if not text:
        raise ValueError("cluster/ip 不能为空")

    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 输入解析失败: {exc}") from exc
        cluster = data.get("cluster") or data.get("cluster_name") or data.get("name")
        ips = _as_list(data.get("ips") or data.get("ip") or data.get("targets"))
        metrics = _as_list(data.get("metrics") or data.get("metric") or ["cpu", "memory"])
        regions = _as_list(data.get("regions") or data.get("region") or ["sg", "us"])
        datasources = _as_list(data.get("datasources") or data.get("datasource"))
        if not datasources:
            datasources = list(DATASOURCES)
        return {
            "cluster": str(cluster).strip() if cluster else "",
            "ips": [_host_from_target(str(ip)) for ip in ips],
            "metrics": [str(metric).strip().lower() for metric in metrics if metric],
            "regions": [str(region).strip().lower() for region in regions if region],
            "datasources": [_normalize_datasource(str(ds)) for ds in datasources if ds],
            "task_id": str(data.get("task_id") or "").strip(),
            "env": str(data.get("env") or "live").strip().lower(),
            "over_range": str(data.get("range") or data.get("over_range") or "1d"),
            "timestamp": int(data.get("timestamp") or time.time()),
            "promql": str(data.get("promql") or "").strip(),
            "include_metrics": bool(data.get("include_metrics", True)),
        }

    return {
        "cluster": text,
        "ips": [],
        "metrics": ["cpu", "memory"],
        "regions": ["sg", "us"],
        "datasources": list(DATASOURCES),
        "task_id": "",
        "env": "live",
        "over_range": "1d",
        "timestamp": int(time.time()),
        "promql": "",
        "include_metrics": True,
    }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def _normalize_datasource(value: str) -> str:
    return DATASOURCE_IDS.get(value, value)


def _monitor_token(env: str) -> str | None:
    direct = (
        os.getenv("KNOTS_MONITOR_ACCESS_TOKEN")
        or os.getenv("MONITOR_ACCESS_TOKEN")
        or os.getenv("KNOTS_MONITOR_TOKEN")
    )
    if direct:
        return direct.removeprefix("Bearer ").strip()

    env_prefix = "NONLIVE" if env == "nonlive" else "LIVE"
    client_id = (
        os.getenv(f"KNOTS_MONITOR_{env_prefix}_CLIENT_ID")
        or os.getenv("KNOTS_MONITOR_CLIENT_ID")
        or os.getenv("MONITOR_CLIENT_ID")
    )
    client_secret = (
        os.getenv(f"KNOTS_MONITOR_{env_prefix}_CLIENT_SECRET")
        or os.getenv("KNOTS_MONITOR_CLIENT_SECRET")
        or os.getenv("MONITOR_CLIENT_SECRET")
    )
    if not client_id or not client_secret:
        return None

    url = urljoin(_monitor_base_url(env), "nodeapi/v1/open_api/token")
    payload = {"client_id": client_id, "client_secret": client_secret}
    data = _http_json(url, method="POST", body=payload)
    if isinstance(data, dict):
        token = data.get("access_token")
        if token:
            return str(token)
    raise RuntimeError("monitor token response missing access_token")


def _discover_monitor_targets(
    cluster: str,
    datasources: Iterable[str],
    env: str,
    task_id: str = "",
) -> tuple[list[dict[str, Any]], list[MonitorError]]:
    if not cluster and not task_id:
        return [], []

    errors: list[MonitorError] = []
    try:
        token = _monitor_token(env)
    except (HTTPError, URLError, TimeoutError, socket.timeout, RuntimeError) as exc:
        return [], [MonitorError(source="monitor-token", error=str(exc))]
    if not token:
        return [], [
            MonitorError(
                source="monitor-token",
                error="missing credentials",
                detail="设置 KNOTS_MONITOR_ACCESS_TOKEN，或 KNOTS_MONITOR_CLIENT_ID/SECRET。",
            )
        ]

    headers = {"Authorization": f"Bearer {token}"}
    targets: list[dict[str, Any]] = []
    task_ids = [task_id] if task_id else []
    task_meta: dict[str, dict[str, Any]] = {}

    if not task_ids:
        for datasource in datasources:
            url = urljoin(
                _monitor_base_url(env),
                f"nodeapi/v1/metricstores/{quote(datasource, safe='')}/tasks",
            )
            try:
                data = _http_json(url, headers=headers)
            except (HTTPError, URLError, TimeoutError, socket.timeout) as exc:
                errors.append(MonitorError(source=f"tasks:{datasource}", error=str(exc)))
                continue

            for task in _extract_list(data, "tasks"):
                task_name = str(task.get("task_name") or "")
                display = str(task.get("display_name") or "")
                if not _is_dc_task(task_name, display):
                    continue
                tid = str(task.get("id") or "")
                if tid:
                    task_ids.append(tid)
                    task_meta[tid] = {
                        "datasource": datasource,
                        "task_id": tid,
                        "task_name": task_name,
                        "display_name": display,
                    }

    for tid in _dedupe(task_ids):
        url = urljoin(
            _monitor_base_url(env),
            f"nodeapi/v1/register/tasks/{quote(tid, safe='')}/targets",
        )
        try:
            data = _http_json(url, headers=headers)
        except (HTTPError, URLError, TimeoutError, socket.timeout) as exc:
            errors.append(MonitorError(source=f"targets:{tid}", error=str(exc)))
            continue

        for target in _extract_list(data, "targets"):
            meta = target.get("meta") if isinstance(target, dict) else {}
            meta = meta if isinstance(meta, dict) else {}
            if cluster and meta.get("cluster") != cluster:
                continue
            middleware_type = str(meta.get("mon_middleware_type") or "").lower()
            if middleware_type and middleware_type not in {"etcd", "zookeeper"}:
                continue
            item = {
                "address": target.get("address"),
                "ip": _host_from_target(str(target.get("address") or "")),
                "target_unique": target.get("target_unique"),
                "disable_scrape": target.get("disable_scrape"),
                "meta": meta,
            }
            item.update(task_meta.get(tid, {"task_id": tid}))
            targets.append(item)

    return targets, errors


def _is_dc_task(task_name: str, display_name: str) -> bool:
    value = f"{task_name} {display_name}".lower()
    return any(keyword in value for keyword in ("etcd", "zk", "zookeeper"))


def _extract_list(data: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        result = data.get("result")
        if isinstance(result, dict) and isinstance(result.get(key), list):
            return [item for item in result[key] if isinstance(item, dict)]
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
    return []


def _query_dc_admin_nodes(cluster: str) -> tuple[list[str], list[MonitorError]]:
    if not cluster:
        return [], []
    encoded = quote(cluster, safe="")
    paths = [
        f"/v2/clusters?name={encoded}",
        f"/v2/concrete-clusters?name={encoded}",
        f"/v1/clusters/{encoded}",
        f"/v1/etcd/clusters/{encoded}",
        f"/v1/zookeeper/clusters/{encoded}",
    ]
    errors: list[MonitorError] = []
    for path in paths:
        try:
            body = _http_json(urljoin(_dc_admin_base_url(), path.lstrip("/")))
        except (HTTPError, URLError, TimeoutError, socket.timeout) as exc:
            errors.append(MonitorError(source=f"dc-admin:{path}", error=str(exc)))
            continue
        ips = _extract_ips(body)
        if ips:
            return ips, errors
    return [], errors


def _extract_ips(data: Any) -> list[str]:
    ips: list[str] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, child_key)
            return
        if isinstance(value, list):
            for item in value:
                visit(item, key)
            return
        if not isinstance(value, str):
            return
        if key not in {"ip", "server_ip", "serverIp", "address", "target"}:
            return
        host = _host_from_target(value)
        if _looks_like_ip(host):
            ips.append(host)

    visit(data)
    return _dedupe(ips)


def _query_metrics(
    ips: list[str],
    metrics: list[str],
    regions: list[str],
    over_range: str,
    timestamp: int,
    custom_promql: str = "",
) -> tuple[dict[str, Any], list[MonitorError]]:
    errors: list[MonitorError] = []
    results: dict[str, Any] = {}
    metric_promql = _metric_promqls(metrics, custom_promql)
    if not metric_promql:
        return results, errors
    if not ips and not custom_promql:
        return results, [
            MonitorError(
                source="metrics",
                error="missing ip targets",
                detail="未发现集群 target/ip，跳过默认 CPU/Memory 宽查询。",
            )
        ]

    for region in regions:
        region_result: dict[str, Any] = {}
        for metric_name, promql in metric_promql.items():
            filtered = _add_ip_filter(promql, ips) if ips else promql
            try:
                region_result[metric_name] = _query_peak_and_avg(
                    filtered,
                    over_range,
                    timestamp,
                    _prom_url(region),
                )
            except (HTTPError, URLError, TimeoutError, socket.timeout, RuntimeError) as exc:
                errors.append(MonitorError(source=f"metrics:{region}:{metric_name}", error=str(exc)))
        if region_result:
            results[region] = region_result
    return results, errors


def _metric_promqls(metrics: list[str], custom_promql: str) -> dict[str, str]:
    if custom_promql:
        return {"custom": custom_promql}
    result: dict[str, str] = {}
    for metric in metrics:
        if metric in {"cpu", "cpu_usage"}:
            result["cpu"] = CPU_PROMQL
        elif metric in {"mem", "memory", "memory_usage"}:
            result["memory"] = MEM_PROMQL
    return result


def _query_peak_and_avg(
    promql: str,
    over_range: str,
    timestamp: int,
    url: str,
) -> list[dict[str, Any]]:
    peak = _query_prometheus(f"max_over_time({promql}[{over_range}])", timestamp, url)
    avg = _query_prometheus(f"avg_over_time({promql}[{over_range}])", timestamp, url)
    return _merge_metric_results(peak, avg)


def _query_prometheus(promql: str, timestamp: int, url: str) -> list[dict[str, Any]]:
    response = _http_form(
        url,
        {
            "query": promql,
            "time": timestamp,
            "timeout": "10m",
            "step": 60,
        },
    )
    if not isinstance(response, dict) or response.get("status") != "success":
        raise RuntimeError(_compact_json(response))
    results = response.get("data", {}).get("result", [])
    parsed = []
    for item in results if isinstance(results, list) else []:
        metric = item.get("metric") or {}
        value = item.get("value") or []
        if not isinstance(metric, dict) or len(value) < 2:
            continue
        try:
            parsed.append({"labels": metric, "value": float(value[1])})
        except (TypeError, ValueError):
            continue
    return parsed


def _merge_metric_results(
    peak: list[dict[str, Any]],
    avg: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in peak:
        key = _metric_key(item)
        merged[key] = {"labels": item["labels"], "peak": item["value"], "avg": None}
    for item in avg:
        key = _metric_key(item)
        if key not in merged:
            merged[key] = {"labels": item["labels"], "peak": None, "avg": item["value"]}
        else:
            merged[key]["avg"] = item["value"]
    return list(merged.values())


def _metric_key(item: dict[str, Any]) -> str:
    labels = item.get("labels") or {}
    return str(labels.get("ip") or labels.get("instance") or sorted(labels.items()))


def _add_ip_filter(promql: str, ips: list[str]) -> str:
    if not ips:
        return promql
    matcher = '|'.join(re.escape(ip) for ip in _dedupe(ips))

    def repl(match: re.Match[str]) -> str:
        selector = match.group(1)
        if re.search(r"(^|,)\s*ip\s*(=|=~|!=|!~)", selector):
            return "{" + selector + "}"
        return "{" + selector + f',ip=~"{matcher}"' + "}"

    return re.sub(r"\{([^{}]*)\}", repl, promql)


def _host_from_target(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if text.startswith("[") and "]" in text:
        return text[1 : text.index("]")]
    if text.count(":") == 1:
        return text.split(":", 1)[0]
    return text


def _looks_like_ip(value: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", value))


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _grafana_links(cluster: str, targets: list[dict[str, Any]], env: str) -> list[str]:
    if not cluster:
        return []
    datasources = _dedupe(
        str(target.get("datasource") or target.get("meta", {}).get("datasource") or "")
        for target in targets
    )
    if not datasources:
        datasources = ["middleware_consul"]

    links = []
    for datasource in datasources:
        base = _monitor_base_url(env).rstrip("/")
        path = "/grafana/d/zk-streamline/zk-streamline"
        query = urlencode(
            {
                "from": "now-5m",
                "to": "now",
                "orgId": "74",
                "var-DS_PROMETHEUS": datasource,
                "var-cluster": cluster,
                "var-env": env,
            }
        )
        links.append(f"{base}{path}?{query}")
    return links


def _compact_json(data: Any, max_chars: int = 12000) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...<truncated>"
    return text


def query_monitor_detail(cluster: str) -> str:
    """查询 Knots 集群监控。

    输入可以是集群名，也可以是 JSON：
    {"cluster": "zk-xxx", "metrics": ["cpu", "memory"], "regions": ["sg", "us"]}
    """
    try:
        parsed = _parse_tool_input(cluster)
    except ValueError as exc:
        return f"参数错误: {exc}"

    cluster_name = parsed["cluster"]
    explicit_ips = _dedupe(parsed["ips"])
    targets, errors = _discover_monitor_targets(
        cluster_name,
        parsed["datasources"],
        parsed["env"],
        parsed["task_id"],
    )
    target_ips = _dedupe(target["ip"] for target in targets if target.get("ip"))
    dc_admin_ips: list[str] = []
    if not explicit_ips and not target_ips:
        dc_admin_ips, dc_errors = _query_dc_admin_nodes(cluster_name)
        errors.extend(dc_errors)

    ips = _dedupe([*explicit_ips, *target_ips, *dc_admin_ips])
    metric_results: dict[str, Any] = {}
    if parsed["include_metrics"]:
        metric_results, metric_errors = _query_metrics(
            ips,
            parsed["metrics"],
            parsed["regions"],
            parsed["over_range"],
            parsed["timestamp"],
            parsed["promql"],
        )
        errors.extend(metric_errors)

    return _compact_json(
        {
            "cluster": cluster_name,
            "source": "knots monitor",
            "env": parsed["env"],
            "targets": targets,
            "ips": ips,
            "metrics": metric_results,
            "grafana_links": _grafana_links(cluster_name, targets, parsed["env"]),
            "errors": [error.__dict__ for error in errors],
            "hint": (
                "监控注册查询需要 KNOTS_MONITOR_ACCESS_TOKEN，"
                "或 KNOTS_MONITOR_CLIENT_ID/KNOTS_MONITOR_CLIENT_SECRET。"
            ),
        }
    )
