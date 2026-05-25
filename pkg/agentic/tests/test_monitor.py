from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from pkg.agentic.tools import monitor


class _FakeHTTPResponse:
    def __init__(self, body: dict, status: int = 200) -> None:
        self.status = status
        self._body = json.dumps(body).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class MonitorToolTest(unittest.TestCase):
    def test_discovers_monitor_targets_by_cluster(self) -> None:
        urls: list[str] = []

        def fake_urlopen(request: object, timeout: float) -> _FakeHTTPResponse:
            url = request.full_url  # type: ignore[attr-defined]
            urls.append(url)
            if url.endswith("/tasks"):
                return _FakeHTTPResponse(
                    {
                        "tasks": [
                            {
                                "id": "task-1",
                                "task_name": "zk_exporter",
                                "display_name": "ZK exporter",
                            }
                        ]
                    }
                )
            return _FakeHTTPResponse(
                {
                    "targets": [
                        {
                            "address": "10.0.0.1:9100",
                            "target_unique": "node-1",
                            "disable_scrape": 0,
                            "meta": {
                                "cluster": "zk-main",
                                "mon_middleware_type": "zookeeper",
                            },
                        },
                        {
                            "address": "10.0.0.2:9100",
                            "meta": {
                                "cluster": "other",
                                "mon_middleware_type": "zookeeper",
                            },
                        },
                    ]
                }
            )

        with mock.patch.dict(
            os.environ,
            {
                "KNOTS_MONITOR_ACCESS_TOKEN": "token",
                "KNOTS_MONITOR_LIVE_BASE_URL": "http://monitor.test",
            },
            clear=False,
        ):
            with mock.patch.object(monitor, "urlopen", side_effect=fake_urlopen):
                result = json.loads(
                    monitor.query_monitor_detail(
                        '{"cluster":"zk-main","datasource":"middleware_consul",'
                        '"include_metrics":false}'
                    )
                )

        self.assertEqual(result["ips"], ["10.0.0.1"])
        self.assertEqual(result["targets"][0]["task_id"], "task-1")
        self.assertEqual(
            urls,
            [
                "http://monitor.test/nodeapi/v1/metricstores/middleware_consul/tasks",
                "http://monitor.test/nodeapi/v1/register/tasks/task-1/targets",
            ],
        )

    def test_queries_prometheus_with_ip_filter(self) -> None:
        queries: list[str] = []

        def fake_urlopen(request: object, timeout: float) -> _FakeHTTPResponse:
            body = request.data.decode("utf-8")  # type: ignore[attr-defined]
            queries.append(body)
            return _FakeHTTPResponse(
                {
                    "status": "success",
                    "data": {
                        "result": [
                            {
                                "metric": {"ip": "10.0.0.1"},
                                "value": [1710000000, "0.42"],
                            }
                        ]
                    },
                }
            )

        with mock.patch.dict(
            os.environ, {"KNOTS_MONITOR_PROM_SG_URL": "http://prom.test/query"}, clear=False
        ):
            with mock.patch.object(monitor, "urlopen", side_effect=fake_urlopen):
                result = json.loads(
                    monitor.query_monitor_detail(
                        '{"ips":["10.0.0.1"],"metrics":["cpu"],"regions":["sg"],'
                        '"timestamp":1710000000}'
                    )
                )

        self.assertEqual(result["ips"], ["10.0.0.1"])
        self.assertEqual(result["metrics"]["sg"]["cpu"][0]["peak"], 0.42)
        self.assertEqual(result["metrics"]["sg"]["cpu"][0]["avg"], 0.42)
        self.assertIn("max_over_time", queries[0])
        self.assertIn("avg_over_time", queries[1])
        self.assertIn("ip%3D~%2210%5C.0%5C.0%5C.1%22", queries[0])

    def test_invalid_input_returns_parameter_error(self) -> None:
        self.assertIn("参数错误", monitor.query_monitor_detail(""))

    def test_default_metrics_skip_when_ip_targets_missing(self) -> None:
        results, errors = monitor._query_metrics([], ["cpu"], ["sg"], "1d", 1710000000)

        self.assertEqual(results, {})
        self.assertEqual(errors[0].source, "metrics")
        self.assertIn("missing ip targets", errors[0].error)


if __name__ == "__main__":
    unittest.main()
