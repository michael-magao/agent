from __future__ import annotations

import io
import json
import os
import unittest
from unittest import mock
from urllib.error import HTTPError

from pkg.agentic.tools import knots_query


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


class KnotsQueryTest(unittest.TestCase):
    def test_falls_back_from_component_route_to_metadata_route(self) -> None:
        requests: list[str] = []

        def fake_urlopen(request: object, timeout: float) -> _FakeHTTPResponse:
            url = request.full_url  # type: ignore[attr-defined]
            requests.append(url)
            if len(requests) == 1:
                raise HTTPError(url, 404, "Not Found", {}, io.BytesIO(b'{"code": 5}'))
            return _FakeHTTPResponse({"code": 0, "result": {"name": "zk-main", "nodes": []}})

        with mock.patch.dict(
            os.environ, {"KNOTS_DC_ADMIN_BASE_URL": "http://knots.test"}, clear=False
        ):
            with mock.patch.object(knots_query, "urlopen", side_effect=fake_urlopen):
                result = json.loads(
                    knots_query.query_cluster_detail(
                        '{"cluster":"zk-main","service":"zookeeper"}'
                    )
                )

        self.assertEqual(
            requests,
            [
                "http://knots.test/v1/zookeeper/clusters/zk-main",
                "http://knots.test/v1/clusters/zk-main",
            ],
        )
        self.assertEqual(result["endpoint"], "/v1/clusters/zk-main")
        self.assertEqual(result["data"]["result"]["name"], "zk-main")

    def test_queries_v2_list_by_name_when_direct_cluster_lookup_misses(self) -> None:
        requests: list[str] = []

        def fake_urlopen(request: object, timeout: float) -> _FakeHTTPResponse:
            url = request.full_url  # type: ignore[attr-defined]
            requests.append(url)
            if len(requests) == 1:
                raise HTTPError(url, 404, "Not Found", {}, io.BytesIO(b"{}"))
            return _FakeHTTPResponse({"code": 0, "clusters": [{"id": "1", "name": "plain-name"}]})

        with mock.patch.dict(
            os.environ, {"KNOTS_DC_ADMIN_BASE_URL": "http://knots.test"}, clear=False
        ):
            with mock.patch.object(knots_query, "urlopen", side_effect=fake_urlopen):
                result = json.loads(knots_query.query_cluster_detail("plain-name"))

        self.assertEqual(
            requests,
            [
                "http://knots.test/v2/clusters/plain-name",
                "http://knots.test/v2/clusters?name=plain-name",
            ],
        )
        self.assertEqual(result["endpoint"], "/v2/clusters?name=plain-name")
        self.assertEqual(result["data"]["clusters"][0]["name"], "plain-name")

    def test_invalid_input_returns_parameter_error(self) -> None:
        self.assertIn("参数错误", knots_query.query_cluster_detail(""))


if __name__ == "__main__":
    unittest.main()
