from __future__ import annotations

import tempfile
import unittest

from pkg.agentic.checkpoint.file import FileCheckpointSaver
from pkg.agentic.tools.manager import calculate_expression


class FileCheckpointSaverTest(unittest.TestCase):
    def test_put_get_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            saver = FileCheckpointSaver(tmp_dir)
            config = {"configurable": {"thread_id": "thread_test"}}
            checkpoint = {"id": "ckpt-test", "channel_values": {"answer": 42}}

            saver.put(config, checkpoint, {"source": "unit-test"}, {})
            saved = saver.get_tuple(config)

            self.assertIsNotNone(saved)
            self.assertEqual(saved.checkpoint["id"], "ckpt-test")
            self.assertEqual(saved.checkpoint["channel_values"]["answer"], 42)
            self.assertEqual(saved.metadata["source"], "unit-test")


class CalculatorTest(unittest.TestCase):
    def test_calculates_allowed_expression(self) -> None:
        self.assertEqual(calculate_expression("sqrt(16) + 2 ** 3"), "12")

    def test_rejects_code_execution(self) -> None:
        result = calculate_expression("__import__('os').system('echo bad')")
        self.assertIn("计算失败", result)


if __name__ == "__main__":
    unittest.main()
