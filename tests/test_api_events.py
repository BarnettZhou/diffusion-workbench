import asyncio
import unittest

from diffusion_workbench_api.events import EventHub
from test_api_jobs import ApiTestCase


class EventHubUnitTests(unittest.TestCase):
    def test_slow_subscriber_drops_oldest_event_without_blocking(self):
        hub = EventHub(loop=None)
        queue = hub.subscribe(max_events=1)

        hub._publish({"type": "queue_progress", "queued": 1, "running": None, "sequence": 1})
        hub._publish({"type": "queue_progress", "queued": 2, "running": None, "sequence": 2})

        self.assertEqual(queue.get_nowait()["sequence"], 2)
        self.assertTrue(queue.empty())

    def test_closed_hub_ignores_events(self):
        hub = EventHub(loop=None)
        hub.close()
        # 不应抛异常
        hub.emit_from_core_thread({"type": "queue_progress"})


class EventStreamTests(ApiTestCase):
    def test_multiple_subscribers_receive_public_events(self):
        with self.client.websocket_connect("/api/v1/events") as first:
            with self.client.websocket_connect("/api/v1/events") as second:
                self.core.sink(
                    {
                        "type": "job_finished",
                        "job_id": "job-1",
                        "status": "completed",
                        "output_path": str(self.core.root / "output" / "x.png"),
                        "steps": 8,
                    }
                )
                for ws in (first, second):
                    event = ws.receive_json()
                    self.assertEqual(event["type"], "job_finished")
                    self.assertEqual(event["job_id"], "job-1")
                    self.assertEqual(event["image_url"], "/api/v1/images/job-1")
                    self.assertIn("event_id", event)
                    self.assertNotIn("output_path", event)
                    self.assertNotIn(str(self.core.root), str(event))

    def test_step_progress_metrics_and_preview_are_forwarded(self):
        with self.client.websocket_connect("/api/v1/events") as ws:
            self.core.sink(
                {
                    "type": "step_progress",
                    "job_id": "job-1",
                    "step": 3,
                    "total": 8,
                    "elapsed_seconds": 6.0,
                    "step_seconds": 2.1,
                    "seconds_per_step": 2.0,
                    "steps_per_second": 0.5,
                    "eta_seconds": 10.0,
                }
            )
            event = ws.receive_json()
            self.assertEqual(event["seconds_per_step"], 2.0)
            self.assertEqual(event["eta_seconds"], 10.0)

            self.core.sink(
                {
                    "type": "preview_image",
                    "job_id": "job-1",
                    "step": 3,
                    "total": 8,
                    "mime_type": "image/jpeg",
                    "encoding": "base64",
                    "width": 72,
                    "height": 72,
                    "data": "/9j/4AAQ",
                }
            )
            preview = ws.receive_json()
            self.assertEqual(preview["type"], "preview_image")
            self.assertEqual(preview["data"], "/9j/4AAQ")
            self.assertEqual(preview["mime_type"], "image/jpeg")

    def test_error_event_only_exposes_first_line(self):
        with self.client.websocket_connect("/api/v1/events") as ws:
            self.core.sink(
                {
                    "type": "job_error",
                    "job_id": "job-1",
                    "error": "RuntimeError: boom\nTraceback (most recent call last):\n  ...",
                }
            )
            event = ws.receive_json()
            self.assertEqual(event["error"], "RuntimeError: boom")
            self.assertNotIn("Traceback", str(event))

    def test_event_ids_are_monotonic(self):
        with self.client.websocket_connect("/api/v1/events") as ws:
            for step in (1, 2):
                self.core.sink(
                    {"type": "step_progress", "job_id": "job-1", "step": step, "total": 2}
                )
            first_id = ws.receive_json()["event_id"]
            second_id = ws.receive_json()["event_id"]
            self.assertGreater(second_id, first_id)

    def test_emit_from_core_thread_is_threadsafe(self):
        # sink 从非 event loop 线程调用（unittest 主线程即模拟 Core 后台线程）
        with self.client.websocket_connect("/api/v1/events") as ws:
            self.core.sink({"type": "queue_progress", "queued": 0, "running": None, "sequence": 1})
            self.assertEqual(ws.receive_json()["type"], "queue_progress")


if __name__ == "__main__":
    unittest.main()
