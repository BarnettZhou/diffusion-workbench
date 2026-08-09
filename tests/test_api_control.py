import unittest

from test_api_jobs import ApiTestCase


class ControlStopTests(ApiTestCase):
    def test_stop_is_explicitly_global(self):
        response = self.client.post("/api/v1/control/stop")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["accepted"])
        self.assertEqual(body["scope"], "running-and-entire-queue")
        self.assertTrue(self.core.stopped)

    def test_stop_is_idempotent(self):
        self.client.post("/api/v1/control/stop")
        response = self.client.post("/api/v1/control/stop")
        self.assertEqual(response.status_code, 200)

    def test_no_single_job_delete_endpoint(self):
        response = self.client.delete("/api/v1/jobs/some-id")
        self.assertEqual(response.status_code, 405)


class ControlSkipTests(ApiTestCase):
    def test_skip_running_job(self):
        self.core.skipped_id = "job-1"
        response = self.client.post("/api/v1/control/skip")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["accepted"])
        self.assertEqual(body["skipped_job_id"], "job-1")
        self.assertEqual(body["scope"], "current-job-only")
        self.assertEqual(self.core.skip_calls, 1)

    def test_skip_when_idle_is_not_an_error(self):
        self.core.skipped_id = None
        response = self.client.post("/api/v1/control/skip")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["accepted"])
        self.assertIsNone(body["skipped_job_id"])
        self.assertEqual(body["reason"], "no-cancellable-job")

    def test_skip_worker_failure_returns_500(self):
        self.core.skipped_id = "job-1"
        self.core.skip_error = "worker 无响应"
        response = self.client.post("/api/v1/control/skip")
        self.assertEqual(response.status_code, 500)


if __name__ == "__main__":
    unittest.main()
