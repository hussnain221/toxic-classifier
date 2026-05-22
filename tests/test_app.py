import unittest

import app as app_module


class ToxicClassifierApiTest(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()

    def test_health_reports_model_state(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("model_loaded", data)
        self.assertIn("status", data)

    def test_classify_rejects_empty_text(self):
        response = self.client.post("/api/classify", json={"text": "   "})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_classify_returns_all_labels(self):
        response = self.client.post(
            "/api/classify",
            json={"text": "Thanks for your thoughtful and helpful comment."},
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(set(data["predictions"]), set(app_module.LABELS))
        self.assertIsInstance(data["is_toxic"], bool)
        self.assertIn(data["top_label"], app_module.LABELS)

    def test_classify_returns_503_when_model_unavailable(self):
        old_model = app_module.model
        old_error = app_module.model_error
        app_module.model = None
        app_module.model_error = "test unavailable"

        try:
            response = self.client.post("/api/classify", json={"text": "hello"})
        finally:
            app_module.model = old_model
            app_module.model_error = old_error

        self.assertEqual(response.status_code, 503)
        self.assertIn("Model is unavailable", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
