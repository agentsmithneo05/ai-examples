# load_test.py
from locust import HttpUser, task, between
import json


class OllamaLoadUser(HttpUser):
    # Wait between 1 to 5 seconds between "user" prompts
    wait_time = between(1, 5)

    @task
    def ask_llama(self):
        payload = {
            "model": "llama3.2",
            "prompt": "Explain the concept of quantum entanglement in one sentence.",
            "stream": False
        }

        # We use the raw /api/generate endpoint for pure performance testing
        with self.client.post("/api/generate", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code: {response.status_code}")

# To run this:
# 1. Open terminal and run: locust -f load_test.py --host http://localhost:11434
# 2. Open http://localhost:8089 in your browser to start the test.