import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from infer_server import create_app


class Engine:
    device = "cpu"
    class Model:
        def num_parameters(self): return 100
    model = Model()

    def encode(self, text):
        return text.split()

    def chat_prompt(self, messages, system=None):
        return "prompt exceeds limit"

    def stream(self, *args, **kwargs):
        yield "ok"

    def generate(self, *args, **kwargs):
        return "ok"


def test_prompt_guard_returns_413():
    client = TestClient(create_app(Engine(), max_prompt_tokens=1))
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}], "stream": False})
    assert response.status_code == 413


def test_health_endpoint():
    client = TestClient(create_app(Engine()))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
