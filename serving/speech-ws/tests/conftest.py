"""
Test fixtures: the server with the model-free engines, so every protocol path
runs here without torch, a GPU or the Hub.

The env is set before `settings` is imported, which is why this lives in
conftest and sets os.environ at import.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TTS_ENGINE", "fake")
os.environ.setdefault("ASR_ENGINE", "fake")
os.environ.setdefault("TTS_LANGS", "fil,ceb")
os.environ.setdefault("ASR_LANGS", "fil,ceb")
os.environ.setdefault("SPEECH_API_KEY", "test-serving-key")
os.environ.setdefault("MAX_CONNS", "4")
os.environ.setdefault("QUEUE_WAIT_S", "2")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    import app as app_module

    with TestClient(app_module.app) as c:
        # the lifespan warms the fake engines on the pool; wait for ready
        import time
        for _ in range(200):
            if app_module._state["ready"]:
                break
            time.sleep(0.02)
        assert app_module._state["ready"], app_module._state
        yield c


@pytest.fixture
def auth():
    return {"Authorization": "Bearer test-serving-key"}
