import json
from pathlib import Path

def test_model_registry_format():
    registry = {
        "active_model": {
            "name": "test_model",
            "features": ["f1", "f2", "f3"]
        }
    }

    path = Path("model_registry.json")
    path.write_text(json.dumps(registry))

    loaded = json.loads(path.read_text())

    assert "active_model" in loaded
    assert isinstance(loaded["active_model"]["features"], list)

    path.unlink()