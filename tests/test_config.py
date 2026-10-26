from musictokenize.config import load_config


def test_config_loads():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert cfg["data"]["session_gap_minutes"] == 30
