import pytest

TEST_ADMIN_PASSWORD = "test-admin-password"


@pytest.fixture(autouse=True)
def isolate_brain_storage(tmp_path, monkeypatch):
    """Keep offline tests away from the user's MongoDB and personal memories."""
    import brain.api as api_module
    import brain.engine as engine_module
    from brain.storage import BrainStorage

    storage = BrainStorage("", "diya-test", tmp_path / "memory.json")
    monkeypatch.setattr(api_module, "get_storage", lambda _settings: storage)
    monkeypatch.setattr(engine_module, "get_storage", lambda _settings: storage)


@pytest.fixture(autouse=True)
def test_admin_credentials(monkeypatch):
    """Give the auth middleware a known password; never touch the real .env one.

    get_settings is lru_cached, so the cache must be cleared around the env
    change or an earlier test's settings (with the user's real password) leak.
    """
    from brain.config import get_settings

    monkeypatch.setenv("BRAIN_ADMIN_PASSWORD", TEST_ADMIN_PASSWORD)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
