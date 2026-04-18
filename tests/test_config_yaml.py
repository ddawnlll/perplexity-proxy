from __future__ import annotations

from app.config import Settings


def test_yaml_config_source_loads_and_flattens_nested_cache(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "port: 9999\n"
        "log-level: warning\n"
        "debug: true\n"
        "cache:\n"
        "  enabled: false\n"
        "  max-size: 12\n"
        "  ttl-seconds: 30\n"
    )

    monkeypatch.setenv("CONFIG_FILE", str(config_path))
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("DEBUG", raising=False)
    monkeypatch.delenv("CACHE_ENABLED", raising=False)
    monkeypatch.delenv("CACHE_MAX_SIZE", raising=False)
    monkeypatch.delenv("CACHE_TTL_SECONDS", raising=False)

    settings = Settings()

    assert settings.PORT == 9999
    assert settings.LOG_LEVEL == "warning"
    assert settings.DEBUG is True
    assert settings.CACHE_ENABLED is False
    assert settings.CACHE_MAX_SIZE == 12
    assert settings.CACHE_TTL_SECONDS == 30


def test_env_vars_override_yaml(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("port: 9999\nlog-level: warning\n")

    monkeypatch.setenv("CONFIG_FILE", str(config_path))
    monkeypatch.setenv("PORT", "1234")
    monkeypatch.setenv("LOG_LEVEL", "error")

    settings = Settings()

    assert settings.PORT == 1234
    assert settings.LOG_LEVEL == "error"
