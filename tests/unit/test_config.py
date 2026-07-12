from __future__ import annotations

import json
import stat

import pytest

from config import (
    API_KEY_ENV,
    ProviderConfig,
    load_api_key,
    load_provider_config,
    save_api_key,
    save_provider_config,
)


def test_config_and_secret_are_separate_and_private(tmp_path, monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    config = ProviderConfig.from_mapping(
        {
            "base_url": "https://sibyl.example/api",
            "project_id": "project_test",
            "memory_space_id": "space_test",
        }
    )

    save_provider_config(config, tmp_path)
    save_api_key("secret-value", tmp_path)

    config_file = tmp_path / "sibyl.json"
    env_file = tmp_path / ".env"
    payload = json.loads(config_file.read_text(encoding="utf-8"))
    assert API_KEY_ENV not in payload
    assert "secret-value" not in config_file.read_text(encoding="utf-8")
    assert stat.S_IMODE(config_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert load_provider_config(tmp_path) == config
    assert load_api_key(tmp_path) == "secret-value"


def test_non_loopback_http_requires_explicit_override():
    with pytest.raises(ValueError, match="non-loopback HTTP"):
        ProviderConfig.from_mapping(
            {
                "base_url": "http://sibyl.example/api",
                "project_id": "project_test",
                "memory_space_id": "space_test",
            }
        )


def test_loopback_http_is_allowed():
    config = ProviderConfig.from_mapping(
        {
            "base_url": "http://localhost:3334/api/",
            "project_id": "project_test",
            "memory_space_id": "space_test",
        }
    )

    assert config.base_url == "http://localhost:3334/api"
