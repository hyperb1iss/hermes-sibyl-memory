from __future__ import annotations

import json
import stat

import pytest

import config as config_module
from config import (
    API_KEY_ENV,
    ProviderConfig,
    load_api_key,
    load_provider_config,
    load_runtime_agent_id,
    save_api_key,
    save_provider_config,
    save_runtime_agent_id,
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


def test_runtime_identity_is_private_and_reusable_by_operator_commands(tmp_path):
    save_runtime_agent_id("hermes:home:nova", tmp_path)

    identity_file = tmp_path / "state" / "sibyl-runtime-identity.json"
    assert load_runtime_agent_id(tmp_path) == "hermes:home:nova"
    assert stat.S_IMODE(identity_file.stat().st_mode) == 0o600


def test_clean_setup_infers_the_exact_pinned_hermes_identity(monkeypatch):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "nova")

    assert config_module._active_runtime_agent_id() == "hermes:hermes:nova"


def test_setup_validation_rejects_a_different_canonical_agent_binding(
    monkeypatch,
    plugin_module,
):
    import importlib

    client_module = importlib.import_module(f"{plugin_module.__name__}.client")
    package_config = importlib.import_module(f"{plugin_module.__name__}.config")
    schemas = importlib.import_module(f"{plugin_module.__name__}.schemas")
    trust = importlib.import_module(f"{plugin_module.__name__}.trust")

    class Client:
        def __init__(self, **kwargs):
            pass

        def auth_me(self):
            return schemas.AuthMeResponse(
                credential=schemas.Credential(
                    type="api_key",
                    id="key-1",
                    scopes=("api:write",),
                    project_ids=("project_test",),
                    memory_space_ids=("space_test",),
                    agent_id="hermes:home:nova",
                    capability_profile="memory_provider",
                ),
                user={"id": "user-1"},
                organization={"id": "org-1"},
                org_role="member",
            )

        def close(self):
            pass

    monkeypatch.setattr(client_module, "SibylClient", Client)
    config = ProviderConfig.from_mapping(
        {
            "base_url": "https://sibyl.example/api",
            "project_id": "project_test",
            "memory_space_id": "space_test",
        }
    )

    with pytest.raises(trust.UnsafeCredentialError, match="does not match Hermes"):
        package_config._validate_setup_credential(
            config,
            "secret",
            agent_id="hermes:hermes:nova",
        )
