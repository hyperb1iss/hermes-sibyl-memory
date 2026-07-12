"""Provider configuration with native secret separation."""

from __future__ import annotations

import ipaddress
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

CONFIG_FILE = "sibyl.json"
API_KEY_ENV = "SIBYL_API_KEY"
DEFAULT_BASE_URL = "http://127.0.0.1:3334/api"


def _as_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{field} must be a boolean")


def _as_int(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return result


def _is_loopback(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validate_base_url(value: str, *, allow_insecure_http: bool) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base_url must be an absolute HTTP or HTTPS URL")
    if parsed.username or parsed.password:
        raise ValueError("base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query or fragment")
    if not parsed.path.rstrip("/").endswith("/api"):
        raise ValueError("base_url must end with /api")
    if parsed.scheme == "http" and not (_is_loopback(parsed.hostname) or allow_insecure_http):
        raise ValueError("non-loopback HTTP requires allow_insecure_http=true")
    return normalized


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    base_url: str = DEFAULT_BASE_URL
    project_id: str = ""
    memory_space_id: str = ""
    context_token_budget: int = 900
    context_limit: int = 8
    context_related_limit: int = 2
    automatic_capture: bool = True
    manual_tools: bool = True
    allow_insecure_http: bool = False

    @property
    def is_complete(self) -> bool:
        return bool(self.project_id and self.memory_space_id)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> ProviderConfig:
        allow_insecure_http = _as_bool(
            values.get("allow_insecure_http", False),
            field="allow_insecure_http",
        )
        base_url = _validate_base_url(
            str(values.get("base_url") or DEFAULT_BASE_URL),
            allow_insecure_http=allow_insecure_http,
        )
        project_id = str(values.get("project_id") or "").strip()
        memory_space_id = str(values.get("memory_space_id") or "").strip()
        if project_id and not project_id.startswith("project_"):
            raise ValueError("project_id must be a Sibyl project ID")
        if any(char.isspace() for char in memory_space_id):
            raise ValueError("memory_space_id must not contain whitespace")

        return cls(
            base_url=base_url,
            project_id=project_id,
            memory_space_id=memory_space_id,
            context_token_budget=_as_int(
                values.get("context_token_budget", 900),
                field="context_token_budget",
                minimum=1,
                maximum=32_000,
            ),
            context_limit=_as_int(
                values.get("context_limit", 8),
                field="context_limit",
                minimum=1,
                maximum=100,
            ),
            context_related_limit=_as_int(
                values.get("context_related_limit", 2),
                field="context_related_limit",
                minimum=0,
                maximum=20,
            ),
            automatic_capture=_as_bool(
                values.get("automatic_capture", True),
                field="automatic_capture",
            ),
            manual_tools=_as_bool(values.get("manual_tools", True), field="manual_tools"),
            allow_insecure_http=allow_insecure_http,
        )


def resolve_hermes_home(hermes_home: str | Path | None = None) -> Path:
    if hermes_home is not None:
        return Path(hermes_home).expanduser()
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except ImportError:
        return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()


def config_path(hermes_home: str | Path | None = None) -> Path:
    return resolve_hermes_home(hermes_home) / CONFIG_FILE


def load_provider_config(hermes_home: str | Path | None = None) -> ProviderConfig:
    path = config_path(hermes_home)
    if not path.exists():
        return ProviderConfig()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return ProviderConfig.from_mapping(data)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
        os.chmod(path, 0o600)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def save_provider_config(
    config: ProviderConfig,
    hermes_home: str | Path | None = None,
) -> None:
    payload = json.dumps(asdict(config), indent=2, sort_keys=True) + "\n"
    _atomic_write(config_path(hermes_home), payload)


def _decode_env_value(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] == '"':
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped[1:-1]
        return decoded if isinstance(decoded, str) else ""
    if len(stripped) >= 2 and stripped[0] == stripped[-1] == "'":
        return stripped[1:-1]
    return stripped


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = _decode_env_value(value)
    return values


def load_api_key(hermes_home: str | Path | None = None) -> str:
    environment_value = os.environ.get(API_KEY_ENV, "").strip()
    if environment_value:
        return environment_value
    env_path = resolve_hermes_home(hermes_home) / ".env"
    return _read_env_file(env_path).get(API_KEY_ENV, "").strip()


def save_api_key(api_key: str, hermes_home: str | Path | None = None) -> None:
    if not api_key or "\n" in api_key or "\r" in api_key:
        raise ValueError("SIBYL_API_KEY must be one non-empty line")

    env_path = resolve_hermes_home(hermes_home) / ".env"
    existing = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    replacement = f"{API_KEY_ENV}={json.dumps(api_key)}"
    output: list[str] = []
    replaced = False
    for line in existing:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key == API_KEY_ENV:
            if not replaced:
                output.append(replacement)
                replaced = True
            continue
        output.append(line)
    if not replaced:
        output.append(replacement)
    _atomic_write(env_path, "\n".join(output) + "\n")


def get_config_schema() -> list[dict[str, Any]]:
    return [
        {
            "key": "api_key",
            "description": "Sibyl API key",
            "secret": True,
            "required": True,
            "env_var": API_KEY_ENV,
        },
        {
            "key": "base_url",
            "description": "Sibyl API base URL",
            "required": True,
            "default": DEFAULT_BASE_URL,
        },
        {
            "key": "project_id",
            "description": "Sibyl project ID",
            "required": True,
        },
        {
            "key": "memory_space_id",
            "description": "Sibyl memory-space ID",
            "required": True,
        },
    ]


def _prompt(label: str, *, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"  {label}{suffix}: ").strip()
    return value or default


def run_setup(hermes_home: str, hermes_config: dict[str, Any]) -> None:
    from hermes_cli.config import save_config as save_hermes_config
    from hermes_cli.secret_prompt import masked_secret_prompt

    home = resolve_hermes_home(hermes_home)
    existing = load_provider_config(home)
    existing_key = load_api_key(home)

    print("\n  Sibyl stores completed user and final-assistant turns.")
    print("  Tool calls, tool results, and interrupted turns are excluded.\n")

    key_prompt = "  Sibyl API key"
    if existing_key:
        key_prompt += " (blank to keep current)"
    api_key = masked_secret_prompt(f"{key_prompt}: ").strip() or existing_key
    base_url = _prompt("Sibyl API base URL", default=existing.base_url)
    project_id = _prompt("Sibyl project ID", default=existing.project_id)
    memory_space_id = _prompt(
        "Sibyl memory-space ID",
        default=existing.memory_space_id,
    )

    if not api_key:
        raise ValueError("SIBYL_API_KEY is required")
    config = ProviderConfig.from_mapping(
        {
            **asdict(existing),
            "base_url": base_url,
            "project_id": project_id,
            "memory_space_id": memory_space_id,
        }
    )
    if not config.is_complete:
        raise ValueError("project_id and memory_space_id are required")

    save_provider_config(config, home)
    if api_key != existing_key:
        save_api_key(api_key, home)

    memory = hermes_config.setdefault("memory", {})
    if not isinstance(memory, dict):
        memory = {}
        hermes_config["memory"] = memory
    memory["provider"] = "sibyl"
    save_hermes_config(hermes_config)

    print("\n  Memory provider: sibyl")
    print("  Configuration saved; start a new session to activate.\n")
