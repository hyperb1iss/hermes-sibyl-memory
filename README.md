# Hermes Sibyl Memory

`hermes-sibyl-memory` is the standalone Sibyl memory provider for Hermes Agent. It keeps Hermes'
local memory files and session search intact while providing a governed, project-scoped memory
layer through Sibyl.

The provider recalls governed project context at turn start, captures only completed conversation
turns, and keeps mutations durable through a process-safe SQLite outbox. Hermes' local memory files
and session search remain available as the always-on local layer.

## Requirements

- Hermes Agent 0.18.2
- Python 3.11 through 3.13
- A Sibyl API key restricted to one project and one memory space
- A Sibyl server exposing the `memory_provider` capability profile

The API key must have `api:write`, exactly one project restriction, exactly one memory-space
restriction, and an agent binding matching `hermes:hermes:<active-profile>`. It does not need the
`mcp` scope unless Sibyl MCP is configured separately.

## Install

```bash
hermes plugins install hyperb1iss/hermes-sibyl-memory --no-enable
hermes memory setup sibyl
hermes memory status
```

Setup writes the API key to `$HERMES_HOME/.env` and non-secret settings to
`$HERMES_HOME/sibyl.json`. Both files are mode `0600`. The key is never written to `sibyl.json`.

Required settings:

- `SIBYL_API_KEY`
- `base_url`, ending in `/api`
- `project_id`
- `memory_space_id`

Plain HTTP is accepted only for loopback addresses by default. Non-loopback deployments must use
HTTPS unless `allow_insecure_http` is deliberately enabled in `sibyl.json`.

`agent_id` is derived from the active Hermes profile and is never prompted. After setup, start a
new Hermes session so the provider initializes with the saved configuration.

Advanced settings can be edited in `$HERMES_HOME/sibyl.json`:

```json
{
  "base_url": "https://sibyl.example.com/api",
  "project_id": "project_1138ef699fee",
  "memory_space_id": "space_home",
  "context_token_budget": 900,
  "context_limit": 8,
  "context_related_limit": 2,
  "automatic_capture": true,
  "manual_tools": true,
  "allow_insecure_http": false
}
```

## Operator commands

Hermes exposes these commands only while Sibyl is the active memory provider:

```bash
hermes sibyl status
hermes sibyl doctor
hermes sibyl flush
hermes sibyl config
```

`status` reports the plugin version, safe configuration, Hermes and authenticated agent identities,
credential scopes and restrictions, current context-read latency, the last successful outbox write,
and counts for every recoverable outbox state. It reports whether an API key exists but never prints
the key.

`doctor` validates the pinned Hermes compatibility floor, plugin installation, URL and TLS policy,
authentication, exact scopes, exact project and memory-space restrictions, agent binding,
`memory_provider` capability profile, context-pack read access, and local outbox writability.
Its context probe always uses `record_exposure=false`; the default command performs no remote
mutation.

An explicit write probe creates a diagnostic raw memory and immediately marks it stale, leaving an
auditable correction receipt in Sibyl:

```bash
hermes sibyl doctor --write-probe
```

`flush` validates the same bound credential, unblocks operations previously stopped by an auth
failure, and replays the durable queue through the production runtime dispatcher. It prints claimed,
outcome, and remaining per-state counts. The outbox lives at
`$HERMES_HOME/state/sibyl-outbox.sqlite3`; its database, write-ahead log, and shared-memory files are
mode `0600`.

`config` prints the configuration path and non-secret effective settings. The API key is rendered
only as `set` or `missing`.

## Recall and recovery

Automatic recall requests a bounded context pack without recording exposure. Only the exact items
returned to Hermes are acknowledged afterward. Slow or unavailable recall fails open, leaving the
manual recall path available without delaying the turn.

Completed-turn writes, delivered-context acknowledgments, and corrections enter the SQLite outbox
before network delivery. Operations keep stable idempotency keys, preserve per-session ordering,
and survive restarts, auth failures, interrupted server receipts, and revision reconciliation.

## Capture boundary

The production provider captures completed user text and final-assistant text. It excludes tool
calls, tool arguments, tool results, hidden reasoning, recalled context, and interrupted turns.
Shared deployments require consent from everyone whose conversation will be stored.

## Development

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest -q tests/unit
uv run --group contract pytest -q tests/contract
```

The contract group pins Hermes Agent 0.18.2 to commit
`9de9c25f620ff7f1ce0fd5457d596052d5159596`, the peeled commit behind tag `v2026.7.7.2`.

## License

MIT
