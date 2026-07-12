# Hermes Sibyl Memory

`hermes-sibyl-memory` is the standalone Sibyl memory provider for Hermes Agent. It keeps Hermes'
local memory files and session search intact while providing a governed, project-scoped memory
layer through Sibyl.

This repository currently contains the Hermes 0.18.2 plugin, configuration, and lifecycle
foundation. Transport, durable capture, recall, correction tools, and recovery land in the next
implementation tasks.

## Requirements

- Hermes Agent 0.18.2
- Python 3.11 through 3.13
- A Sibyl API key restricted to one project and one memory space
- A Sibyl server exposing the memory-provider capability profile

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
