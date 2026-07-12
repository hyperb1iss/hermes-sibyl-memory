# Real-stack release gates

The release harness never provisions, starts, stops, or restarts services. Point it at an
explicitly provisioned disposable Hermes/Sibyl stack:

```text
HERMES_SIBYL_E2E=1
HERMES_SIBYL_E2E_HERMES_URL=https://hermes-e2e.example
HERMES_SIBYL_E2E_SIBYL_URL=https://sibyl-e2e.example/api
HERMES_SIBYL_E2E_PROFILE=release-e2e
HERMES_SIBYL_E2E_API_KEY=...
HERMES_SIBYL_E2E_HERMES_SESSION_TOKEN=...
HERMES_SIBYL_E2E_MUTATION_SCENARIOS=cross-session-recall,...
```

The two credentials have different trust boundaries. `HERMES_SIBYL_E2E_API_KEY` is the confined
Sibyl provider key. `HERMES_SIBYL_E2E_HERMES_SESSION_TOKEN` is the disposable Hermes dashboard
session token used for authenticated HTTP and JSON-RPC WebSocket probes. Neither credential is
serialized into a driver request or release receipt.

The repository-owned driver is invoked once per explicitly selected scenario as:

```text
uv run python scripts/real_stack_driver.py <scenario-slug>
```

It receives one JSON request on stdin. The request contains the two service URLs, profile, gate
mapping, and required check names. The driver uses only released surfaces: Hermes `/api/status`,
authenticated `/api/memory`, and authenticated `/api/ws`; plus Sibyl's authenticated REST API.
Redirects and oversized bodies fail closed.

The driver emits one JSON observation on stdout. Every required check is present and boolean.
Missing checks, missing scenarios, duplicate scenarios, an unpinned or inconsistently reported
Hermes runtime, or a driver failure produce a failed release receipt. Evidence is recursively
redacted before aggregation.

Hermes and Sibyl do not currently expose released controls for isolated key/profile provisioning,
network partitions, process crash windows, key revocation/rotation, or log snapshots. Scenarios
requiring those controls emit a machine-readable failed observation naming the exact missing hook.
They never call invented `/api/e2e` endpoints and cannot make the release eligible. The public API
lane currently executes capability confinement end to end; the other fourteen scenarios remain an
explicit release blocker until disposable-stack hooks are implemented and independently reviewed.

Every scenario requires its exact slug in `HERMES_SIBYL_E2E_MUTATION_SCENARIOS` or a matching
`--allow-mutation` argument. Capability confinement sends denied POST probes, so it also requires
consent even though a correctly confined key prevents domain-state changes. There is deliberately
no implicit "allow all" switch.
