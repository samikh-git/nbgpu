# Roadmap

Ideas for where nbgpu goes next, written down 2026-11-15. nbgpu is a
personal tool for running marimo notebooks on remote GPUs, mostly driven by
coding agents, so the ideas are judged by one question: does it make agent
iteration on a GPU faster, safer or cheaper?

## Near term

Gaps in what exists today.

- **Run against Modal for real.** The Modal contract tests only run with
  `NBGPU_MODAL_TESTS=1`. Run them, plus a manual start → edit → stop on a T4,
  after any change to the Modal backend.
- ~~**Live installs for detached sessions.**~~ Done: `run` and `sync` install
  header changes first.
- ~~**Files back from `run`.**~~ Done: every run syncs back afterwards, and
  `--pull PATTERN` brings back other outputs.
- ~~**Actual spend.**~~ Done: `nbgpu cost`. Estimates now also use the
  account's live rates. Modal bills by App, so spend is for all sessions
  together.
- **Spend per session.** Give each session its own Modal App so the billing
  report can attribute cost to it. Needs `find` and orphan detection to look
  across apps.
- **Check that the skills trigger.** Run a handful of realistic prompts
  through the skill-creator eval and tune the descriptions.
- **A git remote.** Only needed to install on another machine or run CI.

## Agents work inside a live session

- ~~**Run code in the running kernel.**~~ Done: `nbgpu exec` evaluates code
  against the live kernel and runs single cells; see
  [`nbgpu exec`](cli.md#nbgpu-exec). Findings from the probe (2026-11-15,
  marimo 0.25.2) that shaped it:
  - marimo's `POST /api/kernel/execute` (header `Marimo-Stable-Session-Id`,
    ids from `GET /api/sessions`) runs code in the kernel's scratchpad;
    `marimo._code_mode` runs cells.
  - `marimo edit` has no kernel until a client connects; one connection
    creates it, and the session outlives that client. A client that stays
    connected makes a later browser read-only, so `exec` connects and leaves.
  - The kernel applies a reloaded file only between requests, so waiting for
    a pushed edit is a retry loop on the laptop (about a second locally).
  - Latency on Modal: about 0.3 s per call over the tunnel.
  - marimo's own value formatting (HTML, tagged JSON) is not agent-friendly,
    so the code nbgpu sends computes `repr()` itself.
  Still open: streaming output of long cells as they run (output arrives at
  the end, like `run`'s).
- **An MCP server.** Expose start, run, sync, status, stop and cost as MCP
  tools so agents call them directly instead of shelling out and parsing
  JSON. Guardrails ("ask before anything above $X/h") become part of the
  tool rather than a line in a skill.

## Hard spending limits

- **Budgets the runtime enforces.** Cost can be queried but nothing stops
  it. A `max-cost` header key and a daily cap across sessions, enforced by
  the watchdog already running inside each runtime, would make it safe to
  leave an agent running overnight. This is what makes unattended agent use
  possible.

## Experiments instead of single runs

- **Sweeps.** Run one notebook on N runtimes with different parameters and
  collect the outputs side by side. Agents are good at proposing a grid and
  bad at waiting for runs one at a time.
- **A run log.** Record each run's notebook hash, header, GPU, duration,
  cost and output files, giving an agent a history to compare against ("this
  change made it slower") without a separate experiment tracker.

## Faster and more flexible compute

- **Warm starts.** Done: the cache volume, filesystem snapshots, and the
  opt-in cell cache (`mo.persistent_cache` on the volume); see
  [Warm starts](warm-starts.md). Still open: resuming the kernel's memory
  itself. Modal's memory snapshots refuse GPU sandboxes and only restore
  the entrypoint's process tree (probed 2026-11-15); for CPU-only
  notebooks a "hibernate" built on them is possible.
- **More backends.** The backend contract is small by design. An SSH backend
  would cover any reachable GPU machine (a home box, a university cluster);
  RunPod or Lambda would add capacity and price competition.
- **Choosing the GPU by need.** `gpu = "auto"` with a `gpu-memory`
  requirement, picking the cheapest available option from the price table.
  More useful once there are several backends.

## Priorities

Set 2026-11-15:

1. ~~**Run code in the running kernel.**~~ Done (`nbgpu exec`).
2. **Choosing the GPU by need** (`gpu = "auto"` plus `gpu-memory`).

Budgets were suggested first, as what makes it safe to leave an agent alone
with a GPU; they are not in this round.
