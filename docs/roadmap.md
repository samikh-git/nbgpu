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

- **Run code in the running kernel.** Today an agent can only re-run the
  whole notebook. Running one cell, or evaluating an expression against live
  state ("what's the loss so far", "what shape is this tensor"), would let it
  inspect a model already loaded on the GPU without reloading it. For GPU
  work that is the difference between seconds and minutes per iteration.
  Builds on the proxy's existing connection to marimo.
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

- **Warm starts.** Snapshot a session's environment, data and model weights
  and resume from that instead of rebuilding. Modal has sandbox filesystem
  snapshots; whether they cover this case is unchecked.
- **More backends.** The backend contract is small by design. An SSH backend
  would cover any reachable GPU machine (a home box, a university cluster);
  RunPod or Lambda would add capacity and price competition.
- **Choosing the GPU by need.** `gpu = "auto"` with a `gpu-memory`
  requirement, picking the cheapest available option from the price table.
  More useful once there are several backends.

## Priorities

Set 2026-11-15:

1. **Run code in the running kernel.** Re-running the whole notebook is the
   main reason agent iteration on GPUs is slow.
2. **Choosing the GPU by need** (`gpu = "auto"` plus `gpu-memory`).

Budgets were suggested first, as what makes it safe to leave an agent alone
with a GPU; they are not in this round.
