# Sessions, idle and cost

A remote GPU bills by the second whether or not anyone is using it. Most of
nbgpu's lifecycle design exists so that a runtime is never billing without
you being able to see it, and rarely billing without you using it.

## Sessions

A **session** is one notebook running on one **runtime** (a Modal sandbox,
or a directory and some processes on the local backend).

For each session nbgpu keeps a small record in its state directory
(`~/.local/state/nbgpu` by default, or `$NBGPU_STATE_DIR`): the runtime's id,
the URL and the access token. The directory is readable only by you. The
provider remains the source of truth for what is actually running; every
runtime is tagged with its session id and with an id unique to your state
directory.

One notebook has at most one running session. `start` refuses a second.

## Session states

`nbgpu status` shows one of:

| State | Meaning | Next step |
|-------|---------|-----------|
| `running` | marimo is up | |
| `starting` | a start is in progress, or was interrupted before it finished | wait, or `nbgpu stop` to terminate |
| `idle-unsynced` | the idle timeout has passed, but there are browser edits your machine does not have yet | `nbgpu sync` to keep them |
| `idle-stopped` | the watchdog stopped marimo for idleness | `nbgpu stop` pulls files and terminates |
| `not-running` | the runtime is gone; only the record remains | `nbgpu stop` clears it |

## Three ways a session ends

1. **You stop it.** `nbgpu stop NOTEBOOK`: final pull, terminate, remove the
   record. The only path that is guaranteed to lose nothing.
2. **Idle shutdown.** After [`idle-timeout`](notebook-header.md#idle-timeout)
   without activity. Default 20 minutes.
3. **Max runtime.** The provider ends the runtime at
   [`max-runtime`](notebook-header.md#max-runtime) regardless of activity.
   Default 8 hours. There is no final pull.

Ctrl-C in the terminal running `nbgpu start` is *not* on this list. It stops
syncing and leaves the runtime up.

## Idle shutdown

Providers treat any running process as activity, and a session always has
marimo running, so a provider's own idle timeout would never fire. nbgpu
runs a small watchdog inside the runtime that decides from what you are
actually doing. The session is active while any of these holds:

- a file in the project was modified recently
- marimo reports a cell running or queued (this covers cells that sleep or
  wait on a network call without using CPU)
- processes on the runtime are using CPU
- the GPU is in use

An open browser tab is not activity. nbgpu's own sync traffic is not
activity either.

When the idle timeout passes:

1. If the runtime has edits your machine has not pulled, the watchdog waits
   for a sync, for up to one more idle timeout. `status` shows
   `idle-unsynced` and lists the paths.
2. It then stops marimo.
3. A running sync loop sees this, does a final pull, terminates the runtime
   and removes the record. If that pull fails, the runtime is kept and the
   stop retried, so edits are not traded for a few cents.
4. With no sync loop running, Modal ends the now-quiet sandbox itself within
   about five minutes. The record remains (`idle-stopped`, later
   `not-running`) until you run `nbgpu stop`.

So with a sync loop open, an idle session costs you `idle-timeout` of
billing and loses nothing. Detached, it costs about five minutes more, and
browser edits you never synced are lost when the sandbox goes.

If the watchdog itself dies, the sync loop notices its heartbeat has stopped
and restarts it.

## Untracked runtimes

A runtime can outlive its record: a state directory deleted by hand, a
crash at exactly the wrong moment. Such a runtime would bill until its max
runtime with nothing pointing at it.

`nbgpu status` asks the provider for every runtime tagged with your state
directory's id and lists those no record accounts for:

```text
untracked runtimes (terminate with `nbgpu stop --orphans`):
  sb-... on modal, up 0h47m: untracked (no session record), still billing
```

```bash
nbgpu stop --orphans
```

Runtimes created from another machine, by a teammate on the same Modal
workspace, or under a different `NBGPU_STATE_DIR` are listed separately and
never terminated by this command.

To check a backend that no current record uses, name it:

```bash
nbgpu status --backend modal
```

## Failed and interrupted starts

A start that fails after the runtime exists terminates it before reporting
the error. So does Ctrl-C during a start. If that terminate itself fails,
the record is kept in `starting` state so the runtime stays visible in
`status`.

## Cost

`nbgpu start` prints the hourly rate. `nbgpu status` prints the rate and the
cost so far (rate × uptime).

The rate is the sum of the GPU price × count, the CPU price × cores and the
memory price × GiB, from a table of Modal's list prices built into nbgpu
(`src/nbgpu/cost.py`, with the date it was last checked). CPU and memory use
Modal's sandbox rates.

These are estimates to keep you oriented, not invoices:

- CPU and memory are only counted when the header requests them; Modal's
  defaults are not priced.
- Modal bills the larger of what you request and what you use.
- Region multipliers are not included.
- The table is as current as its `checked` date.

A rate shown as `unknown` means the GPU or provider is not in the table.
Modal's dashboard has the authoritative numbers.

## Habits that keep the bill small

- Set `max-runtime` close to how long you expect to work, not the default.
- Keep `idle-timeout` short for interactive work; raise it only for
  notebooks with long quiet stretches that the activity signals above would
  miss.
- Use `nbgpu run` for jobs that do not need the UI. Its runtime exists only
  for the length of the run.
- Run `nbgpu status` before closing the laptop.
