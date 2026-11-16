"""Command-line interface: a thin layer over nbgpu.lifecycle.

Commands (every one accepts --json for machine-readable output and exits
non-zero on failure with a one-line error on stderr):

  nbgpu start NOTEBOOK [--backend NAME] [--gpu G] [--cpu N] [--memory M]
                       [--idle-timeout D] [--max-runtime D] [--project-root DIR]
                       [--detach]
      Start a session and print the URL (with token) and hourly rate.
      Without --detach, keep running a sync loop (poll every second) until
      Ctrl-C, then do a final pull and leave the runtime running. If the
      runtime's idle watchdog stops marimo, the loop pulls, terminates the
      runtime and removes the record (if that pull fails, the runtime and
      record are kept and the stop is retried). The loop also restarts the
      watchdog if it died (its heartbeat file stops changing).
  nbgpu open NOTEBOOK
      Print the URL of the notebook's running session and resume the sync loop.
  nbgpu sync NOTEBOOK
      One sync pass; prints what moved and any conflicts. Then installs
      header dependency changes into the runtime (EnvWatcher.apply), so
      detached sessions get them too.
  nbgpu status [--backend NAME]
      Every session: notebook, backend, GPU, state, uptime, cost so far,
      conflicts; plus runtimes this state directory created that no record
      tracks ("untracked, still billing"), and nbgpu runtimes of other owners
      (other laptops, teammates, other NBGPU_STATE_DIRs), which are only listed.
      For a session whose runtime is gone: a warning that browser edits after
      the last successful sync (its time is shown) are lost.
  nbgpu run NOTEBOOK [--backend NAME] [--max-lines N] [--pull PATTERN]...
                     [--project-root DIR]
      Run the notebook as a script remotely; prints its output; exits with
      the notebook's exit code. In a session, header changes are installed
      first. Afterwards one sync pass brings back what `sync` would, and
      --pull brings back other matching outputs (lifecycle.pull_matching).
  nbgpu exec NOTEBOOK (CODE | -f FILE | --cell NAME... | --all | --list)
                      [--sync | --no-sync] [--max-lines N]
      Run code, or cells, in the live marimo kernel of the notebook's session
      (lifecycle.exec_in_kernel, nbgpu.kernel). Prints the output as it
      arrives (bounded like run's), then the repr of the code's last
      expression; exits 1 if it failed. --cell and --all sync first by
      default; CODE does not.
  nbgpu cost [--days N] [--backend NAME]
      What the provider actually billed for nbgpu's runtimes, per day and per
      resource; also refreshes the cached live prices.
  nbgpu stop NOTEBOOK [--force] [--snapshot | --no-snapshot]
      Final pull, terminate, remove the record. --force: stop despite open
      conflicts, and terminate even if the final pull fails. --snapshot
      (default: the header's `snapshot`) saves the runtime's filesystem first;
      the next `start` boots from it if the environment still matches
      (`start --fresh` builds instead).
  nbgpu stop --orphans [--backend NAME] [--yes]
      List untracked runtimes of this state directory and terminate them
      (after a confirmation unless --yes; --json requires --yes). Runtimes of
      other owners are never touched.

Backend choice: --backend, else $NBGPU_BACKEND, else the spec's provider.
State directory: $NBGPU_STATE_DIR (see session.default_state_dir).
`status` and `stop --orphans` only query backends used by records, named by
$NBGPU_BACKEND or --backend (`stop --orphans --backend NAME`: only NAME), so
they never contact a provider that is not in use.

Errors: one line on stderr, no traceback, non-zero exit.

--json shapes:
  start:  {"session_id", "notebook", "backend", "runtime_id",
           "url" (includes access_token), "local_url" (when the proxy runs),
           "rate": null | {"usd_per_hour", "complete", "prices_checked", "prices_source"},
           "from_snapshot": null | image id, "timings": {"create", "upload", "ready", "total"}}
  status: {"sessions": [{"session_id", "notebook", "backend", "gpu", "running",
           "state", "uptime_s", "cost_usd", "prices_checked" (ISO date) and
           "prices_source" ("live", "list" or "file"; both null when cost_usd
           is), "conflicts", "last_sync_at", "warning", "unsynced"}],
           "orphans": [{"backend", "runtime_id", "uptime_s", "session_tag"}],
           "other_owners": [same shape]}
  Rates are estimates from cost.current_prices() at start ($NBGPU_PRICES, else
  the account's live rates cached by start/cost, else prices.toml); the
  source and date are recorded with the rate and shown beside it. status
  never reads prices, so a broken $NBGPU_PRICES cannot hide what is billing.
  `cost` reports what the provider actually billed (CAP_BILLING).
  sync, stop: {"pushed", "pulled", "merged", "conflicts", "skipped", "warnings"}
           (stop also "snapshot": null | {"image", "created_at", "seconds", ...})
           (sync also "env": header changes installed or reported)
  stop --orphans: {"orphans": [...], "terminated": [ids], "skipped": [ids],
           "failed": {id: error}}
  cost:   {"backend", "start", "end", "days": [{"date", "usd"}], "by_resource",
           "total_usd", "prices_refreshed", "prices_error"}
  run:    {"exit_code", "output", "truncated", "full_output_path",
           "pulled" (files brought back after the run), "warnings", "env"}
  exec:   {"ok", "value", "error", "output", "truncated", "full_output_path",
           "cells", "kernel_created", "ran_notebook", "pushed", "warnings", "env"}
           (exit code 0 when ok, else 1, with the document still printed)

Decisions beyond the contract:
- The chosen backend's name is passed as the spec's `provider` override, so
  prices are looked up for the provider actually used.
- `start` refuses when the notebook already has a running session.
- `run` uses the notebook's running session when it has exactly one
  (and --backend, if given, names its backend); otherwise a one-off runtime.
- `open --json` prints the `start` shape. In --json mode the sync loop's
  progress goes to stderr so stdout stays one JSON document.
- During the sync loop, transient sync errors are reported and retried; the
  loop ends with an error if the runtime is gone. One SyncEngine is kept
  for the whole loop.
- SIGTERM and SIGHUP are handled like Ctrl-C (exit code 128 + signal). On
  shutdown the final pull comes before any output, and output errors (a
  hung-up terminal) are ignored.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

import click

from nbgpu import cost, lifecycle, watchdog
from nbgpu.backends import get_backend
from nbgpu.backends.base import Backend, RuntimeHandle
from nbgpu.cost import Rate, format_usd
from nbgpu.envsync import EnvWatcher
from nbgpu.header import update_dependencies
from nbgpu.proxy import LocalProxy
from nbgpu.session import TAG_SESSION, Session, SessionStore
from nbgpu.spec import load_spec
from nbgpu.sync import SyncEngine, SyncReport

SYNC_INTERVAL_S = 1.0
_INTERRUPTED_EXIT = 130
# The watchdog counts as dead when its heartbeat has not changed for
# 3 check intervals plus this many seconds (laptop clock).
WATCHDOG_STALE_SLACK_S = 15.0
WATCHDOG_RESTART_BACKOFF_S = 60.0
IDLE_STOP_RETRY_S = 10.0


class CliError(Exception):
    """A user-facing error; the message is printed as one line."""


# Colors. click.echo strips them when the stream is not a terminal; NO_COLOR
# (https://no-color.org) turns them off everywhere.
_STYLES: dict[str, dict[str, Any]] = {
    "ok": {"fg": "green"},
    "warn": {"fg": "yellow"},
    "error": {"fg": "red", "bold": True},
    "url": {"fg": "cyan", "underline": True},
    "money": {"fg": "yellow", "bold": True},
    "label": {"bold": True},
    "dim": {"dim": True},
    "cmd": {"fg": "magenta"},
    "id": {"fg": "blue"},
}


def _c(text: object, role: str) -> str:
    if os.environ.get("NO_COLOR"):
        return str(text)
    return click.style(str(text), **_STYLES[role])


def _one_line(message: str) -> str:
    return " | ".join(part.strip() for part in message.splitlines() if part.strip()) or "error"


def _say(message: str, *, err: bool = False) -> None:
    """click.echo that never raises: the terminal may be gone (SIGHUP, EIO)."""
    try:
        click.echo(message, err=err)
    except (OSError, ValueError):
        pass


def _fail(message: str, code: int = 1) -> NoReturn:
    _say(f"{_c('nbgpu: error:', 'error')} {_one_line(message)}", err=True)
    sys.exit(code)


class _Group(click.Group):
    """Turns every non-click exception into a one-line error and exit 1."""

    def invoke(self, ctx: click.Context) -> Any:
        try:
            with lifecycle.signals_as_interrupt():
                return super().invoke(ctx)
        except (click.exceptions.Exit, click.exceptions.Abort, click.ClickException):
            raise
        except lifecycle.Terminated as exc:
            _fail(f"interrupted ({exc})", 128 + exc.signum)
        except KeyboardInterrupt:
            _fail("interrupted", _INTERRUPTED_EXIT)
        except Exception as exc:  # noqa: BLE001 - the CLI's error boundary
            _fail(str(exc) or type(exc).__name__)


# ---------------------------------------------------------------------------
# helpers


def _emit_json(data: object) -> None:
    click.echo(json.dumps(data, indent=2))


def _report_json(report: SyncReport) -> dict[str, list[str]]:
    return {
        "pushed": report.pushed,
        "pulled": report.pulled,
        "merged": report.merged,
        "conflicts": report.conflicts,
        "skipped": report.skipped,
        "warnings": report.warnings,
    }


def _rate_json(rate: Rate | None) -> dict[str, object] | None:
    if rate is None:
        return None
    return {"usd_per_hour": rate.usd_per_hour, "complete": rate.complete,
            "prices_checked": rate.prices_checked or None, "prices_source": rate.prices_source or None}


def _rate_text(rate: Rate | None) -> str:
    if rate is None or not rate.complete:
        return _c("unknown", "warn")
    return _c(f"{format_usd(rate.usd_per_hour)}/h", "money")


def _session_json(session: Session, proxy: LocalProxy | None = None) -> dict[str, object]:
    return {
        "session_id": session.session_id,
        "notebook": str(session.notebook),
        "backend": session.backend,
        "runtime_id": session.runtime_id,
        "url": session.url_with_token(),
        "local_url": proxy.url_with_token(session.token) if proxy is not None else None,
        "rate": _rate_json(lifecycle.session_rate(session)),
        "from_snapshot": lifecycle.session_snapshot(session),
        "timings": lifecycle.session_timings(session) or None,
    }


def _link(url: str) -> str:
    """The URL as a clickable OSC 8 hyperlink when stdout is a terminal (the
    visible text stays the plain URL, so terminals that detect URLs still do)."""
    styled = _c(url, "url")
    if not sys.stdout.isatty() or os.environ.get("NBGPU_NO_HYPERLINKS"):
        return styled
    return f"\x1b]8;;{url}\x1b\\{styled}\x1b]8;;\x1b\\"


def _start_proxy(session: Session, *, quiet: bool, packages: queue.Queue | None = None) -> LocalProxy | None:
    """A localhost proxy serving marimo's frontend from this machine; None
    (with a warning) if it cannot start, in which case the remote URL works.
    Package installs/removals made in marimo's UI are put on `packages`."""
    if os.environ.get("NBGPU_NO_PROXY"):
        return None
    on_packages = (lambda added, removed: packages.put((added, removed))) if packages is not None else None
    try:
        proxy = LocalProxy(session.url, on_packages=on_packages)
        proxy.start()
    except Exception as exc:  # noqa: BLE001 - the direct URL still works
        if not quiet:
            _say(f"{_c('warning:', 'warn')} local proxy unavailable ({_one_line(str(exc))}); use the remote link",
                 err=True)
        return None
    return proxy


def _choose_backend_name(flag: str | None, spec_provider: str) -> str:
    return flag or os.environ.get("NBGPU_BACKEND") or spec_provider


def _number_or_text(value: str | None) -> str | int | None:
    """Plain digits are seconds / MiB (ints); anything else goes to the spec parser."""
    if value is None:
        return None
    return int(value) if value.strip().isdigit() else value


def _find_session(notebook: str, store: SessionStore) -> tuple[Session, Backend]:
    """The notebook's session: the only record, or the only running one."""
    path = Path(notebook).resolve()
    records = lifecycle.session_records(store, path)
    if not records:
        raise CliError(f"no session for {path}; start one with `nbgpu start {notebook}`")
    if len(records) == 1:
        return records[0], get_backend(records[0].backend)
    running = [(s, b) for s in records if lifecycle.resolve_handle(s, b := get_backend(s.backend)) is not None]
    if len(running) == 1:
        return running[0]
    if not running:
        # Only stale records: the newest, so `stop` clears them one by one.
        newest = records[0]
        return newest, get_backend(newest.backend)
    ids = ", ".join(s.session_id for s, _ in running)
    raise CliError(f"{len(running)} sessions are running for {path} ({ids}); stop the extra ones")


def _running_session(notebook: Path, store: SessionStore) -> tuple[Session, Backend] | None:
    for session in lifecycle.session_records(store, notebook):
        backend = get_backend(session.backend)
        if lifecycle.resolve_handle(session, backend) is not None:
            return session, backend
    return None


def _backend_names(store: SessionStore, flag: str | None) -> list[str]:
    """Backends worth asking: those records use, $NBGPU_BACKEND, and `flag`."""
    names = {s.backend for s in store.list()}
    env = os.environ.get("NBGPU_BACKEND")
    if env:
        names.add(env)
    if flag:
        names.add(flag)
    return sorted(names)


def _print_report(report: SyncReport, *, quiet_when_empty: bool = False, err: bool = False) -> None:
    lines = [
        *(f"{_c('↑ pushed  ', 'ok')} {p}" for p in report.pushed),
        *(f"{_c('↓ pulled  ', 'ok')} {p}" for p in report.pulled),
        *(f"{_c('⇅ merged  ', 'ok')} {p}" for p in report.merged),
        *(f"{_c('· skipped ', 'dim')} {p}" for p in report.skipped),
        *(f"{_c('✗ CONFLICT', 'error')} {p}  "
          + _c("(remote version saved next to it as .remote; delete that copy to resolve)", "dim")
          for p in report.conflicts),
    ]
    warnings = [f"{_c('warning:', 'warn')} {w}" for w in report.warnings]
    if not lines and not warnings and not quiet_when_empty:
        lines = [_c("nothing to sync", "dim")]
    for line in lines:
        _say(line, err=err)
    for line in warnings:
        _say(line, err=True)


_ENV_ROLES = {"!": ("!", "warn"), "✓": ("✓", "ok"), "✗": ("✗", "error"), "↻": ("↻", "label"), "·": ("·", "dim")}


def _env_line(message: str) -> str:
    """An EnvWatcher message: its first character picks the symbol and color."""
    symbol, role = _ENV_ROLES.get(message[:1], ("", "label"))
    text = message[1:] if symbol else message
    return f"{_c(symbol + ' env', role)}  {text}"


def _print_session(session: Session, notebook_arg: str, proxy: LocalProxy | None = None) -> None:
    click.echo(f"{_c('✓', 'ok')} marimo is running on {_c(session.backend, 'label')} "
               f"{_c(f'({session.runtime_id})', 'dim')}")
    if proxy is not None:
        click.echo(f"  {_c('open:', 'label')}  {_link(proxy.url_with_token(session.token))}")
        click.echo(f"  {_c('remote:', 'dim')} {_link(session.url_with_token())}  "
                   f"{_c('(direct; slower to load, works without this terminal)', 'dim')}")
    else:
        click.echo(f"  {_c('open:', 'label')}  {_link(session.url_with_token())}")
    rate = lifecycle.session_rate(session)
    note = (f"  {_c(f'({cost.disclaimer(rate.prices_checked, rate.prices_source)})', 'dim')}"
            if rate is not None and rate.complete else "")
    click.echo(f"  {_c('rate:', 'label')}  {_rate_text(rate)}{note}")
    click.echo(f"  {_c('stop:', 'label')}  {_c(f'nbgpu stop {notebook_arg}', 'cmd')}   "
               f"{_c('(the runtime keeps running and billing until then)', 'dim')}")


def _idle_text(info: Mapping[str, object]) -> str:
    timeout = info.get("idle_timeout_s")
    return f"after {_duration(float(timeout))} without activity" if isinstance(timeout, int | float) else "for idleness"


def _unsynced_note(idle_timeout_s: float | None, grace_s: float | None) -> str:
    if idle_timeout_s is None:
        when = "for idleness"
    else:
        extra = f", or up to {_duration(grace_s)} more while browser edits are unsynced" if grace_s else ""
        when = f"for idleness (after {_duration(idle_timeout_s)} without activity{extra})"
    return (
        "Browser edits are saved to your local files only by `nbgpu sync`, `nbgpu open` or `nbgpu stop` "
        f"(or a running sync loop). If the session stops {when}, browser edits not yet synced can be lost."
    )


def _say_stopped_elsewhere(session: Session, *, err: bool) -> None:
    """The record is gone: `nbgpu stop` elsewhere pulled and terminated."""
    _say(f"{_c('✓', 'ok')} session {_c(session.session_id, 'id')} was stopped (by `nbgpu stop`); "
         "its files were pulled first. Stopped syncing.", err=err)


class _WatchdogMonitor:
    """Tells, with one file read per tick, whether the idle watchdog stopped
    marimo (returns its marker), and restarts the watchdog if it died."""

    def __init__(self) -> None:
        self.last: bytes | None = None
        self.changed_at = time.monotonic()
        self.next_restart = 0.0

    def _stale_after(self) -> float:
        interval = watchdog.CHECK_INTERVAL_S
        if self.last is not None:
            try:
                interval = float(json.loads(self.last).get("interval", interval))
            except (ValueError, TypeError, AttributeError):
                pass
        return 3 * interval + WATCHDOG_STALE_SLACK_S

    def check(self, session: Session, backend: Backend, handle: RuntimeHandle, *, err: bool) -> Mapping[str, object] | None:
        beat = lifecycle.read_heartbeat(backend, handle)
        now = time.monotonic()
        if beat is not None and beat != self.last:
            self.last, self.changed_at = beat, now
            return None
        if beat is not None and now - self.changed_at < self._stale_after():
            return None
        idle = lifecycle.read_idle_marker(backend, handle)
        if idle is not None:
            return idle
        if now < self.next_restart:
            return None
        # Missing (exited) or stale (hung or killed): nothing would stop billing.
        try:
            lifecycle.restart_watchdog(session, backend, handle)
        except Exception as exc:  # noqa: BLE001
            self.next_restart = now + WATCHDOG_RESTART_BACKOFF_S
            _say(f"{_c('warning:', 'warn')} the idle watchdog is not running and could not be restarted ({_one_line(str(exc))}); "
                 f"the runtime keeps billing until `nbgpu stop {session.notebook.name}` or its max runtime",
                 err=True)
        else:
            _say("the idle watchdog had stopped; restarted it", err=err)
            self.last, self.changed_at = None, time.monotonic()
        return None


def _record_packages(session: Session, events: queue.Queue, env_watcher: EnvWatcher | None, *, err: bool) -> None:
    """Write packages installed or removed in marimo's UI into the notebook's
    header on this machine (the next sync pushes it)."""
    while True:
        try:
            added, removed = events.get_nowait()
        except queue.Empty:
            return
        try:
            text = session.notebook.read_text(encoding="utf-8")
            new_text, deps = update_dependencies(text, add=added, remove=removed)
        except (OSError, ValueError) as exc:
            _say(f"{_c('! env', 'warn')}  could not record {', '.join(added + removed)} in the header: "
                 f"{_one_line(str(exc))}", err=err)
            continue
        if new_text != text:
            session.notebook.write_text(new_text, encoding="utf-8")
        if env_watcher is not None:
            env_watcher.adopt()  # already installed by marimo: nothing to install again
        changes = [f"+{a}" for a in added] + [f"-{r}" for r in removed]
        _say(f"{_c('✓ env', 'ok')}  recorded {', '.join(changes)} in {session.notebook.name}'s dependencies "
             "(changed in marimo)", err=err)


def _sync_loop(
    session: Session,
    backend: Backend,
    store: SessionStore,
    notebook_arg: str,
    *,
    err: bool,
    packages: queue.Queue | None = None,
) -> None:
    """Sync every second until Ctrl-C (or SIGTERM/SIGHUP), then do a final
    pull; the runtime keeps running. If the idle watchdog stopped marimo,
    pull, terminate the runtime and remove the record instead."""
    with lifecycle.signals_as_interrupt():
        _say(_c("syncing every second; Ctrl-C to stop syncing (the runtime keeps running)", "dim"), err=err)
        last_conflicts: list[str] = []
        engine: SyncEngine | None = None
        handle: RuntimeHandle | None = None
        env_watcher: EnvWatcher | None = None
        monitor = _WatchdogMonitor()
        next_idle_stop = 0.0
        try:
            while True:
                if store.load(session.session_id) is None:
                    _say_stopped_elsewhere(session, err=err)
                    return
                if packages is not None:
                    _record_packages(session, packages, env_watcher, err=err)
                try:
                    if engine is None or handle is None:
                        handle = backend.get(session.runtime_id)
                        if handle is None and store.load(session.session_id) is None:
                            _say_stopped_elsewhere(session, err=err)
                            return
                        if handle is None:
                            raise lifecycle.SessionNotRunning(
                                f"the runtime of session {session.session_id} ({session.runtime_id}) is not running"
                            )
                        engine = lifecycle.session_engine(session, backend=backend, store=store, defer_manifest=True)
                        env_watcher = EnvWatcher(session, backend, handle, store)
                    idle = monitor.check(session, backend, handle, err=err)
                    if idle is not None:
                        if time.monotonic() >= next_idle_stop:
                            if _stop_idle(session, backend, store, engine, idle, notebook_arg, err=err):
                                return
                            next_idle_stop = time.monotonic() + IDLE_STOP_RETRY_S
                        time.sleep(SYNC_INTERVAL_S)
                        continue
                    report = engine.poll_once()
                    lifecycle.note_synced(session, store)

                except lifecycle.SessionNotRunning:
                    raise
                except Exception as exc:  # noqa: BLE001 - transient; retry next tick
                    gone = False
                    try:
                        gone = backend.get(session.runtime_id) is None
                    except Exception:  # noqa: BLE001
                        pass
                    if gone and store.load(session.session_id) is None:
                        _say_stopped_elsewhere(session, err=err)
                        return
                    if gone:
                        raise lifecycle.SessionNotRunning(
                            f"the runtime of session {session.session_id} ({session.runtime_id}) is no longer "
                            f"running; {lifecycle.lost_edits_warning(session, store)}"
                        ) from exc
                    _say(f"{_c('sync failed, retrying:', 'warn')} {_one_line(str(exc))}", err=True)
                else:
                    new_conflicts = report.conflicts != last_conflicts
                    last_conflicts = report.conflicts
                    shown = SyncReport(report.pushed, report.pulled, report.merged,
                                       report.conflicts if new_conflicts else [],
                                       report.skipped if report.changed else [], report.warnings)
                    _print_report(shown, quiet_when_empty=True, err=err)
                    if env_watcher is not None:
                        try:
                            for message in env_watcher.tick():
                                _say(_env_line(message), err=err)
                        except Exception as exc:  # noqa: BLE001 - never stops syncing
                            _say(f"{_c('! env', 'warn')}  could not check dependencies: {_one_line(str(exc))}", err=err)
                time.sleep(SYNC_INTERVAL_S)
        except KeyboardInterrupt:
            # Pull first: after SIGHUP the terminal may be gone, and output
            # must never cost the user's edits.
            if engine is None:
                report = lifecycle.sync_once(session, backend=backend, store=store)
            else:
                report = engine.final_pull()
                engine.flush_manifest()
                lifecycle.note_synced(session, store)
            _say("final pull done", err=err)
            _print_report(report, quiet_when_empty=True, err=err)
            note = _unsynced_note(lifecycle.session_idle_timeout_s(session), lifecycle.session_unsynced_grace_s(session))
            _say(f"{_c('stopped syncing; the runtime is still running and billing.', 'warn')} {note} "
                 f"Stop it with: {_c(f'nbgpu stop {notebook_arg}', 'cmd')}", err=err)


def _stop_idle(
    session: Session,
    backend: Backend,
    store: SessionStore,
    engine: SyncEngine,
    info: Mapping[str, object],
    notebook_arg: str,
    *,
    err: bool,
) -> bool:
    """Final pull and terminate after an idle stop. Returns False (runtime
    and record kept) if the final pull failed; the caller retries."""
    _say(f"marimo was stopped {_idle_text(info)} (no file changes, running cells, CPU or GPU use); "
         "pulling and terminating the runtime...", err=err)
    try:
        report = lifecycle.stop(session, backend=backend, store=store, ignore_conflicts=True, engine=engine)
    except Exception as exc:  # noqa: BLE001 - keep everything; retried
        _say(f"the final pull failed ({_one_line(str(exc))}); the runtime and session record are kept so no edit "
             f"is lost, and the stop is retried every {_duration(IDLE_STOP_RETRY_S)}. "
             f"You can also run: nbgpu stop {notebook_arg}", err=True)
        return False
    _print_report(report, quiet_when_empty=True, err=err)
    _say(f"{_c('✓', 'ok')} session {_c(session.session_id, 'id')} stopped for idleness; the runtime is terminated "
         f"and no longer billing. Start again with: {_c(f'nbgpu start {notebook_arg}', 'cmd')}", err=err)
    return True


def _detach_warning(idle_timeout_s: int, max_runtime_s: int) -> str:
    grace = lifecycle.unsynced_grace_for(idle_timeout_s)
    return (
        "note: with --detach nothing syncs in the background. " + _unsynced_note(idle_timeout_s, grace)
        + f" The same holds when the session reaches its max runtime ({_duration(max_runtime_s)})."
    )


# ---------------------------------------------------------------------------
# commands


json_option = click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
project_root_option = click.option(
    "--project-root",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to upload (default: nearest ancestor with pyproject.toml or .git below your home).",
)


@click.group(cls=_Group)
@click.version_option()
def main() -> None:
    """Run local marimo notebooks on remote GPUs."""


@main.command()
@click.argument("notebook")
@click.option("--backend", "backend_name", help="Backend to use (default: $NBGPU_BACKEND, else the spec's provider).")
@click.option("--gpu", help='GPU, e.g. "A10G" or "H100:2".')
@click.option("--cpu", type=float, help="CPU cores.")
@click.option("--memory", help='Memory, e.g. "32GiB" (plain numbers are MiB).')
@click.option("--idle-timeout", help='e.g. "20m" (plain numbers are seconds).')
@click.option("--max-runtime", help='e.g. "8h" (plain numbers are seconds).')
@project_root_option
@click.option("--detach", is_flag=True, help="Return once marimo is ready instead of syncing.")
@click.option("--fresh", is_flag=True, help="Build the environment even if a snapshot of it was saved.")
@json_option
def start(
    notebook: str,
    backend_name: str | None,
    gpu: str | None,
    cpu: float | None,
    memory: str | None,
    idle_timeout: str | None,
    max_runtime: str | None,
    project_root: Path | None,
    detach: bool,
    fresh: bool,
    as_json: bool,
) -> None:
    """Start a session for NOTEBOOK."""
    overrides: dict[str, object] = {
        "gpu": gpu,
        "cpu": cpu,
        "memory": _number_or_text(memory),
        "idle-timeout": _number_or_text(idle_timeout),
        "max-runtime": _number_or_text(max_runtime),
    }
    spec = load_spec(Path(notebook), overrides)
    name = _choose_backend_name(backend_name, spec.compute.provider)
    store = SessionStore()
    existing = _running_session(spec.path, store)
    if existing is not None:
        raise CliError(
            f"{spec.path} already has a running session ({existing[0].session_id}); "
            f"use `nbgpu open {notebook}` or `nbgpu stop {notebook}`"
        )
    backend = get_backend(name)
    if not as_json:
        click.echo(f"starting {_c(spec.path.name, 'label')} on {_c(name, 'label')}...", err=True)
    if detach:
        click.echo(_detach_warning(spec.compute.idle_timeout_s, spec.compute.max_runtime_s), err=True)
    started = time.monotonic()

    def progress(message: str) -> None:
        if not as_json:
            _say(f"  {_c(f'{time.monotonic() - started:5.1f}s', 'dim')}  {message}", err=True)

    session = lifecycle.start(spec.path, backend=backend, store=store, overrides={**overrides, "provider": name},
                              project_root=project_root, progress=progress, fresh=fresh)
    if not as_json:
        _say(f"  {_c(f'{time.monotonic() - started:5.1f}s', 'dim')}  ready", err=True)
    packages: queue.Queue = queue.Queue()
    proxy = None if detach else _start_proxy(session, quiet=as_json, packages=packages)
    if as_json:
        _emit_json(_session_json(session, proxy))
    else:
        _print_session(session, notebook, proxy)
    if not detach:
        sys.stdout.flush()
        try:
            _sync_loop(session, backend, store, notebook, err=as_json, packages=packages)
        finally:
            if proxy is not None:
                proxy.stop()


@main.command("open")
@click.argument("notebook")
@json_option
def open_(notebook: str, as_json: bool) -> None:
    """Print the URL of NOTEBOOK's running session and resume syncing."""
    store = SessionStore()
    session, backend = _find_session(notebook, store)
    handle = lifecycle.resolve_handle(session, backend)
    if handle is None:
        raise CliError(f"the session for {session.notebook} is no longer running; `nbgpu stop {notebook}` to clear it")
    if lifecycle.is_starting(session):
        raise CliError(f"the session for {session.notebook} has not finished starting (or its start was "
                       f"interrupted); `nbgpu stop {notebook}` to terminate it")
    idle = lifecycle.read_idle_marker(backend, handle)
    if idle is not None:
        raise CliError(f"the session for {session.notebook} was stopped {_idle_text(idle)}; "
                       f"`nbgpu stop {notebook}` pulls its files and terminates the runtime")
    packages: queue.Queue = queue.Queue()
    proxy = _start_proxy(session, quiet=as_json, packages=packages)
    if as_json:
        _emit_json(_session_json(session, proxy))
    else:
        _print_session(session, notebook, proxy)
    sys.stdout.flush()
    try:
        _sync_loop(session, backend, store, notebook, err=as_json, packages=packages)
    finally:
        if proxy is not None:
            proxy.stop()


@main.command()
@click.argument("notebook")
@json_option
def sync(notebook: str, as_json: bool) -> None:
    """One sync pass for NOTEBOOK's session."""
    store = SessionStore()
    session, backend = _find_session(notebook, store)
    report = lifecycle.sync_once(session, backend=backend, store=store)
    handle = lifecycle.resolve_handle(session, backend)
    env = lifecycle.apply_env(session, backend=backend, handle=handle, store=store).messages if handle else []
    if as_json:
        _emit_json({**_report_json(report), "env": [m[1:] for m in env]})
    else:
        _print_report(report)
        for message in env:
            click.echo(_env_line(message))


_STATE_TEXT = {
    lifecycle.STATE_STARTING: "STARTING (or start interrupted; stop to terminate)",
    lifecycle.STATE_IDLE_UNSYNCED: "IDLE, WAITING FOR UNSYNCED BROWSER EDITS (run `nbgpu sync` to keep them)",
    lifecycle.STATE_IDLE_STOPPED: "IDLE-STOPPED (marimo stopped; stop to pull files and terminate)",
    lifecycle.STATE_NOT_RUNNING: "NOT RUNNING (stop to clear the record)",
}


def _orphan_json(orphan: lifecycle.Orphan) -> dict[str, object]:
    return {
        "backend": orphan.backend,
        "runtime_id": orphan.runtime_id,
        "uptime_s": orphan.uptime_s,
        "session_tag": orphan.handle.tags.get(TAG_SESSION),
    }


def _orphan_line(orphan: lifecycle.Orphan) -> str:
    return (f"{orphan.runtime_id} on {orphan.backend}, up {_duration(orphan.uptime_s)}: "
            "untracked (no session record), still billing")


def _other_line(other: lifecycle.Orphan) -> str:
    return (f"{other.runtime_id} on {other.backend}, up {_duration(other.uptime_s)}, owner "
            f"{other.owner or 'unknown'}: not from this state directory; never stopped by `stop --orphans`")


@main.command()
@click.option("--backend", "backend_name", help="Also look for untracked runtimes on this backend.")
@json_option
def status(backend_name: str | None, as_json: bool) -> None:
    """Every session, with uptime, cost and conflicts, and untracked runtimes."""
    store = SessionStore()
    backends: Mapping[str, Backend] = {name: get_backend(name) for name in _backend_names(store, backend_name)}
    entries = lifecycle.status(backends=backends, store=store)
    errors: dict[str, str] = {}
    others: list[lifecycle.Orphan] = []
    orphans = lifecycle.find_orphans(backends=backends, store=store, errors=errors, others=others)
    tracked_ids = {e.session.runtime_id for e in entries}
    others = [o for o in others if o.runtime_id not in tracked_ids]
    for name, error in errors.items():
        click.echo(f"{_c('warning:', 'warn')} could not list {name} runtimes: {_one_line(error)}", err=True)
    if as_json:
        _emit_json({
            "sessions": [
                {
                    "session_id": e.session.session_id,
                    "notebook": str(e.session.notebook),
                    "backend": e.session.backend,
                    "gpu": e.session.gpu,
                    "running": e.running,
                    "state": e.state,
                    "uptime_s": e.uptime_s,
                    "cost_usd": e.cost_usd,
                    "prices_checked": (e.rate.prices_checked or None) if e.cost_usd is not None and e.rate else None,
                    "prices_source": (e.rate.prices_source or None) if e.cost_usd is not None and e.rate else None,
                    "conflicts": list(e.conflicts),
                    "last_sync_at": e.last_sync_at,
                    "warning": e.warning,
                    "unsynced": list(e.unsynced),
                }
                for e in entries
            ],
            "orphans": [_orphan_json(o) for o in orphans],
            "other_owners": [_orphan_json(o) for o in others],
        })
        return
    if not entries and not orphans and not others:
        click.echo(_c("no sessions", "dim"))
        return
    for e in entries:
        s = e.session
        if e.state == lifecycle.STATE_RUNNING:
            state = _c(f"● running {_duration(e.uptime_s)}", "ok")
        else:
            state = _c(_STATE_TEXT[e.state], "warn")
        spent = _c("unknown", "warn") if e.cost_usd is None else _c(format_usd(e.cost_usd), "money")
        click.echo(f"{_c(s.session_id, 'id')}  {_c(s.notebook, 'label')}")
        click.echo(f"  {s.backend}, gpu {_c(s.gpu or 'none', 'label')}, {state}, rate {_rate_text(e.rate)}, "
                   f"cost so far {spent}")
        if e.conflicts:
            click.echo(f"  {_c('conflicts:', 'error')} {', '.join(e.conflicts)}")
        if e.unsynced:
            click.echo(f"  {_c('unsynced browser edits:', 'warn')} {', '.join(e.unsynced)}")
        if e.last_sync_at is not None and e.running:
            click.echo(f"  {_c('last sync:', 'dim')} {_c(lifecycle.format_time(e.last_sync_at), 'dim')}")
        if e.warning:
            click.echo(f"  {_c('WARNING:', 'error')} {e.warning}")
    for checked, source in sorted({(e.rate.prices_checked, e.rate.prices_source)
                                   for e in entries if e.cost_usd is not None and e.rate}):
        click.echo(_c(f"rate and cost: {cost.disclaimer(checked, source)}", "dim"))
    if any(e.cost_usd is not None for e in entries):
        click.echo(_c("what was actually billed: nbgpu cost", "dim"))
    if orphans:
        click.echo(_c("untracked runtimes (terminate with `nbgpu stop --orphans`):", "warn"))
        for o in orphans:
            click.echo(f"  {_orphan_line(o)}")
    if others:
        click.echo(_c("nbgpu runtimes of other owners (other laptops, teammates or state directories):", "dim"))
        for o in others:
            click.echo(f"  {_other_line(o)}")


@main.command(name="cost")
@click.option("--days", type=click.IntRange(1, 31), default=7, show_default=True,
              help="Complete days to report, plus today so far.")
@click.option("--backend", "backend_name", help="Provider to ask (default: $NBGPU_BACKEND, else modal).")
@json_option
def cost_(days: int, backend_name: str | None, as_json: bool) -> None:
    """What nbgpu's runtimes were actually billed, from the provider.

    Also refreshes the cached copy of your account's current prices, used
    for the rates `start` and `status` show."""
    name = backend_name or os.environ.get("NBGPU_BACKEND") or "modal"
    backend = get_backend(name)
    store = SessionStore()
    price_error = lifecycle.refresh_live_prices(backend, store.directory, force=True)
    report = lifecycle.spend(backend, days)
    if as_json:
        _emit_json({
            "backend": name,
            "start": report.start.isoformat(),
            "end": report.end.isoformat(),
            "days": [{"date": d, "usd": c} for d, c in report.days],
            "by_resource": report.by_resource,
            "total_usd": report.total,
            "prices_refreshed": price_error is None,
            "prices_error": price_error,
        })
        return
    click.echo(f"billed for nbgpu's runtimes on {_c(name, 'label')}, "
               f"{report.start:%Y-%m-%d} to {report.end:%Y-%m-%d %H:%M} UTC:")
    for day, usd in report.days:
        click.echo(f"  {day}  {_c(format_usd(usd), 'money')}")
    click.echo(f"  {'total':<10}  {_c(format_usd(report.total), 'money')}")
    if report.by_resource:
        parts = ", ".join(f"{k} {format_usd(v)}" for k, v in report.by_resource.items())
        click.echo(f"  {_c('by resource:', 'dim')} {parts}")
    click.echo(_c("Complete hours only, and the provider's report can lag by an hour or more; "
                  "`nbgpu status` estimates what is running now.", "dim"))
    if price_error is not None:
        click.echo(f"{_c('warning:', 'warn')} could not refresh prices: {_one_line(price_error)}", err=True)


def _duration(seconds: float) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


@main.command()
@click.argument("notebook")
@click.option("--backend", "backend_name", help="Backend to use (default: $NBGPU_BACKEND, else the spec's provider).")
@click.option("--max-lines", type=click.IntRange(min=1), default=200, show_default=True, help="Output lines to show.")
@click.option("--pull", "pull", multiple=True, metavar="PATTERN",
              help="Also bring back runtime files matching PATTERN (.gitignore syntax, relative to the project "
                   "root), of any type and size up to 200 MB. Repeatable.")
@project_root_option
@json_option
def run(notebook: str, backend_name: str | None, max_lines: int, pull: tuple[str, ...], project_root: Path | None,
        as_json: bool) -> None:
    """Run NOTEBOOK as a script remotely and exit with its exit code.

    Afterwards the files the run wrote come back as with `nbgpu sync`, plus
    any matching --pull. In a session, header changes are installed first."""
    store = SessionStore()
    spec = load_spec(Path(notebook))
    existing = _running_session(spec.path, store)
    output_dir = store.directory / "runs"
    if existing is not None and backend_name in (None, existing[0].backend) and project_root is None:
        session, backend = existing
        result = lifecycle.run(spec.path, backend=backend, session=session, store=store,
                               max_output_lines=max_lines, output_dir=output_dir, pull=pull)
    else:
        name = _choose_backend_name(backend_name, spec.compute.provider)
        result = lifecycle.run(spec.path, backend=get_backend(name), overrides={"provider": name}, store=store,
                               max_output_lines=max_lines, output_dir=output_dir, project_root=project_root,
                               pull=pull)
    if as_json:
        _emit_json({
            "exit_code": result.exit_code,
            "output": result.output,
            "truncated": result.truncated,
            "full_output_path": None if result.full_output_path is None else str(result.full_output_path),
            "pulled": result.pulled,
            "warnings": result.warnings,
            "env": [m[1:] for m in result.env],
        })
    else:
        for message in result.env:
            click.echo(_env_line(message), err=True)
        if result.output:
            click.echo(result.output)
        if result.truncated:
            click.echo(_c(f"(output truncated; full output in {result.full_output_path})", "dim"), err=True)
        for path in result.pulled:
            click.echo(f"{_c('↓ pulled  ', 'ok')} {path}", err=True)
        for warning in result.warnings:
            click.echo(f"{_c('warning:', 'warn')} {warning}", err=True)
    sys.exit(result.exit_code)


@main.command(name="exec")
@click.argument("notebook")
@click.argument("code", required=False)
@click.option("-f", "--file", "code_file", type=click.File("r", encoding="utf-8"),
              help="Read the code from this file (- for stdin).")
@click.option("--cell", "cells", multiple=True, metavar="NAME",
              help="Run this cell (name or id) and the cells that depend on it. Repeatable.")
@click.option("--all", "run_all", is_flag=True, help="Run every cell of the notebook in the kernel.")
@click.option("--list", "list_cells", is_flag=True, help="List the notebook's cells with their status.")
@click.option("--sync/--no-sync", "sync_first", default=None,
              help="Sync first (default: only with --cell/--all, so the kernel runs your current file).")
@click.option("--max-lines", type=click.IntRange(min=1), default=200, show_default=True, help="Output lines to show.")
@json_option
def exec_(notebook: str, code: str | None, code_file: Any, cells: tuple[str, ...], run_all: bool, list_cells: bool,
          sync_first: bool | None, max_lines: int, as_json: bool) -> None:
    """Run CODE in the live kernel of NOTEBOOK's session, or run its cells.

    The code sees the notebook's variables (a model already on the GPU
    stays loaded) and prints the value of its last expression. Names it
    defines are discarded afterwards; objects it changes in place stay
    changed. If the session has no kernel yet, one is started and the
    notebook runs first."""
    if code_file is not None:
        if code is not None:
            raise CliError("give the code either as an argument or with --file, not both")
        code = code_file.read()
    chosen = [code is not None, bool(cells), run_all, list_cells]
    if sum(chosen) != 1:
        raise CliError("give exactly one of: CODE (or --file), --cell, --all, --list")
    store = SessionStore()
    session, backend = _find_session(notebook, store)
    # Text mode prints output as it arrives (bounded like run's); --json
    # keeps stdout to the one document.
    result = lifecycle.exec_in_kernel(
        session, backend=backend, store=store, code=code,
        cells="all" if run_all else (list(cells) if cells else None), list_cells=list_cells,
        sync_first=sync_first, max_output_lines=max_lines, output_dir=store.directory / "runs",
        on_line=None if as_json else _say,
    )
    if as_json:
        _emit_json({
            "ok": result.ok,
            "value": result.value,
            "error": result.error,
            "output": result.output,
            "truncated": result.truncated,
            "full_output_path": None if result.full_output_path is None else str(result.full_output_path),
            "cells": result.cells,
            "kernel_created": result.kernel_created,
            "ran_notebook": result.ran_notebook,
            "pushed": result.pushed,
            "warnings": result.warnings,
            "env": [m[1:] for m in result.env],
        })
        sys.exit(0 if result.ok else 1)
    for message in result.env:
        click.echo(_env_line(message), err=True)
    for path in result.pushed:
        click.echo(f"{_c('↑ pushed  ', 'ok')} {path}", err=True)
    if result.ran_notebook:
        click.echo(_c("· no cell had run in the kernel yet: ran the notebook first", "dim"), err=True)
    if result.truncated:
        click.echo(_c(f"(output truncated; full output in {result.full_output_path})", "dim"), err=True)
    if list_cells:
        width = max((len(str(c.get("name") or "")) for c in result.cells), default=4)
        for c in result.cells:
            status = str(c.get("status") or "-")
            role = "error" if status in ("exception", "marimo-error", "cancelled", "interrupted") else (
                "warn" if status == "stale" else "dim")
            click.echo(f"{str(c.get('name') or ''):<{width}}  {_c(str(c.get('id')), 'id')}  "
                       f"{_c(f'{status:<12}', role)} {_c(str(c.get('first_line') or ''), 'dim')}")
    elif result.value is not None:
        click.echo(result.value)
    for warning in result.warnings:
        click.echo(f"{_c('warning:', 'warn')} {warning}", err=True)
    if not result.ok:
        _fail(result.error or "failed")


@main.command()
@click.argument("notebook", required=False)
@click.option("--force", is_flag=True, help="Stop even with open sync conflicts (keeps the conflict copies).")
@click.option("--orphans", is_flag=True, help="Terminate runtimes this tool created that no session record tracks.")
@click.option("--backend", "backend_name", help="With --orphans: only look on this backend.")
@click.option("--yes", is_flag=True, help="With --orphans: do not ask for confirmation.")
@click.option("--snapshot/--no-snapshot", default=None,
              help="Save the runtime's filesystem so the next start resumes from it (default: the header's "
                   "`snapshot` key).")
@json_option
def stop(notebook: str | None, force: bool, orphans: bool, backend_name: str | None, yes: bool,
         snapshot: bool | None, as_json: bool) -> None:
    """Final pull, terminate the runtime, remove the record (or --orphans)."""
    store = SessionStore()
    if orphans:
        if notebook is not None:
            raise CliError("give either NOTEBOOK or --orphans, not both")
        _stop_orphans(store, backend_name, yes=yes, as_json=as_json)
        return
    if notebook is None:
        raise CliError("missing NOTEBOOK (or use --orphans)")
    session, backend = _find_session(notebook, store)
    was_running = lifecycle.resolve_handle(session, backend) is not None
    report = lifecycle.stop(session, backend=backend, store=store, force=force, snapshot=snapshot)
    if as_json:
        _emit_json({**_report_json(report), "snapshot": report.snapshot})
    else:
        _print_report(report, quiet_when_empty=True)
        if report.snapshot is not None:
            click.echo(f"{_c('✓', 'ok')} saved a snapshot of the runtime's filesystem "
                       f"({report.snapshot['seconds']:.0f}s); the next `nbgpu start {notebook}` resumes from it "
                       "(--fresh to build instead)")
        if was_running:
            click.echo(f"{_c('✓', 'ok')} stopped session {_c(session.session_id, 'id')}; the runtime is terminated")
        else:
            click.echo(f"{_c('✓', 'ok')} removed the record of session {_c(session.session_id, 'id')}; "
                       "its runtime was already gone")


def _stop_orphans(store: SessionStore, backend_name: str | None, *, yes: bool, as_json: bool) -> None:
    if as_json and not yes:
        raise CliError("stop --orphans --json needs --yes (there is no confirmation prompt in --json mode)")
    names = [backend_name] if backend_name else _backend_names(store, None)
    backends: Mapping[str, Backend] = {name: get_backend(name) for name in names}
    found = lifecycle.find_orphans(backends=backends, store=store)
    if not as_json:
        if not found:
            looked = ", ".join(names) or "no backends (no records; pass --backend NAME)"
            click.echo(f"no untracked runtimes on {looked}")
            return
        click.echo("untracked runtimes:")
        for o in found:
            click.echo(f"  {_orphan_line(o)}")
        if not yes:
            click.confirm(f"terminate {len(found)} runtime(s)?", abort=True)
    result = lifecycle.stop_orphans(found, backends=backends, store=store)
    if as_json:
        _emit_json({
            "orphans": [_orphan_json(o) for o in found],
            "terminated": [o.runtime_id for o in result.terminated],
            "skipped": [o.runtime_id for o in result.skipped_young],
            "failed": result.failed,
        })
    else:
        for o in result.terminated:
            click.echo(f"{_c('✓', 'ok')} terminated {_c(o.runtime_id, 'id')}")
        for o in result.skipped_young:
            click.echo(f"skipped {o.runtime_id}: created under {_duration(lifecycle.ORPHAN_GRACE_S)} ago, "
                       "a start may still own it; run again later")
        for rid, error in result.failed.items():
            click.echo(f"{_c('FAILED', 'error')} to terminate {rid}: {_one_line(error)}", err=True)
    if result.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
