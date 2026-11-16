# Troubleshooting

nbgpu reports errors as one line starting with `nbgpu: error:`. Find the
phrase below.

## Starting

**`Modal: ... failed: AuthError`** (or another authentication error)
You are not logged in to Modal. Run `modal setup`.

**`Unknown Modal GPU 'X'`**
The message lists the accepted names. See [the header reference](notebook-header.md#gpu).

**`unknown key 'X' under [tool.nbgpu]`**
A typo in the header. The message lists the known keys.

**`Modal volume 'X' (for /path) does not exist`**
Create it first: `modal volume create X`.

**`the project directory ... is too large to upload`**
The project root has more than 2,000 files or 500 MB after exclusions. Add
the heavy paths to `.gitignore` or to [`exclude`](notebook-header.md#exclude),
pass a smaller `--project-root`, or move data to a volume.

**`... is directly in /Users/you; refusing to upload that whole directory`**
The notebook is in your home directory with no project around it. Move it
into a directory of its own or pass `--project-root`.

**`... is ignored by .gitignore or [tool.nbgpu] exclude`**
A pattern matches the notebook itself, so it would not be uploaded. Fix the
pattern.

**`the nbgpu state directory ... is inside the project directory`**
Set `NBGPU_STATE_DIR` to somewhere outside the project.

**`already has a running session`**
Use `nbgpu open NOTEBOOK` to attach, or `nbgpu stop NOTEBOOK` first.

**The start sits at "creating the runtime" for minutes**
The image is being built for a new set of dependencies. This happens once
per set. A scarce GPU type can also take a while to schedule.

**`the snapshot could not be used (...); building the environment instead`**
The provider no longer has the snapshot image (expired or deleted). The
start carries on with a normal build; nothing to do.

**`/nbgpu-cache is where the cache volume is mounted`**
A `volumes` entry uses the cache's mount path. Mount yours elsewhere, or set
`cache = false`.

**`cell-cache keeps marimo's cache on the cache volume; it needs `cache``**
The header has `cell-cache = true` and `cache = false`. Turn one of them
around.

**A `mo.persistent_cache` block runs again in every session**
Add `cell-cache = true` to the header. Without it marimo's cache lives on
the runtime and goes when it stops.

**marimo did not become ready; `last output:` follows**
marimo failed on the runtime and its last lines are in the message. The
usual causes are a syntax error in the notebook or a dependency that
installed but cannot be imported. The runtime has already been terminated.

## During a session

**The page loads slowly**
You are on the `remote:` link, or the proxy could not start (a warning says
so at start). Use the `open:` link.

**`sync failed, retrying: ...`**
A transient network or provider error. The loop keeps trying. If the runtime
is really gone the loop exits and says so.

**`✗ CONFLICT path`**
Both sides changed the same lines. See [resolving conflicts](sync.md#conflicts):
merge by hand, then delete the `.remote` copy.

**A file the notebook wrote does not appear locally**
New remote files are pulled only if they are small text files. See
[what comes back](sync.md#what-comes-back). Bring others back with
`nbgpu run --pull PATTERN`; `warnings` in its output says why a match was
left out. Use a volume for anything over 200 MB.

**`the notebook was not run: installing ... failed`**
`nbgpu run` installs packages added to the header before running, and the
install failed. uv's error is in the message. Fix the requirement and run
again; the runtime's environment is unchanged.

**`another install into this runtime is still running`**
A sync loop or another `nbgpu run` is installing into the same session. It
waits up to 30 minutes; run again once that install finishes.

**A local file is missing on the runtime**
It is ignored (`.gitignore`, `exclude`), larger than 50 MB, or outside the
project root.

**`! env  ... changed (...); it applies on the next start`**
Compute and Python version are fixed for the life of a runtime.
`nbgpu stop`, then `nbgpu start`.

**`✗ env  installing ... failed`**
uv's output is in the message. The environment is unchanged. Fix the
dependency and save the notebook to retry.

**`torch.cuda.is_available()` is False**
The header has no `gpu`, or the session is on the local backend
(`NBGPU_BACKEND` set?). `nbgpu status` shows the backend and GPU.

**The session stopped while a long job was running**
The idle watchdog looks at file changes, running cells, CPU and GPU. A job
that waits quietly outside a marimo cell for longer than `idle-timeout` can
look idle. Raise `idle-timeout` for that notebook. Check also that the job
did not simply hit `max-runtime`.

## Stopping

**`open sync conflicts: ...`**
Resolve them, or `nbgpu stop NOTEBOOK --force` to keep your local versions
and the `.remote` copies.

**`the runtime of session ... is gone, so edits made in the browser after the last successful sync (...) could not be recovered`**
The runtime ended without a final pull: max runtime, an idle stop with no
sync loop running, or a manual termination. Your files are as of the time
shown. Keeping the `nbgpu start` terminal open avoids this.

**The final pull failed**
The runtime and record are kept so you can retry. `--force` terminates
anyway and accepts the loss.

## Money

**`nbgpu cost` says `cannot report what was billed`**
The backend has no billing API (the local backend). Use `--backend modal`.

**`could not refresh prices: ...`**
nbgpu could not read your account's rates. Estimates fall back to the last
cached rates (up to 7 days old), then to the built-in list prices, and say
which they used.

**"Is anything still running?"**

```bash
nbgpu status --backend modal
```

Sessions, untracked runtimes and other owners' runtimes are all listed. If
status prints `could not list modal runtimes`, the check was incomplete;
look at Modal's dashboard.

**Untracked runtimes are listed**

```bash
nbgpu stop --orphans
```

**`skipped ...: created under 2m00s ago`**
Orphan cleanup leaves very young runtimes alone because a start may still be
creating them. Run it again in a couple of minutes.

**`the idle watchdog is not running and could not be restarted`**
Nothing will stop this session for idleness. Stop it yourself when done; its
max runtime still applies.

## Starting over

Session records live in the state directory (`~/.local/state/nbgpu` unless
`NBGPU_STATE_DIR` is set). Prefer `nbgpu stop` to deleting files there: a
deleted record leaves its runtime running, and you will need
`nbgpu stop --orphans --backend modal` to find it again.
