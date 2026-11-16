# Using nbgpu with agents

marimo notebooks are plain Python files, which is what makes them workable
for coding agents: an agent can read, diff and edit one with ordinary file
tools. nbgpu adds the missing half, which is running that file on a GPU and
getting the result back as text.

## Why the CLI suits agents

- **`--json` everywhere.** One JSON document on stdout, progress on stderr.
- **One-line errors**, non-zero exit codes, no tracebacks to wade through.
- **`nbgpu run` is a function call.** Notebook in, `{exit_code, output, pulled}`
  out, with the files the run wrote back in the working tree.
- **`nbgpu exec` is a REPL on the GPU.** An expression or a cell in,
  `{ok, value, output}` out, against the live kernel's state.
- **Bounded output.** Long logs are cut to their head and tail, with the
  full text in a file the agent can read selectively.
- **Edits are local.** The agent changes files in the working tree; sync
  moves them. It never needs a shell on the remote machine.
- **Cost is queryable.** `status --json` reports what is running and its
  estimated cost; `cost --json` reports what was actually billed.

## The loop

One-off, when a single run answers the question:

```bash
nbgpu run train.py --json
```

Iterating, when there will be several runs:

```bash
nbgpu start train.py --detach --json
```

```bash
nbgpu run train.py --json
```

```bash
nbgpu sync train.py --json
```

```bash
nbgpu stop train.py --json
```

`run` inside a session syncs first and installs any packages the agent added
to the header, so each run sees the agent's latest edits without rebuilding
anything. After the run, small text files it wrote (`.json`, `.txt`, `.md`,
...) come back on their own; add `--pull 'plots/'` or similar for anything
else (see [`nbgpu run`](cli.md#nbgpu-run)).

Without reloading, when the slow part is loading a model or a dataset:

```bash
nbgpu exec train.py --cell evaluate --json
```

```bash
nbgpu exec train.py "confusion[:3]" --json
```

`exec` works in the session's live kernel. The agent edits a cell in the
file and runs just that cell; the model loaded above it stays on the GPU.
It can also evaluate expressions against what is in memory. Each call takes
a fraction of a second plus the cell's own time. A final
`nbgpu run --json` confirms the result from a clean process. See
[`nbgpu exec`](cli.md#nbgpu-exec).

## Working alongside an agent

The arrangement nbgpu is built for: you have the notebook open in the
browser on the GPU, and an agent edits the same file on your machine.

```bash
nbgpu start train.py
```

Leave that running, then ask the agent to change the notebook. Its edits are
pushed within a second and marimo reloads the affected cells in your tab.
Edits you make in the browser arrive in the file the agent reads. If you
both touch different parts of the file they merge; if you collide, a
`.remote` copy appears and either of you can resolve it.

A dependency the agent adds to the header is installed live by the loop, or
by the agent's next `nbgpu run` or `nbgpu sync`.

The agent's `nbgpu exec` calls run in the same kernel as your tab, so cells
it runs update there too. Its expressions can read your variables, and
in-place changes it makes to them are real. The bundled skill tells it to
ask before mutating state you are working with.

## Things to tell an agent

The bundled skills cover these, but if you are writing your own instructions:

- A runtime bills until stopped. End every task with `nbgpu status --json`
  and stop what was started.
- `nbgpu start` without `--detach`, and `nbgpu open`, never return. Run them
  as background processes or not at all.
- In script mode a cell's last expression is not printed. `print()` results.
- The `url` in `start`'s output contains an access token. Do not echo it
  into logs, commits or issue trackers.
- Add packages by editing the header, not by installing on the runtime.
- Do not escalate the GPU without asking.

## Bundled skills

| Skill | Use |
|-------|-----|
| [`nbgpu`](../src/nbgpu/skills/nbgpu/SKILL.md) | Operating the CLI: choosing a mode, reading results, cost and cleanup, conflicts. Carries a JSON reference and a troubleshooting table. |
| [`nbgpu-notebook`](../src/nbgpu/skills/nbgpu-notebook/SKILL.md) | Authoring: the header, marimo's cell rules, writing for script mode, converting from Jupyter. |
| [`nbgpu-dev`](../.claude/skills/nbgpu-dev/SKILL.md) | Changing nbgpu itself: module map, invariants, tests, how to add a key, flag or backend. |

Skills are folders with a `SKILL.md`. An agent that supports them reads the
short description up front and loads the body only when a task matches.

### Installing the skills

`nbgpu` and `nbgpu-notebook` ship inside the package, matching the CLI
version you have. Install them for every project:

```bash
nbgpu skills install
```

That copies them into `~/.claude/skills`; run it again after upgrading
nbgpu. `--dir` picks another skills directory (a single project's
`.claude/skills/`, or wherever another agent reads `SKILL.md` folders), and
`--link` symlinks instead of copying, which keeps them current from an
editable checkout.

Inside this repository all three are already active for Claude Code:
`.claude/skills/` holds `nbgpu-dev` and links to the other two in
`src/nbgpu/skills/`.

### Keeping them honest

The skills describe the CLI's behaviour, so they go stale the same way docs
do. When a flag, a JSON field or a limit changes, update
`src/nbgpu/skills/nbgpu/references/cli.md` and the matching page here in the same
change. The `nbgpu-dev` skill tells agents working on the code to do this.
