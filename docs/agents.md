# Using nbgpu with agents

marimo notebooks are plain Python files, which is what makes them workable
for coding agents: an agent can read, diff and edit one with ordinary file
tools. nbgpu adds the missing half, which is running that file on a GPU and
getting the result back as text.

## Why the CLI suits agents

- **`--json` everywhere.** One JSON document on stdout, progress on stderr.
- **One-line errors**, non-zero exit codes, no tracebacks to wade through.
- **`nbgpu run` is a function call.** Notebook in, `{exit_code, output}` out.
- **Bounded output.** Long logs are cut to their head and tail, with the
  full text in a file the agent can read selectively.
- **Edits are local.** The agent changes files in the working tree; sync
  moves them. It never needs a shell on the remote machine.
- **Cost is queryable.** `status --json` reports what is running and what it
  has cost.

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

`run` inside a session syncs first, so each run sees the agent's latest
edits without rebuilding anything. `sync` brings back small text files the
run wrote (`.json`, `.txt`, `.md`, ...; see
[Sync and conflicts](sync.md#what-comes-back)).

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

With the sync loop running, a dependency the agent adds to the header is
installed live.

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
| [`nbgpu`](../skills/nbgpu/SKILL.md) | Operating the CLI: choosing a mode, reading results, cost and cleanup, conflicts. Carries a JSON reference and a troubleshooting table. |
| [`nbgpu-notebook`](../skills/nbgpu-notebook/SKILL.md) | Authoring: the header, marimo's cell rules, writing for script mode, converting from Jupyter. |
| [`nbgpu-dev`](../.claude/skills/nbgpu-dev/SKILL.md) | Changing nbgpu itself: module map, invariants, tests, how to add a key, flag or backend. |

Skills are folders with a `SKILL.md`. An agent that supports them reads the
short description up front and loads the body only when a task matches.

### Installing the skills

Inside this repository all three are already active for Claude Code:
`.claude/skills/` holds `nbgpu-dev` and links to the other two.

To have `nbgpu` and `nbgpu-notebook` in every project, link them into your
user skills directory from a checkout of this repository:

```bash
ln -s "$PWD/skills/nbgpu" ~/.claude/skills/nbgpu
```

```bash
ln -s "$PWD/skills/nbgpu-notebook" ~/.claude/skills/nbgpu-notebook
```

Symlinks keep them current as the repository changes. For a single other
project, link or copy them into that project's `.claude/skills/` instead.
Other agents that read `SKILL.md` folders can be pointed at `skills/`
directly.

### Keeping them honest

The skills describe the CLI's behaviour, so they go stale the same way docs
do. When a flag, a JSON field or a limit changes, update
`skills/nbgpu/references/cli.md` and the matching page here in the same
change. The `nbgpu-dev` skill tells agents working on the code to do this.
