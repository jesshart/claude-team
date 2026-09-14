# claude-team

Launch named [Claude Code](https://claude.com/claude-code) sessions for a small
agent team — a **Boss**, a **Developer**, and a **QA** — each pinned to the right
model and opening prompt.

## Install

Requires [`uv`](https://docs.astral.sh/uv/) and `claude` (Claude Code) on your
`PATH` — `claude-team` just launches named `claude` sessions.

### From GitHub (no clone)

```bash
uv tool install git+https://github.com/jesshart/claude-team   # installs `claude-team` + `cteam`
# or run once, without installing:
uvx --from git+https://github.com/jesshart/claude-team claude-team --help
```

Pin a version with `@`, e.g. `git+https://github.com/jesshart/claude-team@v0.1.0`.

### From a local clone

```bash
cd claude-team
uv tool install .          # installs `claude-team` and the short alias `cteam`
# or, without installing:
uv run claude-team --help
```

## Usage

```bash
claude-team boss PROJ-123          # -> claude -n "Boss: PROJ-123"      --model fable
claude-team dev  PROJ-123          # -> claude -n "Developer: PROJ-123" --model claude-opus-4-8 "Check in with boss"
claude-team qa   PROJ-123          # -> claude -n "QA: PROJ-123"        --model claude-opus-4-8 "Check in with boss"
```

`TOPIC` is a Linear ticket **or any free-form string**:

```bash
claude-team dev "refactor the pull-config loader"
```

Use `--suffix` / `-s` to spin up another agent on the same topic without a name
collision:

```bash
claude-team dev PROJ-123 -s "Handoff 1"   # -> claude -n "Developer: PROJ-123 Handoff 1" ...
```

Preview without launching:

```bash
claude-team qa PROJ-123 --dry-run
```

## Launch a whole team at once

```bash
claude-team team PROJ-123                 # boss + dev + qa, each backgrounded
claude-team team PROJ-123 --roles boss,qa # a subset
claude-team team PROJ-123 -s "Handoff 1"  # same topic, fresh names
claude-team team PROJ-123 --dry-run       # preview all launches
```

Each member prints its own `claude attach <id>`:

```
  ✓ Boss: PROJ-123       →  claude attach 1a2b3c4d
  ✓ Developer: PROJ-123  →  claude attach 5e6f7a8b
  ✓ QA: PROJ-123         →  claude attach 9c0d1e2f
```

A batch is **always background** — a foreground launch would replace this
process on the first session, so the rest would never start. That's the point:
every member comes back attachable.

## Foreground vs. background

The launch mode is **auto-detected**:

- **Human at a terminal** → foreground: the session takes over your terminal.
- **Spawned by another agent** (no TTY, or `CLAUDECODE` set by Claude Code's
  Bash tool) → background: it detaches, prints an attach `id`, and stays
  reachable.

Override the guess with `--bg` (force background, attachable) or `--fg` (force
foreground). Only background sessions can be `claude attach`-ed; foreground
sessions can only be `claude --resume`-d once idle.

```bash
claude-team boss PROJ-123 --bg     # detach + print `claude attach <id>`
```

## quickstart

An agent-oriented guide that detects the current
environment and explains how to use the tool. Read-only.

```bash
claude-team quickstart             # printed guide: state checks + how-it-works + playbook
claude-team quickstart --json      # detected state only, for scripting
```

It reports whether `claude` / `claude-team` are on PATH, whether this context
launches foreground or background, and a live snapshot of running team sessions
(from `claude agents --json`).

## Roles

| Command | Session name       | Model            | Opening prompt      |
| ------- | ------------------ | ---------------- | ------------------- |
| `boss`  | `Boss: <topic>`      | `fable`          | (none)              |
| `dev`   | `Developer: <topic>` | `claude-opus-4-8` | `Check in with boss` |
| `qa`    | `QA: <topic>`        | `claude-opus-4-8` | `Check in with boss` |

## Development

```bash
uv run pytest              # run the test suite
```
