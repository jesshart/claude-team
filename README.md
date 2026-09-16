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

## Lay a team out in Herdr panes

If you run inside [Herdr](https://herdr.dev) (a terminal workspace manager for
coding agents), `claude-team` can build a fresh workspace and launch each role
**foreground in its own pane** — so the whole team is visible and controllable
in one place, not detached. Requires `HERDR_ENV=1` (i.e. you're in a Herdr pane)
and the `herdr` binary on `PATH`; otherwise use `team` above.

```bash
claude-team space PROJ-123                    # new workspace, one tab, boss/dev/qa in panes
claude-team space PROJ-123 --roles boss,qa    # a subset
claude-team space PROJ-123 --layout columns   # side-by-side columns instead of left-stack
claude-team space PROJ-123 --cwd ~/code/app   # working dir for the panes
claude-team space PROJ-123 --dry-run          # preview the herdr commands, launch nothing
```

**Per-role opening prompts.** By default the boss opens with no prompt and dev/qa
open with `"Check in with boss"`. Override any role with `--boss-prompt` /
`--dev-prompt` / `--qa-prompt` — e.g. resume a whole team where each role reloads
its own handoff:

```bash
claude-team space CHOM-893 --suffix "Handoff Day 2" \
  --boss-prompt '/resume_handoff boss.md' \
  --dev-prompt  '/resume_handoff dev.md' \
  --qa-prompt   '/resume_handoff qa.md'
```

Omit a flag and that role keeps its default. A prompt for a role not in `--roles`
is a no-op with a warning. The same three flags work on `worktree` below.

Dev and qa always check in with the boss, even with a custom prompt. Since a
custom prompt is usually a slash command (and Claude Code won't let extra text be
appended to one), the check-in is carried as a `--append-system-prompt`
instruction rather than tacked onto the prompt — the slash command stays the
clean opening message.

Start the team **on a fresh git worktree** in one step — it creates the branch,
the checkout, and the workspace, then launches the team on that branch:

```bash
claude-team worktree jesse/chom-123-thing --repo ~/code/app    # branch off origin/dev
claude-team worktree my-spike --base main --topic "spike idea"  # custom base + topic
claude-team worktree my-spike --dry-run                         # preview, create nothing
```

**Open an existing worktree** with `--open` (uses `herdr worktree open`), so the
team nests under that repo's tree in Herdr instead of floating in a standalone
space — the difference between a *worktree-workspace* and a bare `space`:

```bash
claude-team worktree jesse/chom-123-thing --repo ~/code/app --open   # reuse the existing checkout
```

A git worktree maps to a single Herdr workspace, so `--open` on a worktree that
already has a team reuses that workspace; the command **refuses** to add a second
team into a workspace that already holds agents unless you pass `--force`. With
`--open`, `--base` and `--label` are ignored (the worktree keeps its own label).
Because it opened a pre-existing worktree, teardown just closes the workspace
(`herdr workspace close <ws>`) — it never removes the checkout or deletes the branch.

Both print the workspace id, the pane→role map with live status, and the exact
teardown command:

```
✓ workspace w2E
  Boss      w2E:p1  [idle]
  Developer w2E:p2  [working]
  QA        w2E:p3  [working]
  focus:    herdr workspace focus w2E
  teardown: herdr workspace close w2E
```

**Layouts** (`--layout`): `left-stack` (default — boss on the left full-height,
the rest stacked in an even right column) or `columns` (equal side-by-side
columns). Splits are computed deterministically, so the same roles always
produce the same geometry.

**Teardown.** A plain space: `herdr workspace close <ws>`. A worktree space:
`herdr worktree remove --workspace <ws>` (removes the checkout and closes the
workspace) — it leaves the branch ref behind, so delete that too with
`git -C <repo> branch -D <branch>`. The `worktree` command prints both.

Unlike `team`, each role runs foreground in a pane (`--fg`), so there are no
attach ids — you watch and drive them in Herdr directly. The boss is always
launched first, so the dev/qa "check in with boss" prompt lands on a live
session.

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

## Launching from inside a Claude Code session

`claude-team` is often run *by another agent* — a Boss session spinning up its
Developer and QA. When the parent session is in **auto mode**, what decides
whether the launch is allowed is the *shape* of the command, not `claude-team`
itself.

**Keep the launch a single bare command.** Explicit `permissions.allow` rules
are resolved *before* the auto-mode classifier, so with a rule that matches your
launcher (e.g. an existing `Bash(uvx:*)`), a plain command runs in auto mode
with no prompt:

```bash
uvx --from git+https://github.com/jesshart/claude-team claude-team qa "PROJ-123"
```

**A compound command gets denied.** Permission rules match **per sub-command**,
so a variable assignment, a `cd … &&`, a `;`, a pipe, or a subshell falls
through to the classifier — which denies the launch, with no prompt shown:

```
Permission for this action was denied by the Claude Code auto mode classifier.
Reason: [Create Unsafe Agents]
```

```bash
# denied — three sub-commands, none matched by a uvx allow rule:
DEV=/path/to/worktree; (cd "$DEV" && uvx … claude-team dev "PROJ-123")
```

(`--dry-run` and `quickstart` are unaffected — only a real launch is classified.)

### Targeting another worktree

`claude-team` launches in the caller's current directory and has no `--cwd`
flag, which tempts you toward `cd <worktree> && …` — the compound shape that
gets denied. Avoid it by letting `uvx` change directory, which keeps the whole
thing one matchable command:

```bash
uvx --directory /path/to/other/worktree \
    --from git+https://github.com/jesshart/claude-team claude-team dev "PROJ-123"
```

### Allow rules

Add to your project `.claude/settings.local.json` (or user
`~/.claude/settings.json`):

```json
{ "permissions": { "allow": ["Bash(uvx:*)"] } }
```

- `:*` is a trailing wildcard. `Bash(uvx:*)` allows *any* `uvx` command (broad —
  it lets an agent run arbitrary packages); scope it tighter with the full
  prefix `Bash(uvx --from git+https://github.com/jesshart/claude-team claude-team:*)`,
  or use `Bash(claude-team:*)` if the tool is installed on `PATH`. The runner
  (`uvx`) is *not* stripped, so it has to be part of the prefix you allow.
- `permissions.allow` works in project `.claude/settings.local.json` or user
  `~/.claude/settings.json` (a non-local project `.claude/settings.json` also
  requires that you've trusted the workspace). The `autoMode.*` keys (prose
  allow / soft_deny / hard_deny, `classifyAllShell`, `environment`) are read
  **only** from `~/.claude/settings.json`.
- Inspect the built-in rule behind the denial with
  `claude auto-mode defaults --label 'Create Unsafe Agents'`.

Alternatively, press <kbd>Shift</kbd>+<kbd>Tab</kbd> to cycle the parent session
out of auto mode for the launch and approve the prompt normally. The agent
**cannot** add the allow rule or change the mode for itself — that has to come
from the human.

### Still rough

- **No `--permission-mode` passthrough.** A backgrounded session starts in
  *default* mode and stalls at its first permission prompt until you
  `claude attach <id>` and switch it (<kbd>Shift</kbd>+<kbd>Tab</kbd>). A
  `--permission-mode` flag — or a generic `-- <extra claude args>` — would fix
  this.
- **No `--cwd` option.** `claude-team` always launches in the caller's
  directory; `uvx --directory` (above) is the current workaround.

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

Opening prompts are overridable: the single-role commands take `--prompt`, and
the paned `space` / `worktree` commands take `--boss-prompt` / `--dev-prompt` /
`--qa-prompt`.

## Development

```bash
uv run pytest              # run the test suite
```
