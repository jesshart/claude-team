"""CLI that launches named `claude` sessions for a Boss / Developer / QA team.

Each role maps to a model and an optional opening prompt. The session name is
built as ``"<Role>: <topic>"`` — where ``topic`` is a Linear ticket (e.g.
``PROJ-123``) or any free-form string — with an optional ``--suffix`` appended
(e.g. ``"PROJ-123 Handoff 1"``) so you can spin up a fresh agent on the same
topic without a name collision.

Launch mode is decided from the environment: a human at a real terminal gets a
foreground session that takes over the terminal; a session spawned by another
Claude agent (or any non-TTY caller) gets a background session with an
attachable id. ``--bg`` / ``--fg`` override the guess.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional

import typer

# --- Models --------------------------------------------------------------
# `fable` is an alias for the latest Fable; Opus 4.8 is pinned by full name
# because the `opus` alias tracks the newest Opus, not this specific version.
FABLE = "fable"
OPUS_48 = "claude-opus-4-8"

CHECK_IN_PROMPT = "Check in with boss"


@dataclass(frozen=True)
class Role:
    """A team role: its display label, model, and optional opening prompt."""

    label: str
    model: str
    prompt: Optional[str] = None


BOSS = Role(label="Boss", model=FABLE)
DEVELOPER = Role(label="Developer", model=OPUS_48, prompt=CHECK_IN_PROMPT)
QA = Role(label="QA", model=OPUS_48, prompt=CHECK_IN_PROMPT)

ROLES = {"boss": BOSS, "developer": DEVELOPER, "qa": QA}

# Tokens accepted by `team --roles` (dev and developer are aliases).
TEAM_ROLES = {"boss": BOSS, "dev": DEVELOPER, "developer": DEVELOPER, "qa": QA}


# --- Pure helpers (unit-tested) ------------------------------------------
def build_name(role: Role, topic: str, suffix: Optional[str]) -> str:
    """Compose the session name, e.g. ``"Boss: PROJ-123 Handoff 1"``."""
    name = f"{role.label}: {topic.strip()}"
    if suffix and suffix.strip():
        name = f"{name} {suffix.strip()}"
    return name


def decide_background(
    override: Optional[bool], env: Mapping[str, str], isatty: bool
) -> bool:
    """Resolve whether to launch in the background.

    Explicit ``override`` (from ``--bg`` / ``--fg``) always wins. Otherwise
    auto-detect: a foreground session only makes sense with a real terminal, so
    launch in the background when there is no TTY or when we are running inside
    another Claude session (``CLAUDECODE`` is set by Claude Code's Bash tool).
    """
    if override is not None:
        return override
    return (not isatty) or ("CLAUDECODE" in env)


def build_command(role: Role, topic: str, suffix: Optional[str], background: bool) -> List[str]:
    """Build the ``claude`` argv for ``role``."""
    cmd = ["claude"]
    if background:
        cmd.append("--bg")
    cmd += ["--name", build_name(role, topic, suffix), "--model", role.model]
    if role.prompt:
        cmd.append(role.prompt)
    return cmd


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def extract_session_id(text: str) -> Optional[str]:
    """Pull the short attach id out of `claude --bg` output, if present.

    The live format is::

        backgrounded · <id> · <name> (idle — …)
          claude attach <id>    open in this terminal

    so we anchor on those markers first (the id always precedes the echoed
    name), and only fall back to a bare hex token for unknown formats.
    """
    clean = _ANSI_RE.sub("", text or "")
    for pattern in (
        r"backgrounded\s*·\s*([0-9a-f]{6,})\b",
        r"claude\s+attach\s+([0-9a-f]{6,})\b",
    ):
        match = re.search(pattern, clean)
        if match:
            return match.group(1)
    match = re.search(r"\b[0-9a-f]{6,}\b", clean)
    return match.group(0) if match else None


def validate_topic(value: str) -> str:
    """Reject an empty / whitespace-only topic before it becomes a junk name."""
    if not value.strip():
        raise typer.BadParameter("TOPIC must not be empty.")
    return value


def stdout_isatty() -> bool:
    """Whether stdout is a real terminal (indirected so tests can control it)."""
    return sys.stdout.isatty()


# --- Launch plans (Command pattern) --------------------------------------
@dataclass(frozen=True)
class LaunchPlan:
    """An inert, fully-resolved description of one `claude` session to start.

    Separating the *plan* from the *doing* is what lets `team` build several,
    preview them all, then execute them — and keeps the argv logic pure.
    """

    role: Role
    topic: str
    suffix: Optional[str]
    background: bool

    @property
    def name(self) -> str:
        return build_name(self.role, self.topic, self.suffix)

    def argv(self) -> List[str]:
        return build_command(self.role, self.topic, self.suffix, self.background)

    def preview(self) -> str:
        return shlex.join(self.argv())


def spawn_background(plan: LaunchPlan) -> "subprocess.CompletedProcess":
    """Start a background session and return the completed subprocess."""
    return subprocess.run(plan.argv(), capture_output=True, text=True)


def parse_roles(spec: str) -> List[Role]:
    """Parse a `--roles` spec like ``"boss,qa"`` into Roles in canonical order."""
    tokens = [t.strip().lower() for t in spec.split(",") if t.strip()]
    if not tokens:
        raise typer.BadParameter("no roles selected")
    unknown = [t for t in tokens if t not in TEAM_ROLES]
    if unknown:
        raise typer.BadParameter(
            f"unknown role(s): {', '.join(unknown)}; choose from boss, dev, qa"
        )
    selected = {TEAM_ROLES[t] for t in tokens}
    # Canonical order so the Boss is always launched first.
    return [r for r in (BOSS, DEVELOPER, QA) if r in selected]


# --- Launch --------------------------------------------------------------
def launch(
    role: Role,
    topic: str,
    suffix: Optional[str],
    background_override: Optional[bool],
    dry_run: bool,
) -> None:
    """Build the invocation for ``role`` and start it (or preview it)."""
    background = decide_background(background_override, os.environ, stdout_isatty())
    plan = LaunchPlan(role, topic, suffix, background)

    if dry_run:
        typer.echo(plan.preview())
        raise typer.Exit()

    if background:
        # Run claude --bg, relay its output, and surface the attach command.
        result = spawn_background(plan)
        if result.stdout:
            sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
        session_id = extract_session_id(result.stdout)
        if session_id:
            typer.echo(f"→ attach with: claude attach {session_id}")
        raise typer.Exit(result.returncode)

    # Foreground: replace this process with claude so it owns the TTY directly.
    argv = plan.argv()
    os.execvp(argv[0], argv)


def launch_team(plans: List[LaunchPlan], dry_run: bool) -> None:
    """Preview or start a batch of background sessions, one summary line each.

    A batch is always background: a foreground launch would `execvp`-replace this
    process on the first session, so the rest would never start.
    """
    if dry_run:
        for plan in plans:
            typer.echo(plan.preview())
        raise typer.Exit()

    failures = 0
    for plan in plans:
        result = spawn_background(plan)
        if result.stderr:
            sys.stderr.write(result.stderr)
        session_id = extract_session_id(result.stdout)
        if result.returncode == 0 and session_id:
            typer.echo(f"  ✓ {plan.name}  →  claude attach {session_id}")
        else:
            failures += 1
            typer.echo(f"  ✗ {plan.name}  (launch failed, rc={result.returncode})")
    raise typer.Exit(1 if failures else 0)


# --- State detection for quickstart --------------------------------------
def fetch_agents() -> Optional[List[dict]]:
    """Return the live session list from `claude agents --json`, or None."""
    if not shutil.which("claude"):
        return None
    try:
        result = subprocess.run(
            ["claude", "agents", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def classify_team(agents: List[dict]) -> dict:
    """Count sessions that look like our roles, by the leading word of the name."""
    counts = {"boss": 0, "developer": 0, "qa": 0}
    for agent in agents:
        name = (agent.get("name") or "").strip().lower()
        head = name.split(":")[0].strip()
        first = head.split()[0] if head else ""
        if first in counts:
            counts[first] += 1
    return counts


def detect_state(
    env: Mapping[str, str],
    isatty: bool,
    agents_provider: Optional[Callable[[], Optional[List[dict]]]] = None,
) -> dict:
    """Gather the repo/environment state the quickstart guide reports."""
    # Resolve at call time so tests can monkeypatch `fetch_agents`.
    provider = agents_provider or fetch_agents
    agents = provider()
    background = decide_background(None, env, isatty)
    return {
        "claude_cli": shutil.which("claude"),
        "claude_team_cli": shutil.which("claude-team"),
        "launch_context": "background" if background else "foreground",
        "in_claude_session": "CLAUDECODE" in env,
        "isatty": isatty,
        "active_sessions": None if agents is None else len(agents),
        "team": None if agents is None else classify_team(agents),
    }


# --- Typer app -----------------------------------------------------------
app = typer.Typer(
    help="Launch named Claude Code sessions for a Boss / Developer / QA team.",
    no_args_is_help=True,
    add_completion=False,
)

TopicArg = typer.Argument(
    ...,
    metavar="TOPIC",
    callback=validate_topic,
    help="Linear ticket (e.g. PROJ-123) or any custom topic string.",
)
SuffixOpt = typer.Option(
    None,
    "--suffix",
    "-s",
    help='Appended after the topic, e.g. "Handoff 1" -> "PROJ-123 Handoff 1".',
)
BgOpt = typer.Option(
    None,
    "--bg/--fg",
    help="Force background (attachable) or foreground. Default: auto-detect.",
)
DryRunOpt = typer.Option(
    False,
    "--dry-run",
    help="Print the claude command that would run, then exit.",
)


@app.command()
def boss(
    topic: str = TopicArg,
    suffix: Optional[str] = SuffixOpt,
    background: Optional[bool] = BgOpt,
    dry_run: bool = DryRunOpt,
) -> None:
    """Launch the Boss agent (Fable, no opening prompt)."""
    launch(BOSS, topic, suffix, background, dry_run)


@app.command()
def dev(
    topic: str = TopicArg,
    suffix: Optional[str] = SuffixOpt,
    background: Optional[bool] = BgOpt,
    dry_run: bool = DryRunOpt,
) -> None:
    """Launch the Developer agent (Opus 4.8, opens by checking in with boss)."""
    launch(DEVELOPER, topic, suffix, background, dry_run)


@app.command()
def qa(
    topic: str = TopicArg,
    suffix: Optional[str] = SuffixOpt,
    background: Optional[bool] = BgOpt,
    dry_run: bool = DryRunOpt,
) -> None:
    """Launch the QA agent (Opus 4.8, opens by checking in with boss)."""
    launch(QA, topic, suffix, background, dry_run)


@app.command()
def team(
    topic: str = TopicArg,
    suffix: Optional[str] = SuffixOpt,
    roles: str = typer.Option(
        "boss,dev,qa",
        "--roles",
        help="Comma-separated subset of the team to launch, e.g. boss,qa.",
    ),
    dry_run: bool = DryRunOpt,
) -> None:
    """Launch the whole team on TOPIC — each role backgrounded and attachable."""
    plans = [
        LaunchPlan(role, topic, suffix, background=True) for role in parse_roles(roles)
    ]
    launch_team(plans, dry_run)


def render_quickstart(state: dict) -> str:
    """Render the agent-oriented setup + usage guide from detected state."""
    def check(ok: bool, name: str, ok_msg: str, fix: Optional[str] = None) -> str:
        tag = "[ok]     " if ok else "[missing]"
        line = f"{tag} {name} - {ok_msg}"
        if not ok and fix:
            line += f"\n          fix: {fix}"
        return line

    team = state["team"]
    if state["active_sessions"] is None:
        active = "[info]    active_team - could not query `claude agents` (is claude on PATH?)"
    else:
        t = team or {"boss": 0, "developer": 0, "qa": 0}
        active = (
            f"[info]    active_team - {state['active_sessions']} live sessions "
            f"({t['boss']} boss / {t['developer']} developer / {t['qa']} qa)"
        )

    lines = [
        "# claude-team setup",
        "",
        "## Current state",
        "",
        check(bool(state["claude_cli"]), "claude_cli", "claude on PATH",
              "install Claude Code (https://claude.com/claude-code)"),
        check(bool(state["claude_team_cli"]), "claude_team_cli",
              "claude-team on PATH", "uv tool install ."),
        f"[info]    launch_context - launches {state['launch_context']} "
        f"(tty={state['isatty']}, in_claude_session={state['in_claude_session']})",
        active,
        "",
        "## How claude-team works",
        "",
        "Three roles, each a named `claude` session pinned to a model:",
        "",
        "| command | session name       | model             | opening prompt      |",
        "| ------- | ------------------ | ----------------- | ------------------- |",
        "| boss    | Boss: <topic>      | fable             | (none)              |",
        "| dev     | Developer: <topic> | claude-opus-4-8   | Check in with boss  |",
        "| qa      | QA: <topic>        | claude-opus-4-8   | Check in with boss  |",
        "",
        "`<topic>` is a Linear ticket or any string. `--suffix \"Handoff 1\"` appends"
        " to the name so you can spin up a fresh agent on the same topic.",
        "",
        "Launch mode is auto-detected: a human terminal gets a **foreground**"
        " session; a session spawned by another agent (no TTY / CLAUDECODE set)"
        " gets a **background** session with an attachable id. Override with"
        " `--bg` / `--fg`.",
        "",
        "Gotcha: a foreground session spawned by another agent has no terminal and"
        " no attach id — it is unreachable. Always launch agent-spawned team"
        " members with `--bg` (which is the auto-default in that context).",
        "",
        "## Playbook",
        "",
        "You are helping a human run a Boss/Developer/QA team. Confirm before"
        " launching sessions on their behalf.",
        "",
        "Start a team on a ticket:",
        "```",
        "claude-team boss PROJ-123        # foreground for you, or --bg to attach later",
        "claude-team dev  PROJ-123 --bg",
        "claude-team qa   PROJ-123 --bg",
        "```",
        "",
        "Find and reach sessions:",
        "```",
        "claude agents --json             # every session (interactive + background)",
        "claude attach <id>               # background sessions only",
        "claude --resume <session-id>     # foreground sessions: new run, once idle",
        "```",
    ]
    return "\n".join(lines)


@app.command()
def quickstart(
    as_json: bool = typer.Option(
        False, "--json", help="Emit detected state as JSON (no explainer)."
    ),
) -> None:
    """Print an agent-oriented guide to setting up and using claude-team here."""
    state = detect_state(os.environ, stdout_isatty())
    if as_json:
        typer.echo(json.dumps(state, indent=2))
    else:
        typer.echo(render_quickstart(state))


if __name__ == "__main__":
    app()
