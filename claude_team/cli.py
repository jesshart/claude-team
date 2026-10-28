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
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Tuple

import typer

# --- Models --------------------------------------------------------------
# `fable` is an alias for the latest Fable; Opus 5.5 is pinned by full name
# because the `opus` alias tracks the newest Opus, not this specific version.
# Each is only a role default: every launch command takes a model override.
FABLE = "fable"
OPUS_55 = "claude-opus-5-5"

CHECK_IN_PROMPT = "Check in with boss"


@dataclass(frozen=True)
class Role:
    """A team role: its display label, model, and optional opening prompt."""

    label: str
    model: str
    prompt: Optional[str] = None


BOSS = Role(label="Boss", model=FABLE)
DEVELOPER = Role(label="Developer", model=OPUS_55, prompt=CHECK_IN_PROMPT)
QA = Role(label="QA", model=OPUS_55, prompt=CHECK_IN_PROMPT)

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


def build_command(
    role: Role,
    topic: str,
    suffix: Optional[str],
    background: bool,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
) -> List[str]:
    """Build the ``claude`` argv for ``role``.

    ``model`` overrides the role's default model when given (``None`` keeps it).

    ``prompt`` overrides the role's default opening prompt when given — e.g. a
    ``/resume_handoff …`` slash command for the boss, which otherwise opens with
    no prompt. Pass ``None`` to keep the role default.

    Every non-boss role must check in with the boss. With no custom prompt that
    check-in *is* their opening prompt. With a custom prompt it cannot simply be
    appended to the prompt text: the custom prompt is usually a slash command,
    and Claude Code folds any trailing text into the command's ``$ARGUMENTS`` and
    blocks a command containing a newline. So the custom prompt stays the clean
    opening message and the check-in rides along as an ``--append-system-prompt``
    standing instruction instead (skipped if the prompt already checks in).
    """
    cmd = ["claude"]
    if background:
        cmd.append("--bg")
    cmd += ["--name", build_name(role, topic, suffix), "--model", model or role.model]
    if (
        prompt is not None
        and role != BOSS
        and CHECK_IN_PROMPT.lower() not in prompt.lower()
    ):
        cmd += ["--append-system-prompt", f"{CHECK_IN_PROMPT}."]
    effective = prompt if prompt is not None else role.prompt
    if effective:
        cmd.append(effective)
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
    prompt: Optional[str] = None
    model: Optional[str] = None

    @property
    def name(self) -> str:
        return build_name(self.role, self.topic, self.suffix)

    def argv(self) -> List[str]:
        return build_command(
            self.role, self.topic, self.suffix, self.background, self.prompt,
            self.model,
        )

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
    prompt: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """Build the invocation for ``role`` and start it (or preview it).

    ``prompt`` / ``model`` override the role's default opening prompt / model.
    """
    background = decide_background(background_override, os.environ, stdout_isatty())
    plan = LaunchPlan(role, topic, suffix, background, prompt, model)

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


# --- Multiplexers: place a team into terminal panes ----------------------
# Two terminal multiplexers are supported, each behind its own facade:
#   * Herdr (https://herdr.dev) — a workspace manager for coding agents. Its
#     `herdr` binary drives the running session and returns JSON.
#   * Rex (https://www.superlogical.com/rex) — a terminal server whose `rex`
#     binary creates sessions, splits blocks, and types into them.
# These helpers create a workspace / session (or git worktree), lay panes out
# deterministically, and launch one role foreground per pane — where each pane
# just re-invokes *this* CLI (reusing the LaunchPlan command pattern) for a
# single role with `--fg`.
#
# Three patterns from the GoF catalog carry the weight here:
#   * Strategy  — interchangeable pane-layout algorithms (`LayoutStrategy`).
#   * Facade    — `Herdr` / `Rex` hide the subprocess + JSON of their CLIs
#                 behind the same few semantic operations; the commands never
#                 see either, nor which multiplexer they are driving.
#   * Command   — the per-pane launch is `claude-team <role> --fg`, which builds
#                 and runs a `LaunchPlan` just like the standalone commands do.

BOOT_REGEX = "shortcuts|Welcome|bypass|Claude Code|esc to"

# Re-invoke this same CLI for one role, so a token maps back to its subcommand.
ROLE_TOKEN = {BOSS: "boss", DEVELOPER: "dev", QA: "qa"}


def role_token(role: Role) -> str:
    """The CLI subcommand token that launches ``role`` (boss / dev / qa)."""
    return ROLE_TOKEN[role]


@dataclass(frozen=True)
class SplitStep:
    """One `herdr pane split`.

    Split the pane at index ``source`` (into the *growing* list of panes, where
    index 0 is the workspace's root pane) in ``direction`` (``right`` | ``down``).
    ``ratio`` is the fraction of space **kept by the original** pane; the new
    pane gets ``1 - ratio``. ``None`` leaves it to herdr's default.
    """

    source: int
    direction: str
    ratio: Optional[float] = None


# --- Strategy: pane-layout algorithms ------------------------------------
class LayoutStrategy(ABC):
    """A family of interchangeable layout algorithms.

    Each turns a pane count into a deterministic list of :class:`SplitStep`, so
    the same role set always produces the same geometry, run after run.
    """

    name: str

    @abstractmethod
    def plan(self, n: int) -> List[SplitStep]:
        """Return the splits that grow one root pane into ``n`` panes."""


class LeftStackLayout(LayoutStrategy):
    """Boss takes the left half full-height; the rest stack evenly in the right
    column, top to bottom. 3 roles → boss left, dev top-right, qa bottom-right."""

    name = "left-stack"

    def plan(self, n: int) -> List[SplitStep]:
        if n < 1:
            raise ValueError("need at least one pane")
        steps: List[SplitStep] = []
        if n >= 2:
            steps.append(SplitStep(source=0, direction="right", ratio=0.5))
        # Even rows in the right column: the i-th down-split hands the new pane
        # 1/(rows-left) of the shrinking remainder.
        for i in range(1, n - 1):
            steps.append(SplitStep(source=i, direction="down", ratio=1.0 / (n - i)))
        return steps


class ColumnsLayout(LayoutStrategy):
    """``n`` side-by-side columns of equal width, boss leftmost."""

    name = "columns"

    def plan(self, n: int) -> List[SplitStep]:
        if n < 1:
            raise ValueError("need at least one pane")
        # Peel one even column off the shrinking remainder each time: keeping
        # 1/(n-k) of the current width leaves exactly one column on the left.
        return [
            SplitStep(source=k, direction="right", ratio=1.0 / (n - k))
            for k in range(n - 1)
        ]


LAYOUTS = {s.name: s for s in (LeftStackLayout(), ColumnsLayout())}
DEFAULT_LAYOUT = "left-stack"


def get_layout(name: str) -> LayoutStrategy:
    """Resolve a ``--layout`` name to its strategy, or reject it."""
    try:
        return LAYOUTS[name]
    except KeyError:
        raise typer.BadParameter(
            f"unknown layout {name!r}; choose from {', '.join(LAYOUTS)}"
        )


# --- Pure argv builders (shared by the facade and by --dry-run previews) --
def workspace_create_argv(cwd: str, label: str) -> List[str]:
    return ["workspace", "create", "--cwd", cwd, "--label", label, "--no-focus"]


def worktree_create_argv(repo: str, branch: str, base: str, label: str) -> List[str]:
    return [
        "worktree", "create", "--cwd", repo, "--branch", branch,
        "--base", base, "--label", label, "--no-focus",
    ]


def worktree_open_argv(repo: str, branch: str) -> List[str]:
    # herdr resolves the worktree from the repo PARENT, selecting the linked
    # checkout by --branch. Passing the checkout itself as --cwd/--path errors, so
    # --cwd must be the repo root. No --base (the branch already exists) and no
    # --label: opening an already-open worktree with a --label *renames* its
    # workspace, so we never pass one — the worktree keeps its natural label.
    return ["worktree", "open", "--cwd", repo, "--branch", branch, "--no-focus"]


def pane_split_argv(pane: str, step: SplitStep, cwd: str) -> List[str]:
    argv = ["pane", "split", pane, "--direction", step.direction]
    if step.ratio is not None:
        argv += ["--ratio", f"{step.ratio:.4f}"]
    argv += ["--cwd", cwd, "--no-focus"]
    return argv


def pane_run_argv(pane: str, command: str) -> List[str]:
    return ["pane", "run", pane, command]


# --- Relaunch: how a pane re-invokes this CLI for one role ----------------
def claude_team_invocation(
    which: Optional[Callable[[str], Optional[str]]] = None,
    package_root: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> List[str]:
    """Resolve how to invoke this CLI *as a bare command in a fresh pane shell*.

    The pane runs a plain login shell, so the invocation must resolve there — not
    just in our own (possibly ``uv run``) environment. So a ``claude-team`` that
    merely comes from the transient ``uv run`` venv we are in now does not count
    as durably installed.

    Prefer a durably-installed ``claude-team``; else ``uv run --project <clone>
    claude-team`` (works in a bare shell as long as ``uv`` is on PATH — the way
    the tool runs before it is installed); else this interpreter + module.
    """
    resolve = which or shutil.which  # resolve at call time so tests can patch it
    environ = env if env is not None else os.environ
    found = resolve("claude-team")
    venv = environ.get("VIRTUAL_ENV")
    in_current_env = bool(found) and (
        (bool(venv) and found.startswith(venv)) or found.startswith(sys.prefix)
    )
    if found and not in_current_env:
        return ["claude-team"]
    if resolve("uv"):
        root = package_root or str(Path(__file__).resolve().parents[1])
        return ["uv", "run", "--project", root, "claude-team"]
    return [sys.executable, "-m", "claude_team.cli"]


def pane_launch_command(
    invocation: List[str],
    role: Role,
    topic: str,
    suffix: Optional[str],
    prompt: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """The shell command sent into a pane to launch one role foreground there.

    The model is always passed explicitly (``model`` or the role default), so the
    pane runs the model this invocation resolved, whatever its own default is.
    """
    argv = list(invocation) + [role_token(role), topic]
    if suffix and suffix.strip():
        argv += ["--suffix", suffix]
    if prompt:
        argv += ["--prompt", prompt]
    argv += ["--model", model or role.model]
    argv.append("--fg")
    return shlex.join(argv)


# --- Facade over the `herdr` CLI subsystem -------------------------------
HerdrRunner = Callable[[List[str]], "subprocess.CompletedProcess"]


def default_herdr_runner(args: List[str]) -> "subprocess.CompletedProcess":
    """Run ``herdr <args>`` and capture its output (the injectable subsystem)."""
    return subprocess.run(["herdr", *args], capture_output=True, text=True)


class BackendError(RuntimeError):
    """A multiplexer (herdr / rex) or git call failed."""


class HerdrError(BackendError):
    """A ``herdr`` CLI call failed or returned an error payload."""


class Herdr:
    """Facade over the ``herdr`` CLI: semantic operations in, parsed data out.

    The client (the ``space`` / ``worktree`` commands) never touches subprocess
    plumbing or JSON shapes. A ``runner`` is injected so tests can drive the
    facade with canned responses instead of a live herdr server.
    """

    name = "herdr"
    noun = "workspace"
    worktree_unit = "workspace"  # what one worktree maps to

    def __init__(self, runner: Optional[HerdrRunner] = None) -> None:
        # Resolve the module default at call time so tests can monkeypatch it.
        self._run = runner or default_herdr_runner

    def _call(self, args: List[str]) -> dict:
        """Run one herdr command; return its ``.result`` payload or raise.

        Some commands (notably ``pane run``) succeed with no output; an empty
        stdout on a clean exit is treated as an empty result, not an error.
        """
        proc = self._run(args)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise HerdrError(
                f"`herdr {shlex.join(args)}` failed (rc={proc.returncode}): {detail}"
            )
        if not (proc.stdout or "").strip():
            return {}
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise HerdrError(
                f"`herdr {shlex.join(args)}`: non-JSON output: {proc.stdout[:200]!r}"
            )
        if isinstance(data, dict) and data.get("error"):
            raise HerdrError(f"`herdr {shlex.join(args)}`: {data['error']}")
        return data.get("result", {}) if isinstance(data, dict) else {}

    def create_workspace(self, cwd: str, label: str) -> Tuple[str, str]:
        """Create a workspace (one tab); return ``(workspace_id, root_pane_id)``."""
        r = self._call(workspace_create_argv(cwd, label))
        return r["workspace"]["workspace_id"], r["root_pane"]["pane_id"]

    def create_worktree(
        self, repo: str, branch: str, base: str, label: str
    ) -> Tuple[str, str, str]:
        """Create a git worktree + bound workspace; return
        ``(workspace_id, root_pane_id, checkout_path)``."""
        r = self._call(worktree_create_argv(repo, branch, base, label))
        ws = r["workspace"]
        checkout = (ws.get("worktree") or {}).get("checkout_path", "")
        return ws["workspace_id"], r["root_pane"]["pane_id"], checkout

    def open_worktree(self, repo: str, branch: str) -> Tuple[str, str, str]:
        """Open an EXISTING git worktree's bound workspace; return
        ``(workspace_id, root_pane_id, checkout_path)``.

        A worktree maps to a single workspace: opening one already open returns
        that same workspace (its id + current root pane), it does not make a
        second. Callers that mean to *populate* a fresh team must first check the
        workspace is empty (see ``agents_in_workspace``)."""
        r = self._call(worktree_open_argv(repo, branch))
        ws = r["workspace"]
        checkout = (ws.get("worktree") or {}).get("checkout_path", "")
        return ws["workspace_id"], r["root_pane"]["pane_id"], checkout

    def split(self, pane: str, step: SplitStep, cwd: str) -> str:
        """Split ``pane`` per ``step``; return the new pane's id."""
        return self._call(pane_split_argv(pane, step, cwd))["pane"]["pane_id"]

    def run_in_pane(self, pane: str, command: str) -> None:
        """Type ``command`` + Enter into ``pane``'s shell."""
        self._call(pane_run_argv(pane, command))

    def agents_in_workspace(self, ws_id: str) -> List[dict]:
        """Recognized agents currently living in ``ws_id``."""
        agents = self._call(["agent", "list"]).get("agents", [])
        return [a for a in agents if a.get("workspace_id") == ws_id]

    def wait_for_output(self, pane: str, regex: str, timeout_ms: int) -> None:
        """Best-effort wait for matching pane output; never fatal (boot probe)."""
        try:
            self._call(
                ["pane", "wait-output", pane, "--regex", regex, "--timeout", str(timeout_ms)]
            )
        except HerdrError:
            pass

    # Strings for previews and reports (no herdr call involved).
    @staticmethod
    def preview_space(cwd: str, label: str) -> List[str]:
        return [shlex.join(["herdr", *workspace_create_argv(cwd, label)])]

    @staticmethod
    def preview_worktree(
        repo: str, branch: str, base: str, label: str, open_existing: bool
    ) -> List[str]:
        argv = (
            worktree_open_argv(repo, branch)
            if open_existing
            else worktree_create_argv(repo, branch, base, label)
        )
        return [shlex.join(["herdr", *argv])]

    @staticmethod
    def split_line(ref: str, step: SplitStep, cwd: str) -> str:
        return "herdr " + " ".join(pane_split_argv(ref, step, cwd))

    @staticmethod
    def focus_hint(ws_id: str) -> str:
        return f"herdr workspace focus {ws_id}"

    @staticmethod
    def close_command(ws_id: str) -> str:
        return f"herdr workspace close {ws_id}"

    def open_teardown(self, ws_id: str) -> str:
        return self.close_command(ws_id)

    @staticmethod
    def worktree_teardown(ws_id: str, repo: str, branch: str, checkout: str) -> str:
        return (
            f"herdr worktree remove --workspace {ws_id}   "
            f"# then: git -C {repo} branch -D {branch}"
        )


def herdr_available(which: Optional[Callable[[str], Optional[str]]] = None) -> bool:
    resolve = which or shutil.which  # resolve at call time so tests can patch it
    return resolve("herdr") is not None


def require_herdr(env: Mapping[str, str]) -> None:
    """Guard: real control needs a live Herdr session and the binary on PATH."""
    if env.get("HERDR_ENV") != "1":
        raise typer.BadParameter(
            "not inside a Herdr session (HERDR_ENV != 1); open a herdr pane, "
            "or pass --dry-run to preview."
        )
    if not herdr_available():
        raise typer.BadParameter("herdr not found on PATH")


# --- Rex: the same operations over the `rex` CLI -------------------------
# Rex nests sessions > windows (tabs) > blocks. Inside a git repo the team is
# filed by git: the Rex session is named after the repo (shared by every team
# on it), the team's window after its worktree (the checkout directory's name),
# and each role gets a block with Rex's default name. Outside a git repo a team
# gets a session of its own. Rex has no worktree command, so worktrees are made
# with git directly.
REX_WINDOW = "team"
REX_DIRECTION = {"right": "right", "down": "below"}


def rex_new_argv(cwd: str, label: str, window: str = REX_WINDOW) -> List[str]:
    return ["new", label, "--cwd", cwd, "--window", window, "--json"]


def rex_window_argv(session: str, cwd: str, label: str = REX_WINDOW) -> List[str]:
    return [
        "window", "new", "-s", session, label, "--cwd", cwd, "--focus=false", "--json",
    ]


def rex_split_argv(session: str, block: str, step: SplitStep, cwd: str) -> List[str]:
    argv = ["split", "-s", session, "-b", block, "--split", REX_DIRECTION[step.direction]]
    if step.ratio is not None:
        # SplitStep.ratio is the share the original keeps; rex's --ratio is the
        # percent the *new* block gets.
        argv += ["--ratio", str(round((1 - step.ratio) * 100))]
    argv += ["--cwd", cwd, "--focus=false", "--json"]
    return argv


def rex_send_argv(session: str, block: str, command: str) -> List[str]:
    # `rex send` types text verbatim; the trailing CR is the Enter key.
    return ["send", "-s", session, "-b", block, command + "\r"]


def worktree_checkout_path(repo: str, branch: str) -> str:
    """Where a new worktree goes: ``<repo>.worktrees/<branch>`` beside the repo,
    with ``/`` in the branch flattened so it stays one directory deep."""
    root = Path(repo).resolve()
    return str(root.parent / f"{root.name}.worktrees" / branch.replace("/", "-"))


def git_worktree_add_argv(repo: str, branch: str, base: str, path: str) -> List[str]:
    return ["-C", repo, "worktree", "add", "-b", branch, path, base]


def find_worktree(porcelain: str, branch: str) -> Optional[str]:
    """The checkout path for ``branch`` in ``git worktree list --porcelain``."""
    path = None
    for line in porcelain.splitlines():
        if line.startswith("worktree "):
            path = line[len("worktree "):]
        elif line == f"branch refs/heads/{branch}":
            return path
    return None


Runner = Callable[[List[str]], "subprocess.CompletedProcess"]


def default_rex_runner(args: List[str]) -> "subprocess.CompletedProcess":
    """Run ``rex <args>`` and capture its output (the injectable subsystem)."""
    return subprocess.run(["rex", *args], capture_output=True, text=True)


def default_git_runner(args: List[str]) -> "subprocess.CompletedProcess":
    return subprocess.run(["git", *args], capture_output=True, text=True)


def git_location(path: str, git: Optional[Runner] = None) -> Optional[Tuple[str, str]]:
    """``(repo name, worktree name)`` for ``path``, or ``None`` outside a git
    working tree.

    The repo name comes from the *common* git dir, so a linked worktree reports
    its main repo (``app``); the worktree name is its own checkout directory
    (``me-br`` for ``app.worktrees/me-br``, ``app`` for the main checkout)."""
    run = git or default_git_runner  # resolve at call time so tests can patch it
    try:
        proc = run([
            "-C", path, "rev-parse", "--path-format=absolute",
            "--git-common-dir", "--show-toplevel",
        ])
    except OSError:
        return None
    lines = proc.stdout.split("\n") if proc.returncode == 0 else []
    if len(lines) < 2 or not lines[0].strip() or not lines[1].strip():
        return None
    common, toplevel = Path(lines[0].strip()), Path(lines[1].strip())
    repo = common.parent.name if common.name == ".git" else common.name
    return repo, toplevel.name


class RexError(BackendError):
    """A ``rex`` (or supporting ``git``) call failed or returned bad output."""


class Rex:
    """Facade over the ``rex`` CLI, mirroring :class:`Herdr` operation for
    operation so the commands and :func:`place_team` drive either one.

    Rex addresses a block within a session, so the facade remembers the session
    and window the team lives in. ``runner`` / ``git`` / ``sleep`` / ``clock``
    are injected so tests run without a Rex server, a repo, or real time.
    """

    name = "rex"
    noun = "session"
    worktree_unit = "window"  # what one worktree maps to

    def __init__(
        self,
        runner: Optional[Runner] = None,
        git: Optional[Runner] = None,
        sleep: Optional[Callable[[float], None]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        # Resolve the module defaults at call time so tests can monkeypatch them.
        self._run = runner or default_rex_runner
        self._git = git or default_git_runner
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self.session = ""
        self.window = ""
        self.window_label = REX_WINDOW
        # A team in a shared repo session is closed by its window, never by
        # killing the session (which would take the repo's other teams too).
        self.shared_session = False

    def _text(self, tool: str, run: Runner, args: List[str]) -> str:
        proc = run(args)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RexError(f"`{tool} {shlex.join(args)}` failed (rc={proc.returncode}): {detail}")
        return proc.stdout or ""

    def _call(self, args: List[str]) -> dict:
        """Run one rex command; return its JSON output, or ``{}`` if it printed
        nothing (``send`` is silent on success)."""
        out = self._text("rex", self._run, args)
        if not out.strip():
            return {}
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            raise RexError(f"`rex {shlex.join(args)}`: non-JSON output: {out[:200]!r}")
        return data if isinstance(data, dict) else {}

    def create_workspace(self, cwd: str, label: str) -> Tuple[str, str]:
        """Make a window for a team; return ``(session_id, root_block_id)``.

        In a git repo the window is named after ``cwd``'s worktree, inside the
        repo's session (created on first use, reused after); ``label`` is not
        used. Outside one, the team gets a session of its own named ``label``."""
        where = git_location(cwd, self._git)
        if where is None:
            return self._new_session(cwd, label, REX_WINDOW)
        repo, worktree = where
        self.shared_session = True
        session = self.find_session(repo)
        if session is None:
            return self._new_session(cwd, repo, worktree)
        self.session = session
        return session, self.add_window(cwd, worktree)

    def _new_session(self, cwd: str, label: str, window: str) -> Tuple[str, str]:
        r = self._call(rex_new_argv(cwd, label, window))
        first = r["initial_windows"][0]
        self.session, self.window = r["session_id"], first["window_id"]
        self.window_label = window
        return self.session, first["block_ids"][0]

    def create_worktree(
        self, repo: str, branch: str, base: str, label: str
    ) -> Tuple[str, str, str]:
        """``git worktree add`` a new branch, then open a session in it; return
        ``(session_id, root_block_id, checkout_path)``."""
        checkout = worktree_checkout_path(repo, branch)
        self._text("git", self._git, git_worktree_add_argv(repo, branch, base, checkout))
        session, root = self.create_workspace(checkout, label)
        return session, root, checkout

    def open_worktree(self, repo: str, branch: str) -> Tuple[str, Optional[str], str]:
        """Find the EXISTING checkout of ``branch`` and its window (named after
        the worktree, in the repo's session); return
        ``(session_id, root_block_id, checkout_path)``.

        Like a herdr worktree, the worktree maps to one window: if it is already
        open, that window is the one checked for a running team and the root is
        ``None`` — nothing is created until the caller has run that check, and
        then :meth:`add_window` gives the new team a window of its own.
        """
        listing = self._text("git", self._git, ["-C", repo, "worktree", "list", "--porcelain"])
        checkout = find_worktree(listing, branch)
        if checkout is None:
            raise RexError(f"no worktree of {repo} has branch {branch} checked out")
        where = git_location(checkout, self._git)
        session = self.find_session(where[0]) if where else None
        window = self.find_window(session, where[1]) if session else None
        if window is None:
            session, root = self.create_workspace(checkout, branch)
            return session, root, checkout
        self.session, self.window, self.window_label = session, window, where[1]
        self.shared_session = True
        return session, None, checkout

    def find_session(self, label: str) -> Optional[str]:
        """The id of the first session labelled ``label``, if any."""
        for s in self._call(["ls", "--json"]).get("sessions", []):
            if s.get("label") == label:
                return s["session_id"]
        return None

    def find_window(self, session: str, label: str) -> Optional[str]:
        """The id of the first window in ``session`` labelled ``label``, if any."""
        for w in self._call(["window", "ls", "-s", session, "--json"]).get("windows", []):
            if w.get("label") == label:
                return w["window_id"]
        return None

    def add_window(self, cwd: str, label: Optional[str] = None) -> str:
        """Open a new (unfocused) window in the current session; return its block.

        ``label`` defaults to the label of the window this team was matched to."""
        self.window_label = label or self.window_label
        r = self._call(rex_window_argv(self.session, cwd, self.window_label))
        self.window = r["window_id"]
        return r["block_ids"][0]

    def split(self, pane: str, step: SplitStep, cwd: str) -> str:
        """Split block ``pane`` per ``step``; return the new block's id."""
        return self._call(rex_split_argv(self.session, pane, step, cwd))["block_ids"][0]

    def run_in_pane(self, pane: str, command: str) -> None:
        """Type ``command`` + Enter into block ``pane``'s shell."""
        self._call(rex_send_argv(self.session, pane, command))

    def agents_in_workspace(self, ws_id: str) -> List[dict]:
        """The blocks in this team's window of ``ws_id`` (the whole session if
        no window is known yet).

        Rex does not recognize agents itself, so any terminal in a worktree's
        window counts as its team: the window exists only to hold one. Scoping
        to the window keeps the repo's other worktrees out of the count."""
        blocks = self._call(["block", "ls", "-s", ws_id, "--json"]).get("blocks", [])
        return [
            {"pane_id": b["block_id"]}
            for b in blocks
            if not self.window or b.get("window_id") == self.window
        ]

    def wait_for_output(self, pane: str, regex: str, timeout_ms: int) -> None:
        """Best-effort poll of the block's screen for ``regex``; never fatal."""
        pattern = re.compile(regex)
        deadline = self._clock() + timeout_ms / 1000
        while True:
            try:
                screen = self._text(
                    "rex", self._run, ["capture", "-s", self.session, "-b", pane, "--trim"]
                )
            except RexError:
                return
            if pattern.search(screen) or self._clock() >= deadline:
                return
            self._sleep(0.5)

    # Strings for previews and reports. Previews make only read-only git calls
    # (to name the repo session); they never touch rex.
    @staticmethod
    def _preview_window(cwd: str, repo: Optional[str], window: str) -> List[str]:
        if repo is None:
            return [shlex.join(["rex", *rex_new_argv(cwd, window)])]
        return [
            f"# git repo {repo!r}: window {window!r} in session {repo!r}"
            " (rex window new, if that session is open), else:",
            shlex.join(["rex", *rex_new_argv(cwd, repo, window)]),
        ]

    @staticmethod
    def preview_space(cwd: str, label: str) -> List[str]:
        where = git_location(cwd)
        if where is None:
            return Rex._preview_window(cwd, None, label)
        return Rex._preview_window(cwd, *where)

    @staticmethod
    def preview_worktree(
        repo: str, branch: str, base: str, label: str, open_existing: bool
    ) -> List[str]:
        # Before the checkout exists, name the session from the repo itself.
        where = git_location(repo)
        repo_name = where[0] if where else Path(repo).resolve().name
        if open_existing:
            return [
                f"git -C {shlex.quote(repo)} worktree list --porcelain"
                f"   # find {branch}'s <checkout>",
                f"# reuse the <checkout>'s window in session {repo_name!r} if open, else:",
                *Rex._preview_window("<checkout>", repo_name, "<checkout name>")[1:],
            ]
        checkout = worktree_checkout_path(repo, branch)
        return [
            shlex.join(["git", *git_worktree_add_argv(repo, branch, base, checkout)]),
            *Rex._preview_window(checkout, repo_name, Path(checkout).name),
        ]

    @staticmethod
    def split_line(ref: str, step: SplitStep, cwd: str) -> str:
        return "rex " + " ".join(rex_split_argv("<session>", ref, step, cwd))

    def focus_hint(self, ws_id: str) -> str:
        return f"rex focus -s {ws_id} {self.window}"

    def close_command(self, ws_id: str) -> str:
        if self.shared_session:
            return f"rex window close -s {ws_id} {self.window}"
        return f"rex kill {ws_id}"

    def open_teardown(self, ws_id: str) -> str:
        return self.close_command(ws_id)

    def worktree_teardown(self, ws_id: str, repo: str, branch: str, checkout: str) -> str:
        return (
            f"{self.close_command(ws_id)}   # then: git -C {repo} worktree remove {checkout}"
            f" && git -C {repo} branch -D {branch}"
        )


def rex_available(which: Optional[Callable[[str], Optional[str]]] = None) -> bool:
    resolve = which or shutil.which  # resolve at call time so tests can patch it
    return resolve("rex") is not None


def require_rex() -> None:
    """Guard: rex reaches its server from any terminal, so only the binary is needed."""
    if not rex_available():
        raise typer.BadParameter("rex not found on PATH")


# --- Backend selection ---------------------------------------------------
BACKENDS = {"herdr": Herdr, "rex": Rex}


def resolve_backend(choice: str, env: Mapping[str, str]) -> str:
    """Resolve ``--backend``. ``auto`` picks the multiplexer we are running in:
    herdr if ``HERDR_ENV=1``, rex if ``REX_SESSION`` is set, else herdr."""
    if choice == "auto":
        if env.get("HERDR_ENV") == "1":
            return "herdr"
        return "rex" if env.get("REX_SESSION") else "herdr"
    if choice not in BACKENDS:
        raise typer.BadParameter(
            f"unknown backend {choice!r}; choose from auto, {', '.join(BACKENDS)}"
        )
    return choice


def open_backend(name: str):
    """Check the backend is usable here, then return a live facade for it."""
    if name == "rex":
        require_rex()
        return Rex()
    require_herdr(os.environ)
    return Herdr()


# --- Director: split panes, launch a role in each ------------------------
def place_team(
    mux,
    layout: LayoutStrategy,
    root_pane: str,
    cwd: str,
    roles: List[Role],
    topic: str,
    suffix: Optional[str],
    invocation: List[str],
    prompts: Optional[Mapping[Role, str]] = None,
    models: Optional[Mapping[Role, str]] = None,
) -> List[Tuple[Role, str]]:
    """Grow ``root_pane`` into the layout, then launch one role per pane.

    ``mux`` is a :class:`Herdr` or :class:`Rex` facade. Roles arrive in
    canonical order (boss first), and pane 0 is the root, so the boss is always
    launched first — the dev/qa "check in with boss" prompt lands on a live
    session. ``prompts`` / ``models`` override a role's opening prompt / model
    by role.
    """
    prompts = prompts or {}
    models = models or {}
    panes = [root_pane]
    for step in layout.plan(len(roles)):
        panes.append(mux.split(panes[step.source], step, cwd))
    placements = list(zip(roles, panes))
    for role, pane in placements:
        cmd = pane_launch_command(
            invocation, role, topic, suffix, prompts.get(role), models.get(role)
        )
        mux.run_in_pane(pane, cmd)
    return placements


def preview_placement(
    layout: LayoutStrategy,
    roles: List[Role],
    topic: str,
    suffix: Optional[str],
    invocation: List[str],
    cwd: str,
    prompts: Optional[Mapping[Role, str]] = None,
    models: Optional[Mapping[Role, str]] = None,
    split_line: Callable[[str, SplitStep, str], str] = Herdr.split_line,
) -> List[str]:
    """Human-readable dry-run of the splits + per-pane launches (no pane ids yet)."""
    prompts = prompts or {}
    models = models or {}
    tokens = [role_token(r) for r in roles]
    refs = [f"<{tokens[0]}>"]
    lines = [f"# layout: {layout.name}, {len(roles)} pane(s)"]
    for idx, step in enumerate(layout.plan(len(roles)), start=1):
        lines.append(split_line(refs[step.source], step, cwd))
        refs.append(f"<{tokens[idx]}>")
    lines.append("# per-pane launch (foreground):")
    for role in roles:
        cmd = pane_launch_command(
            invocation, role, topic, suffix, prompts.get(role), models.get(role)
        )
        lines.append(f"[{role_token(role)}] {cmd}")
    return lines


def report_team(
    mux,
    ws_id: str,
    placements: List[Tuple[Role, str]],
    wait: bool,
    teardown: str,
    extra: Optional[str] = None,
) -> None:
    """Print the created workspace, pane→role map, live status, and next steps."""
    if wait and placements:
        mux.wait_for_output(placements[0][1], BOOT_REGEX, 20000)
    statuses: dict = {}
    if wait:
        try:
            # Only herdr reports an agent status; rex blocks carry none.
            statuses = {
                a["pane_id"]: a["agent_status"]
                for a in mux.agents_in_workspace(ws_id)
                if "agent_status" in a
            }
        except BackendError:
            statuses = {}

    typer.echo(f"✓ {mux.noun} {ws_id}")
    if extra:
        typer.echo(f"  {extra}")
    for role, pane in placements:
        stag = f"  [{statuses[pane]}]" if pane in statuses else ""
        typer.echo(f"  {role.label:9} {pane}{stag}")
    typer.echo(f"  focus:    {mux.focus_hint(ws_id)}")
    typer.echo(f"  teardown: {teardown}")


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
PromptOpt = typer.Option(
    None,
    "--prompt",
    "-p",
    help="Opening prompt for the session (overrides the role default), e.g. a "
    '"/resume_handoff <path>" slash command.',
)



def model_option(default: str, flag: str = "--model", role: str = "the session"):
    """A model-override option for ``role``, defaulting to its pinned model."""
    return typer.Option(default, flag, help=f"Model for {role} (default: {default}).")


BossModelOpt = model_option(FABLE, "--boss-model", "the boss")
DevModelOpt = model_option(OPUS_55, "--dev-model", "the developer")
QaModelOpt = model_option(OPUS_55, "--qa-model", "QA")


def role_models(boss_model: str, dev_model: str, qa_model: str) -> dict:
    """Map each role to the model it should launch with."""
    return {BOSS: boss_model, DEVELOPER: dev_model, QA: qa_model}


@app.command()
def boss(
    topic: str = TopicArg,
    suffix: Optional[str] = SuffixOpt,
    background: Optional[bool] = BgOpt,
    prompt: Optional[str] = PromptOpt,
    model: str = model_option(FABLE),
    dry_run: bool = DryRunOpt,
) -> None:
    """Launch the Boss agent (Fable by default; no opening prompt unless --prompt given)."""
    launch(BOSS, topic, suffix, background, dry_run, prompt, model)


@app.command()
def dev(
    topic: str = TopicArg,
    suffix: Optional[str] = SuffixOpt,
    background: Optional[bool] = BgOpt,
    prompt: Optional[str] = PromptOpt,
    model: str = model_option(OPUS_55),
    dry_run: bool = DryRunOpt,
) -> None:
    """Launch the Developer agent (Opus 5.5 by default, opens by checking in with boss)."""
    launch(DEVELOPER, topic, suffix, background, dry_run, prompt, model)


@app.command()
def qa(
    topic: str = TopicArg,
    suffix: Optional[str] = SuffixOpt,
    background: Optional[bool] = BgOpt,
    prompt: Optional[str] = PromptOpt,
    model: str = model_option(OPUS_55),
    dry_run: bool = DryRunOpt,
) -> None:
    """Launch the QA agent (Opus 5.5 by default, opens by checking in with boss)."""
    launch(QA, topic, suffix, background, dry_run, prompt, model)


@app.command()
def team(
    topic: str = TopicArg,
    suffix: Optional[str] = SuffixOpt,
    roles: str = typer.Option(
        "boss,dev,qa",
        "--roles",
        help="Comma-separated subset of the team to launch, e.g. boss,qa.",
    ),
    boss_model: str = BossModelOpt,
    dev_model: str = DevModelOpt,
    qa_model: str = QaModelOpt,
    dry_run: bool = DryRunOpt,
) -> None:
    """Launch the whole team on TOPIC — each role backgrounded and attachable."""
    models = role_models(boss_model, dev_model, qa_model)
    plans = [
        LaunchPlan(role, topic, suffix, background=True, model=models[role])
        for role in parse_roles(roles)
    ]
    launch_team(plans, dry_run)


# --- Herdr commands: a team laid out in panes ----------------------------
RolesOpt = typer.Option(
    "boss,dev,qa", "--roles", help="Comma-separated subset of the team, e.g. boss,qa."
)
LayoutOpt = typer.Option(
    DEFAULT_LAYOUT, "--layout", help=f"Pane layout. Choices: {', '.join(LAYOUTS)}."
)
CwdOpt = typer.Option(None, "--cwd", help="Working dir for the panes (default: current dir).")
LabelOpt = typer.Option(
    None,
    "--label",
    help="Workspace label (default: the topic / branch). Not used by rex in a "
    "git repo, where the window takes the worktree's name.",
)
WaitOpt = typer.Option(
    True, "--wait/--no-wait", help="Wait for the boss to boot, then report agent status."
)
BackendOpt = typer.Option(
    "auto",
    "--backend",
    help="Terminal multiplexer: herdr, rex, or auto (the one this terminal runs "
    "in; herdr if neither is detected).",
)
BossPromptOpt = typer.Option(
    None,
    "--boss-prompt",
    help="Opening prompt for the boss pane, e.g. a \"/resume_handoff <path>\" "
    "slash command (the boss otherwise opens with no prompt).",
)
DevPromptOpt = typer.Option(
    None,
    "--dev-prompt",
    help="Opening prompt for the dev pane, overriding its 'check in with boss' "
    "default (e.g. a \"/resume_handoff <path>\" slash command).",
)
QaPromptOpt = typer.Option(
    None,
    "--qa-prompt",
    help="Opening prompt for the qa pane, overriding its 'check in with boss' "
    "default (e.g. a \"/resume_handoff <path>\" slash command).",
)


def role_prompts(
    selected: List[Role],
    boss_prompt: Optional[str] = None,
    dev_prompt: Optional[str] = None,
    qa_prompt: Optional[str] = None,
) -> dict:
    """Build the per-role opening-prompt overrides for a paned team.

    Each ``*_prompt`` overrides that role's default opening prompt (boss: none;
    dev/qa: "Check in with boss"). A prompt given for a role that is not in
    ``selected`` is a no-op with a non-fatal warning, so a ``--roles`` typo never
    silently swallows a prompt.
    """
    prompts: dict = {}
    for role, prompt in ((BOSS, boss_prompt), (DEVELOPER, dev_prompt), (QA, qa_prompt)):
        if not prompt:
            continue
        if role not in selected:
            token = role_token(role)
            typer.echo(
                f"warning: --{token}-prompt given but '{token}' is not in --roles",
                err=True,
            )
            continue
        prompts[role] = prompt
    return prompts


@app.command()
def space(
    topic: str = TopicArg,
    roles: str = RolesOpt,
    layout: str = LayoutOpt,
    cwd: Optional[str] = CwdOpt,
    label: Optional[str] = LabelOpt,
    suffix: Optional[str] = SuffixOpt,
    boss_prompt: Optional[str] = BossPromptOpt,
    dev_prompt: Optional[str] = DevPromptOpt,
    qa_prompt: Optional[str] = QaPromptOpt,
    boss_model: str = BossModelOpt,
    dev_model: str = DevModelOpt,
    qa_model: str = QaModelOpt,
    backend: str = BackendOpt,
    wait: bool = WaitOpt,
    dry_run: bool = DryRunOpt,
) -> None:
    """Create a Herdr workspace / Rex session (one tab) and launch a role per pane in it."""
    selected = parse_roles(roles)
    strategy = get_layout(layout)
    kind = BACKENDS[resolve_backend(backend, os.environ)]
    work_cwd = cwd or os.getcwd()
    ws_label = (label or topic).strip()
    invocation = claude_team_invocation()
    prompts = role_prompts(selected, boss_prompt, dev_prompt, qa_prompt)
    models = role_models(boss_model, dev_model, qa_model)

    if dry_run:
        for line in kind.preview_space(work_cwd, ws_label):
            typer.echo(line)
        for line in preview_placement(
            strategy, selected, topic, suffix, invocation, work_cwd, prompts, models,
            split_line=kind.split_line,
        ):
            typer.echo(line)
        raise typer.Exit()

    mux = open_backend(kind.name)
    try:
        ws_id, root_pane = mux.create_workspace(work_cwd, ws_label)
        placements = place_team(
            mux, strategy, root_pane, work_cwd, selected, topic, suffix, invocation, prompts, models
        )
    except BackendError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    report_team(mux, ws_id, placements, wait, teardown=mux.close_command(ws_id))


@app.command()
def worktree(
    branch: str = typer.Argument(
        ..., metavar="BRANCH", help="New branch / worktree name (e.g. me/proj-123-...)."
    ),
    repo: Optional[str] = typer.Option(
        None, "--repo", help="Repo path to branch from (default: current dir)."
    ),
    base: str = typer.Option("origin/dev", "--base", help="Base ref for the new branch (ignored with --open)."),
    open_existing: bool = typer.Option(
        False,
        "--open",
        help="Open an EXISTING worktree for BRANCH (herdr worktree open, or its "
        "rex session) and nest the team under it, instead of creating a new "
        "branch. --base and --label are ignored (the worktree keeps its own label).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="With --open, launch even if the worktree's workspace already holds "
        "agents (its team). Default: refuse, to avoid clobbering a running team.",
    ),
    topic: Optional[str] = typer.Option(
        None, "--topic", help="Team topic / session-name stem (default: the branch name)."
    ),
    roles: str = RolesOpt,
    layout: str = LayoutOpt,
    label: Optional[str] = LabelOpt,
    suffix: Optional[str] = SuffixOpt,
    boss_prompt: Optional[str] = BossPromptOpt,
    dev_prompt: Optional[str] = DevPromptOpt,
    qa_prompt: Optional[str] = QaPromptOpt,
    boss_model: str = BossModelOpt,
    dev_model: str = DevModelOpt,
    qa_model: str = QaModelOpt,
    backend: str = BackendOpt,
    wait: bool = WaitOpt,
    dry_run: bool = DryRunOpt,
) -> None:
    """Create (or, with --open, reuse an existing) git worktree and launch a team on it.

    A git worktree maps to a single Herdr workspace / Rex session, so --open on a
    worktree that already has a team reuses it; the guard refuses to add a second
    team into it unless --force is given.
    """
    if not branch.strip():
        raise typer.BadParameter("BRANCH must not be empty.")
    selected = parse_roles(roles)
    strategy = get_layout(layout)
    kind = BACKENDS[resolve_backend(backend, os.environ)]
    repo_path = repo or os.getcwd()
    team_topic = (topic or branch).strip()
    ws_label = (label or branch).strip()
    invocation = claude_team_invocation()
    prompts = role_prompts(selected, boss_prompt, dev_prompt, qa_prompt)
    models = role_models(boss_model, dev_model, qa_model)

    if dry_run:
        for line in kind.preview_worktree(repo_path, branch, base, ws_label, open_existing):
            typer.echo(line)
        for line in preview_placement(
            strategy, selected, team_topic, suffix, invocation, "<checkout>", prompts, models,
            split_line=kind.split_line,
        ):
            typer.echo(line)
        raise typer.Exit()

    mux = open_backend(kind.name)
    try:
        if open_existing:
            ws_id, root_pane, checkout = mux.open_worktree(repo_path, branch)
            if not force:
                existing = mux.agents_in_workspace(ws_id)
                if existing:
                    typer.echo(
                        f"error: worktree {branch} is already open in {mux.noun} {ws_id} "
                        f"with {len(existing)} agent(s). A worktree maps to one {mux.worktree_unit}, "
                        f"so this would add a team into the running one. Close it first "
                        f"({mux.close_command(ws_id)}) or pass --force.",
                        err=True,
                    )
                    raise typer.Exit(1)
            if root_pane is None:
                # Rex reused the worktree's open session: give this team a window.
                root_pane = mux.add_window(checkout or repo_path)
        else:
            ws_id, root_pane, checkout = mux.create_worktree(repo_path, branch, base, ws_label)
        placements = place_team(
            mux, strategy, root_pane, checkout or repo_path, selected, team_topic, suffix, invocation, prompts, models
        )
    except BackendError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
    if open_existing:
        # We opened a pre-existing worktree; teardown only closes the workspace,
        # it must not remove the checkout or delete the branch.
        teardown = mux.open_teardown(ws_id)
        extra = f"opened existing worktree {branch}  ->  {checkout or '(worktree)'}"
    else:
        teardown = mux.worktree_teardown(ws_id, repo_path, branch, checkout)
        extra = f"branch {branch} @ {base}  ->  {checkout or '(worktree)'}"
    report_team(mux, ws_id, placements, wait, teardown=teardown, extra=extra)


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
        "| dev     | Developer: <topic> | claude-opus-5-5   | Check in with boss  |",
        "| qa      | QA: <topic>        | claude-opus-5-5   | Check in with boss  |",
        "",
        "Override a model with `--model` on boss/dev/qa, or `--boss-model` /"
        " `--dev-model` / `--qa-model` on team, space and worktree.",
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
        "In a Herdr pane or a Rex terminal (auto-detected; force with --backend"
        " herdr|rex), lay the team out in panes and give each role its own"
        " opening prompt (omit a flag to keep that role's default — boss: none,"
        ' dev/qa: "Check in with boss"):',
        "```",
        "claude-team space PROJ-123 \\",
        "  --boss-prompt '/resume_handoff boss.md' \\",
        "  --dev-prompt  '/resume_handoff dev.md' \\",
        "  --qa-prompt   '/resume_handoff qa.md'",
        "# fresh git worktree:    claude-team worktree <branch> ...",
        "# existing worktree:     claude-team worktree <branch> --repo <root> --open  (nests under the repo)",
        "```",
        "",
        "dev/qa still check in with the boss even with a custom prompt: the check-in"
        " rides along as --append-system-prompt so a leading slash command stays intact.",
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
