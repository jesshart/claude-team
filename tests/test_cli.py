"""Behavior tests for the claude-team CLI."""

from __future__ import annotations

import json
import subprocess

import pytest
from typer.testing import CliRunner

from claude_team import cli
from claude_team.cli import (
    BOSS,
    CHECK_IN_PROMPT,
    ColumnsLayout,
    DEVELOPER,
    FABLE,
    Herdr,
    HerdrError,
    LaunchPlan,
    LeftStackLayout,
    OPUS_48,
    QA,
    SplitStep,
    app,
    boss_prompts,
    build_command,
    build_name,
    claude_team_invocation,
    classify_team,
    decide_background,
    detect_state,
    extract_session_id,
    get_layout,
    pane_launch_command,
    pane_run_argv,
    pane_split_argv,
    parse_roles,
    place_team,
    role_token,
    workspace_create_argv,
    worktree_create_argv,
)

runner = CliRunner()


# --- Fake herdr subsystem (drives the Herdr facade in tests) -------------
class FakeHerdrRunner:
    """Records `herdr <args>` calls and returns canned JSON, no live server."""

    def __init__(self):
        self.calls = []
        self._pane_seq = 1

    def __call__(self, args):
        self.calls.append(args)
        return subprocess.CompletedProcess(
            args, 0, stdout=json.dumps({"id": "x", "result": self._result(args)}), stderr=""
        )

    def _result(self, args):
        head = args[:2]
        if head == ["workspace", "create"]:
            return {"workspace": {"workspace_id": "w1"}, "tab": {"tab_id": "w1:t1"},
                    "root_pane": {"pane_id": "w1:p1"}}
        if head == ["worktree", "create"]:
            return {"workspace": {"workspace_id": "w2", "worktree": {"checkout_path": "/co"}},
                    "root_pane": {"pane_id": "w2:p1"}}
        if head == ["pane", "split"]:
            self._pane_seq += 1
            return {"pane": {"pane_id": f"w1:p{self._pane_seq}"}}
        if head == ["agent", "list"]:
            return {"agents": []}
        return {"type": "ok"}

    def pane_runs(self):
        """(pane_id, command) for each `pane run`, in call order."""
        return [(c[2], c[3]) for c in self.calls if c[:2] == ["pane", "run"]]


# --- build_name ----------------------------------------------------------
def test_build_name_basic():
    assert build_name(BOSS, "PROJ-123", None) == "Boss: PROJ-123"


def test_build_name_with_suffix():
    assert build_name(DEVELOPER, "PROJ-123", "Handoff 1") == "Developer: PROJ-123 Handoff 1"


def test_build_name_trims_whitespace():
    assert build_name(QA, "  PROJ-123  ", "  Handoff 2 ") == "QA: PROJ-123 Handoff 2"


def test_build_name_empty_suffix_is_ignored():
    assert build_name(BOSS, "topic", "   ") == "Boss: topic"


def test_build_name_custom_topic():
    assert build_name(QA, "refactor pull-config", None) == "QA: refactor pull-config"


# --- role definitions ----------------------------------------------------
def test_role_models_and_prompts():
    assert BOSS.model == FABLE and BOSS.prompt is None
    assert DEVELOPER.model == OPUS_48 and DEVELOPER.prompt == CHECK_IN_PROMPT
    assert QA.model == OPUS_48 and QA.prompt == CHECK_IN_PROMPT


# --- decide_background ---------------------------------------------------
def test_decide_background_override_wins_over_environment():
    # Even in a plain terminal, --bg forces background; --fg forces foreground.
    assert decide_background(True, {}, isatty=True) is True
    assert decide_background(False, {"CLAUDECODE": "1"}, isatty=False) is False


def test_decide_background_auto_foreground_in_terminal():
    assert decide_background(None, {}, isatty=True) is False


def test_decide_background_auto_background_without_tty():
    assert decide_background(None, {}, isatty=False) is True


def test_decide_background_auto_background_inside_claude_session():
    # CLAUDECODE set (agent Bash tool) => background even if a tty is present.
    assert decide_background(None, {"CLAUDECODE": "1"}, isatty=True) is True


# --- build_command -------------------------------------------------------
def test_build_command_foreground_has_no_bg_flag():
    cmd = build_command(BOSS, "PROJ-123", None, background=False)
    assert cmd == ["claude", "--name", "Boss: PROJ-123", "--model", "fable"]


def test_build_command_background_inserts_bg_flag():
    cmd = build_command(BOSS, "PROJ-123", None, background=True)
    assert cmd == ["claude", "--bg", "--name", "Boss: PROJ-123", "--model", "fable"]


def test_build_command_dev_appends_prompt():
    cmd = build_command(DEVELOPER, "PROJ-123", "Handoff 1", background=False)
    assert cmd == [
        "claude",
        "--name",
        "Developer: PROJ-123 Handoff 1",
        "--model",
        "claude-opus-4-8",
        "Check in with boss",
    ]


# --- extract_session_id --------------------------------------------------
# Captured verbatim from a real `claude --bg` run.
REAL_BG_OUTPUT = (
    "backgrounded · 178ae072 · ctlaunch-probe (idle — send a prompt to start)\n"
    "  claude agents             list sessions\n"
    "  claude attach 178ae072    open in this terminal\n"
)


def test_extract_session_id_from_real_bg_output():
    assert extract_session_id(REAL_BG_OUTPUT) == "178ae072"


def test_extract_session_id_ignores_hex_in_name():
    # A hex-looking name must not shadow the real id (id precedes the name).
    out = "backgrounded · 9f75fa16 · Boss: deadbeef99 (idle)\n"
    assert extract_session_id(out) == "9f75fa16"


def test_extract_session_id_strips_ansi():
    assert extract_session_id("\x1b[32m20e4e79f\x1b[0m") == "20e4e79f"


def test_extract_session_id_absent():
    assert extract_session_id("no id here") is None


# --- CLI: dry-run --------------------------------------------------------
@pytest.fixture
def terminal_env(monkeypatch):
    """Simulate a human at a real terminal (foreground auto-default)."""
    monkeypatch.delenv("CLAUDECODE", raising=False)
    monkeypatch.setattr(cli, "stdout_isatty", lambda: True)


def test_dry_run_boss(terminal_env):
    result = runner.invoke(app, ["boss", "PROJ-123", "--dry-run"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "claude --name 'Boss: PROJ-123' --model fable"


def test_dry_run_qa_with_prompt(terminal_env):
    result = runner.invoke(app, ["qa", "PROJ-123", "--dry-run"])
    assert result.exit_code == 0
    assert result.stdout.strip() == (
        "claude --name 'QA: PROJ-123' --model claude-opus-4-8 'Check in with boss'"
    )


def test_dry_run_bg_override_shows_bg_flag(terminal_env):
    result = runner.invoke(app, ["dev", "PROJ-123", "--bg", "--dry-run"])
    assert result.exit_code == 0
    assert result.stdout.strip().startswith("claude --bg --name 'Developer: PROJ-123'")


# --- pressure: hostile input ---------------------------------------------
@pytest.mark.parametrize("topic", ["", "   ", "\t", "\n"])
def test_empty_topic_rejected(terminal_env, topic):
    result = runner.invoke(app, ["boss", topic, "--dry-run"])
    assert result.exit_code != 0
    assert "TOPIC must not be empty" in result.output


@pytest.mark.parametrize(
    "topic",
    ["; rm -rf ~ #", "$(whoami)`id`", 'it\'s "quoted"', "café-🚀", "a b | c && d"],
)
def test_shell_metacharacters_are_inert(terminal_env, topic):
    # No shell is ever invoked, so metachars survive as a literal name value and
    # the previewed command re-quotes them safely (round-trips through shlex).
    import shlex

    result = runner.invoke(app, ["boss", topic, "--dry-run"])
    assert result.exit_code == 0
    parsed = shlex.split(result.stdout.strip())
    assert parsed[: parsed.index("--name") + 2][-1] == f"Boss: {topic.strip()}"


def test_flag_like_topic_via_double_dash(terminal_env):
    # `--` stops option parsing so a topic that looks like a flag is accepted.
    result = runner.invoke(app, ["boss", "--dry-run", "--", "--bg"])
    assert result.exit_code == 0
    assert "--name 'Boss: --bg'" in result.stdout


def test_bg_fg_conflict_last_wins(terminal_env):
    bg_then_fg = runner.invoke(app, ["dev", "X", "--bg", "--fg", "--dry-run"])
    fg_then_bg = runner.invoke(app, ["dev", "X", "--fg", "--bg", "--dry-run"])
    assert " --bg " not in f" {bg_then_fg.stdout} "  # --fg won
    assert fg_then_bg.stdout.strip().startswith("claude --bg")  # --bg won


def test_unknown_command_errors(terminal_env):
    result = runner.invoke(app, ["intern", "X"])
    assert result.exit_code != 0


# --- LaunchPlan (Command pattern) ----------------------------------------
def test_launch_plan_is_inert_and_pure():
    plan = LaunchPlan(DEVELOPER, "PROJ-123", "Handoff 1", background=True)
    assert plan.name == "Developer: PROJ-123 Handoff 1"
    assert plan.argv() == [
        "claude",
        "--bg",
        "--name",
        "Developer: PROJ-123 Handoff 1",
        "--model",
        "claude-opus-4-8",
        "Check in with boss",
    ]
    assert plan.preview() == (
        "claude --bg --name 'Developer: PROJ-123 Handoff 1' "
        "--model claude-opus-4-8 'Check in with boss'"
    )


# --- parse_roles ---------------------------------------------------------
def test_parse_roles_default_all_three_canonical_order():
    assert parse_roles("boss,dev,qa") == [BOSS, DEVELOPER, QA]


def test_parse_roles_subset():
    assert parse_roles("qa,boss") == [BOSS, QA]  # reordered to canonical, boss first


def test_parse_roles_dedupes_and_accepts_developer_alias():
    assert parse_roles("developer,dev,dev") == [DEVELOPER]


def test_parse_roles_rejects_unknown():
    with pytest.raises(cli.typer.BadParameter):
        parse_roles("boss,intern")


def test_parse_roles_rejects_empty():
    with pytest.raises(cli.typer.BadParameter):
        parse_roles(" , ")


# --- team command --------------------------------------------------------
def test_team_dry_run_previews_all_three_backgrounded(terminal_env):
    result = runner.invoke(app, ["team", "PROJ-123", "--dry-run"])
    assert result.exit_code == 0
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 3
    assert all(ln.startswith("claude --bg --name ") for ln in lines)
    assert "Boss: PROJ-123" in lines[0]
    assert "Developer: PROJ-123" in lines[1]
    assert "QA: PROJ-123" in lines[2]


def test_team_dry_run_respects_roles_subset(terminal_env):
    result = runner.invoke(app, ["team", "PROJ-123", "--roles", "boss,qa", "--dry-run"])
    assert result.exit_code == 0
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert "Developer" not in result.stdout


def test_team_launches_each_and_prints_attach_lines(terminal_env, monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        sid = f"aaaa000{len(calls)}"
        stdout = f"backgrounded · {sid} · {argv[argv.index('--name') + 1]} (idle)\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    result = runner.invoke(app, ["team", "PROJ-123"])
    assert result.exit_code == 0
    assert len(calls) == 3
    assert all("--bg" in argv for argv in calls)
    assert "claude attach aaaa0001" in result.stdout
    assert "claude attach aaaa0003" in result.stdout
    assert result.stdout.count("✓") == 3


def test_team_reports_failure_and_exits_nonzero(terminal_env, monkeypatch):
    def fake_run(argv, **kwargs):
        # Boss succeeds, everyone else fails to launch.
        if "Boss: PROJ-123" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="backgrounded · dead0001 · Boss: PROJ-123\n", stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom\n")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    result = runner.invoke(app, ["team", "PROJ-123"])
    assert result.exit_code == 1
    assert "✓ Boss: PROJ-123" in result.stdout
    assert result.stdout.count("✗") == 2


def test_team_rejects_unknown_role(terminal_env):
    result = runner.invoke(app, ["team", "PROJ-123", "--roles", "boss,intern"])
    assert result.exit_code != 0


def test_dry_run_does_not_launch(terminal_env, monkeypatch):
    def boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("dry-run must not exec or spawn")

    monkeypatch.setattr(cli.os, "execvp", boom)
    monkeypatch.setattr(cli.subprocess, "run", boom)
    result = runner.invoke(app, ["boss", "PROJ-123", "--dry-run"])
    assert result.exit_code == 0


# --- CLI: real launch side effects ---------------------------------------
def test_foreground_launch_execs_expected_argv(terminal_env, monkeypatch):
    captured = {}

    def fake_execvp(file, args):
        captured["file"] = file
        captured["args"] = args
        raise SystemExit(0)  # execvp would replace the process; stop here

    monkeypatch.setattr(cli.os, "execvp", fake_execvp)
    result = runner.invoke(app, ["dev", "PROJ-123"])
    assert result.exit_code == 0
    assert captured["file"] == "claude"
    assert captured["args"] == [
        "claude",
        "--name",
        "Developer: PROJ-123",
        "--model",
        "claude-opus-4-8",
        "Check in with boss",
    ]


def test_background_launch_prints_attach_hint(monkeypatch):
    # Force background regardless of the test runner's tty state.
    monkeypatch.setattr(cli, "stdout_isatty", lambda: False)
    monkeypatch.delenv("CLAUDECODE", raising=False)

    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="20e4e79f\n", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    result = runner.invoke(app, ["boss", "PROJ-123"])
    assert result.exit_code == 0
    assert calls["cmd"] == [
        "claude",
        "--bg",
        "--name",
        "Boss: PROJ-123",
        "--model",
        "fable",
    ]
    assert "claude attach 20e4e79f" in result.stdout


# --- quickstart ----------------------------------------------------------
def test_classify_team_counts_by_leading_word():
    agents = [
        {"name": "Boss: PROJ-456"},
        {"name": "boss 2"},
        {"name": "Developer: onboarding handoff 2"},
        {"name": "Developer A"},
        {"name": "QA: PROJ-456"},
        {"name": "QA 1"},
        {"name": "some-worktree-79"},
    ]
    assert classify_team(agents) == {"boss": 2, "developer": 2, "qa": 2}


def test_detect_state_with_injected_agents(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    agents = [{"name": "Boss: X"}, {"name": "QA: X"}]
    state = detect_state({}, isatty=True, agents_provider=lambda: agents)
    assert state["launch_context"] == "foreground"
    assert state["active_sessions"] == 2
    assert state["team"] == {"boss": 1, "developer": 0, "qa": 1}


def test_detect_state_handles_unavailable_agents(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    state = detect_state({"CLAUDECODE": "1"}, isatty=False, agents_provider=lambda: None)
    assert state["launch_context"] == "background"
    assert state["active_sessions"] is None
    assert state["team"] is None
    assert state["claude_cli"] is None


def test_quickstart_json_output(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli, "fetch_agents", lambda: [{"name": "Boss: X"}])
    result = runner.invoke(app, ["quickstart", "--json"])
    assert result.exit_code == 0
    assert '"active_sessions": 1' in result.stdout
    assert '"claude_team_cli"' in result.stdout


def test_quickstart_guide_mentions_roles_and_attach(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cli, "fetch_agents", lambda: [])
    result = runner.invoke(app, ["quickstart"])
    assert result.exit_code == 0
    assert "## Current state" in result.stdout
    assert "claude attach <id>" in result.stdout
    assert "Check in with boss" in result.stdout


# --- role_token ----------------------------------------------------------
def test_role_token_maps_each_role_to_its_subcommand():
    assert role_token(BOSS) == "boss"
    assert role_token(DEVELOPER) == "dev"
    assert role_token(QA) == "qa"


# --- Strategy: pane layouts ----------------------------------------------
def test_left_stack_layout_single_pane_has_no_splits():
    assert LeftStackLayout().plan(1) == []


def test_left_stack_layout_two_panes_split_right_evenly():
    assert LeftStackLayout().plan(2) == [SplitStep(0, "right", 0.5)]


def test_left_stack_layout_three_panes_boss_left_two_stacked_right():
    assert LeftStackLayout().plan(3) == [
        SplitStep(0, "right", 0.5),
        SplitStep(1, "down", 0.5),
    ]


def test_left_stack_layout_four_panes_even_right_column_rows():
    # right column of 3 rows: down-splits hand out 1/3 then 1/2 of the remainder
    assert LeftStackLayout().plan(4) == [
        SplitStep(0, "right", 0.5),
        SplitStep(1, "down", 1.0 / 3.0),
        SplitStep(2, "down", 0.5),
    ]


def test_columns_layout_three_even_columns():
    # peel one even column each time: keep 1/3, then 1/2 of the remainder
    assert ColumnsLayout().plan(3) == [
        SplitStep(0, "right", 1.0 / 3.0),
        SplitStep(1, "right", 0.5),
    ]


def test_layout_rejects_zero_panes():
    with pytest.raises(ValueError):
        LeftStackLayout().plan(0)


def test_get_layout_resolves_and_rejects():
    assert get_layout("left-stack").name == "left-stack"
    assert get_layout("columns").name == "columns"
    with pytest.raises(cli.typer.BadParameter):
        get_layout("spiral")


# --- argv builders -------------------------------------------------------
def test_workspace_create_argv():
    assert workspace_create_argv("/x", "lbl") == [
        "workspace", "create", "--cwd", "/x", "--label", "lbl", "--no-focus"
    ]


def test_worktree_create_argv():
    assert worktree_create_argv("/repo", "br", "origin/dev", "br") == [
        "worktree", "create", "--cwd", "/repo", "--branch", "br",
        "--base", "origin/dev", "--label", "br", "--no-focus",
    ]


def test_pane_split_argv_formats_ratio_and_omits_when_none():
    with_ratio = pane_split_argv("w1:p1", SplitStep(0, "right", 0.5), "/x")
    assert with_ratio == [
        "pane", "split", "w1:p1", "--direction", "right",
        "--ratio", "0.5000", "--cwd", "/x", "--no-focus",
    ]
    no_ratio = pane_split_argv("w1:p1", SplitStep(0, "down", None), "/x")
    assert "--ratio" not in no_ratio


def test_pane_run_argv_passes_command_as_single_arg():
    assert pane_run_argv("w1:p2", "echo hi") == ["pane", "run", "w1:p2", "echo hi"]


# --- claude_team_invocation ----------------------------------------------
def test_invocation_prefers_durably_installed_claude_team():
    which = lambda name: "/Users/j/.local/bin/claude-team" if name == "claude-team" else None
    assert claude_team_invocation(which=which, env={}) == ["claude-team"]


def test_invocation_ignores_claude_team_from_transient_uv_venv():
    # A claude-team that lives inside the active VIRTUAL_ENV is not durable for a
    # fresh pane shell, so fall back to the `uv run --project` form.
    def which(name):
        return {"claude-team": "/tmp/venv/bin/claude-team", "uv": "/usr/bin/uv"}.get(name)

    got = claude_team_invocation(which=which, package_root="/clone", env={"VIRTUAL_ENV": "/tmp/venv"})
    assert got == ["uv", "run", "--project", "/clone", "claude-team"]


def test_invocation_falls_back_to_uv_run_when_not_installed():
    which = lambda name: "/usr/bin/uv" if name == "uv" else None
    got = claude_team_invocation(which=which, package_root="/clone", env={})
    assert got == ["uv", "run", "--project", "/clone", "claude-team"]


def test_invocation_last_resort_is_this_interpreter_and_module():
    got = claude_team_invocation(which=lambda name: None, env={})
    assert got == [cli.sys.executable, "-m", "claude_team.cli"]


# --- pane_launch_command -------------------------------------------------
def test_pane_launch_command_appends_fg_and_role_token():
    cmd = pane_launch_command(["claude-team"], DEVELOPER, "PROJ-1", None)
    assert cmd == "claude-team dev PROJ-1 --fg"


def test_pane_launch_command_includes_suffix():
    cmd = pane_launch_command(["claude-team"], QA, "PROJ-1", "Handoff 2")
    assert cmd == "claude-team qa PROJ-1 --suffix 'Handoff 2' --fg"


def test_pane_launch_command_quotes_hostile_topic_safely():
    cmd = pane_launch_command(["ct"], BOSS, "a b | c && rm -rf ~", None)
    # round-trips through shlex, so the whole topic is a single inert token
    assert cli.shlex.split(cmd) == ["ct", "boss", "a b | c && rm -rf ~", "--fg"]


# --- Facade: Herdr -------------------------------------------------------
def test_herdr_create_workspace_parses_ids():
    fake = FakeHerdrRunner()
    ws, pane = Herdr(fake).create_workspace("/x", "lbl")
    assert (ws, pane) == ("w1", "w1:p1")
    assert fake.calls[0] == workspace_create_argv("/x", "lbl")


def test_herdr_create_worktree_returns_checkout():
    ws, pane, checkout = Herdr(FakeHerdrRunner()).create_worktree("/r", "b", "origin/dev", "b")
    assert (ws, pane, checkout) == ("w2", "w2:p1", "/co")


def test_herdr_split_returns_new_pane_id():
    assert Herdr(FakeHerdrRunner()).split("w1:p1", SplitStep(0, "right", 0.5), "/x") == "w1:p2"


def test_herdr_raises_on_nonzero_exit():
    def boom(args):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="socket gone\n")

    with pytest.raises(HerdrError, match="socket gone"):
        Herdr(boom).create_workspace("/x", "l")


def test_herdr_raises_on_error_payload():
    def err(args):
        return subprocess.CompletedProcess(
            args, 0, stdout=json.dumps({"error": {"code": "not_git_worktree"}}), stderr=""
        )

    with pytest.raises(HerdrError, match="not_git_worktree"):
        Herdr(err).create_workspace("/x", "l")


def test_herdr_raises_on_non_json():
    def junk(args):
        return subprocess.CompletedProcess(args, 0, stdout="not json", stderr="")

    with pytest.raises(HerdrError, match="non-JSON"):
        Herdr(junk).create_workspace("/x", "l")


def test_herdr_wait_for_output_swallows_errors():
    def boom(args):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="timeout")

    # Best-effort boot probe must never raise.
    Herdr(boom).wait_for_output("w1:p1", "x", 10)


# --- Director: place_team ------------------------------------------------
def test_place_team_splits_then_launches_boss_first():
    fake = FakeHerdrRunner()
    herdr = Herdr(fake)
    placements = place_team(
        herdr, LeftStackLayout(), "w1:p1", "/x",
        [BOSS, DEVELOPER, QA], "TOPIC", None, ["claude-team"],
    )
    # boss keeps the root pane; the two splits create p2, p3
    assert placements == [(BOSS, "w1:p1"), (DEVELOPER, "w1:p2"), (QA, "w1:p3")]
    # each role launched foreground in its pane, boss first
    assert fake.pane_runs() == [
        ("w1:p1", "claude-team boss TOPIC --fg"),
        ("w1:p2", "claude-team dev TOPIC --fg"),
        ("w1:p3", "claude-team qa TOPIC --fg"),
    ]


def test_place_team_single_role_no_splits():
    fake = FakeHerdrRunner()
    place_team(Herdr(fake), LeftStackLayout(), "w1:p1", "/x", [BOSS], "T", None, ["ct"])
    assert not [c for c in fake.calls if c[:2] == ["pane", "split"]]
    assert fake.pane_runs() == [("w1:p1", "ct boss T --fg")]


# --- CLI: space / worktree dry-run + guard -------------------------------
@pytest.fixture
def installed_ct(monkeypatch):
    """Make invocation deterministic: claude-team is durably installed."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(
        cli.shutil, "which",
        lambda name: "/opt/bin/claude-team" if name == "claude-team" else None,
    )


def test_space_dry_run_previews_workspace_and_launches(installed_ct):
    result = runner.invoke(app, ["space", "DEMO", "--dry-run"])
    assert result.exit_code == 0
    assert "herdr workspace create --cwd" in result.stdout
    assert "# layout: left-stack, 3 pane(s)" in result.stdout
    assert "[boss] claude-team boss DEMO --fg" in result.stdout
    assert "[qa] claude-team qa DEMO --fg" in result.stdout


def test_space_dry_run_respects_layout_and_roles(installed_ct):
    result = runner.invoke(
        app, ["space", "DEMO", "--layout", "columns", "--roles", "boss,qa", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "# layout: columns, 2 pane(s)" in result.stdout
    assert "dev DEMO" not in result.stdout


def test_worktree_dry_run_previews_worktree_create(installed_ct):
    result = runner.invoke(
        app, ["worktree", "my-branch", "--repo", "/repo", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "herdr worktree create --cwd /repo --branch my-branch --base origin/dev" in result.stdout
    assert "[boss] claude-team boss my-branch --fg" in result.stdout


def test_space_requires_herdr_env_when_not_dry_run(installed_ct, monkeypatch):
    monkeypatch.delenv("HERDR_ENV", raising=False)
    # Must never actually talk to herdr when the guard fails.
    monkeypatch.setattr(cli, "default_herdr_runner", lambda args: 1 / 0)
    result = runner.invoke(app, ["space", "DEMO"])
    assert result.exit_code != 0
    assert "not inside a Herdr session" in result.output


def test_space_real_run_reports_workspace_and_panes(installed_ct, monkeypatch):
    fake = FakeHerdrRunner()
    monkeypatch.setenv("HERDR_ENV", "1")
    monkeypatch.setattr(cli, "default_herdr_runner", fake)
    monkeypatch.setattr(cli, "herdr_available", lambda which=None: True)

    result = runner.invoke(app, ["space", "DEMO", "--no-wait"])
    assert result.exit_code == 0, result.output
    assert "✓ workspace w1" in result.stdout
    assert "Boss" in result.stdout and "w1:p1" in result.stdout
    assert "teardown: herdr workspace close w1" in result.stdout
    # boss launched first, in the root pane
    assert fake.pane_runs()[0] == ("w1:p1", "claude-team boss DEMO --fg")


# --- prompt overrides: boss --prompt / space|worktree --boss-prompt -------
def test_build_command_prompt_overrides_role_default():
    # boss has no default prompt; --prompt supplies one
    assert build_command(BOSS, "T", None, background=False, prompt="/resume x") == [
        "claude", "--name", "Boss: T", "--model", "fable", "/resume x",
    ]
    # dev's "Check in with boss" default is overridden
    cmd = build_command(DEVELOPER, "T", None, background=False, prompt="do this")
    assert cmd[-1] == "do this" and "Check in with boss" not in cmd


def test_build_command_no_prompt_keeps_role_default():
    cmd = build_command(DEVELOPER, "T", None, background=False)
    assert cmd[-1] == "Check in with boss"


def test_launch_plan_carries_prompt_into_argv():
    plan = LaunchPlan(BOSS, "T", None, background=True, prompt="/resume x")
    assert plan.argv()[-1] == "/resume x"


def test_pane_launch_command_includes_prompt():
    cmd = pane_launch_command(["claude-team"], BOSS, "T", None, prompt="/resume_handoff a b.md")
    assert cmd == "claude-team boss T --prompt '/resume_handoff a b.md' --fg"


def test_boss_prompts_maps_boss_when_selected():
    assert boss_prompts([BOSS, DEVELOPER, QA], "/resume x") == {BOSS: "/resume x"}


def test_boss_prompts_empty_without_prompt():
    assert boss_prompts([BOSS], None) == {}


def test_boss_prompts_warns_when_boss_not_selected(capsys):
    assert boss_prompts([DEVELOPER, QA], "/resume x") == {}
    assert "boss" in capsys.readouterr().err.lower()


def test_place_team_applies_boss_prompt_only_to_boss():
    fake = FakeHerdrRunner()
    place_team(
        Herdr(fake), LeftStackLayout(), "w1:p1", "/x",
        [BOSS, DEVELOPER], "T", None, ["ct"], {BOSS: "/resume x"},
    )
    runs = dict(fake.pane_runs())
    assert runs["w1:p1"] == "ct boss T --prompt '/resume x' --fg"  # boss carries it
    assert runs["w1:p2"] == "ct dev T --fg"                        # dev does not


def test_boss_prompt_cli_dry_run(terminal_env):
    result = runner.invoke(app, ["boss", "T", "--prompt", "/resume_handoff h.md", "--dry-run"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "claude --name 'Boss: T' --model fable '/resume_handoff h.md'"


def test_space_boss_prompt_dry_run(installed_ct):
    result = runner.invoke(
        app,
        ["space", "T", "--roles", "boss,dev", "--boss-prompt", "/resume_handoff h.md", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "[boss] claude-team boss T --prompt '/resume_handoff h.md' --fg" in result.stdout
    assert "[dev] claude-team dev T --fg" in result.stdout  # dev unaffected


def test_worktree_boss_prompt_dry_run(installed_ct):
    result = runner.invoke(
        app,
        ["worktree", "br", "--repo", "/r", "--roles", "boss", "--boss-prompt", "/resume h.md", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "[boss] claude-team boss br --prompt '/resume h.md' --fg" in result.stdout


def test_space_boss_prompt_ignored_when_boss_not_in_roles(installed_ct):
    result = runner.invoke(
        app, ["space", "T", "--roles", "dev,qa", "--boss-prompt", "/x", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "--prompt" not in result.stdout          # not applied
    assert "warning" in result.output.lower()       # but warned (stderr)
