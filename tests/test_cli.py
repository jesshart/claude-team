"""Behavior tests for the claude-team CLI."""

from __future__ import annotations

import subprocess

import pytest
from typer.testing import CliRunner

from claude_team import cli
from claude_team.cli import (
    BOSS,
    CHECK_IN_PROMPT,
    DEVELOPER,
    FABLE,
    LaunchPlan,
    OPUS_48,
    QA,
    app,
    build_command,
    build_name,
    classify_team,
    decide_background,
    detect_state,
    extract_session_id,
    parse_roles,
)

runner = CliRunner()


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
