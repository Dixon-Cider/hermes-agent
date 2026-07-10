"""Regression: the three update executors must be fork-aware.

When a managed checkout carries local-only commits (a fork with custom
features), ``git reset --hard origin/$BRANCH`` would destroy that work. On a
fork — detected by the presence of an ``upstream`` remote — the updater must
instead MERGE ``upstream/main`` into the branch (rerere auto-resolves recurring
conflicts; a genuinely new conflict aborts cleanly) and must NOT hard-reset.

Covers all three paths that a user's update can flow through:
  * ``scripts/install.ps1``  — Windows desktop "Update" button / bootstrap
  * ``scripts/install.sh``   — macOS/Linux bootstrap
  * ``hermes_cli/main.py``   — the ``hermes update`` CLI

These are static assertions on the update code (matching the style of
``test_install_diverged_update.py``); they don't execute git.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"
MAIN_PY = REPO_ROOT / "hermes_cli" / "main.py"


def _install_sh_block() -> str:
    text = INSTALL_SH.read_text(encoding="utf-8")
    m = re.search(
        r"(?P<block>git checkout \"\$BRANCH\".*?fi\n\n            if \[ -n \"\$autostash_ref\" \])",
        text,
        re.DOTALL,
    )
    assert m is not None, "managed-install update block not found in install.sh"
    return m["block"]


def _install_ps1_block() -> str:
    text = INSTALL_PS1.read_text(encoding="utf-8")
    m = re.search(
        r"(?P<block>git -c windows\.appendAtomically=false checkout \$Branch.*?elseif \(\$Tag\))",
        text,
        re.DOTALL,
    )
    assert m is not None, "branch update block not found in install.ps1"
    return m["block"]


# --------------------------- install.sh ---------------------------

def test_install_sh_merges_upstream_on_fork() -> None:
    block = _install_sh_block()

    # Fork detection, merge (not reset), and clean abort on a new conflict.
    assert "git remote | grep -qx upstream" in block
    assert "git merge --no-edit upstream/main" in block
    assert "git merge --abort" in block

    # The upstream merge is attempted before (and instead of) the reset.
    assert block.find("git merge --no-edit upstream/main") < block.find(
        'git reset --hard "origin/$BRANCH"'
    )


def test_install_sh_reset_is_gated_behind_non_fork_else() -> None:
    block = _install_sh_block()
    # The destructive reset must live in the `else` (non-fork) branch, i.e.
    # after the `else` keyword — never in the fork path.
    assert re.search(r"\n            else\b", block), "expected a non-fork else branch"
    assert block.find("\n            else") < block.find('git reset --hard "origin/$BRANCH"')


# --------------------------- install.ps1 ---------------------------

def test_install_ps1_merges_upstream_on_fork() -> None:
    block = _install_ps1_block()

    assert '-contains "upstream"' in block
    assert "merge --no-edit upstream/main" in block
    assert "merge --abort" in block

    assert block.find("merge --no-edit upstream/main") < block.find(
        'reset --hard "origin/$Branch"'
    )


def test_install_ps1_reset_is_gated_behind_non_fork_else() -> None:
    block = _install_ps1_block()
    assert "} else {" in block
    assert block.find("} else {") < block.find('reset --hard "origin/$Branch"')


# --------------------------- hermes update (CLI) ---------------------------

def test_cmd_update_defines_fork_merge_helper() -> None:
    text = MAIN_PY.read_text(encoding="utf-8")
    assert "def _fork_merge_upstream(" in text
    # The helper merges (not fast-forward-only) and aborts cleanly on conflict.
    helper = text[text.index("def _fork_merge_upstream(") :]
    helper = helper[: helper.index("\n\ndef ")]
    assert '"merge", "--no-edit", "upstream/main"' in helper
    assert '"merge", "--abort"' in helper
    assert "sys.exit(1)" in helper


def test_cmd_update_gates_destructive_reset_in_fork_mode() -> None:
    text = MAIN_PY.read_text(encoding="utf-8")
    # Fork mode is computed from origin being a fork + an upstream remote.
    assert "_fork_mode = is_fork and _has_upstream_remote(" in text
    # The "already up to date with origin" short-circuit yields to a fresh merge.
    assert "if commit_count == 0 and not _fork_updated:" in text
    # The origin reset --hard fallback is skipped in fork mode.
    assert "if pull_result.returncode != 0 and not _fork_mode:" in text
