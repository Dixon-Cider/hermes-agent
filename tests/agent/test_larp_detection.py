"""Unit tests for the post-turn LARP guard (agent/larp_detection.py).

Asserts the four-way contract: claim+no-tool -> re-prompt; claim+calls-but-none
-succeeded -> re-prompt unless the message owns the failure (require_success,
default True); claim+success -> pass; Tier-2 judge -> re-prompt on an UNGROUNDED
verdict. Tier-2 is exercised with the judge STUBBED — these tests never call an
LLM; they assert only when it is consulted and which question it is asked.
"""

from __future__ import annotations

from agent.larp_detection import build_larp_nudge, larp_detection_enabled

CFG = {"larp_detection": {"max_reprompts": 2, "exempt_toolsets": []}}


def _u(c):
    return {"role": "user", "content": c}


def _a(name):
    return {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": name, "arguments": "{}"}}]}


def _t(name, content):
    return {"role": "tool", "tool_call_id": "1", "name": name, "content": content}


def test_claim_with_no_tool_call_is_larp():
    assert build_larp_nudge(messages=[_u("do x")], final_response="I have updated the file.", config=CFG) is not None


def test_claim_with_successful_tool_passes():
    msgs = [_u("do x"), _a("write_file"), _t("write_file", "ok wrote 10 lines")]
    assert build_larp_nudge(messages=msgs, final_response="I have updated the file.", config=CFG) is None


def test_silent_claim_over_failed_tool_is_larp():
    # (b) the write failed, so "I have updated the file" is false — and the
    # message never says so. Previously passed; that was the guard's blind spot.
    msgs = [_u("do x"), _a("write_file"), _t("write_file", "Error executing tool 'write_file': denied")]
    nudge = build_larp_nudge(messages=msgs, final_response="I have updated the file.", config=CFG)
    assert nudge is not None
    assert "every tool call this turn failed" in nudge


def test_claim_that_owns_the_failure_passes():
    # (b) honest narration of a broken tool -> pass, as before.
    msgs = [_u("do x"), _a("write_file"), _t("write_file", "Error executing tool 'write_file': denied")]
    fr = "I ran the write but it failed: permission denied on that path."
    assert build_larp_nudge(messages=msgs, final_response=fr, config=CFG) is None


def test_launch_family_gerunds_are_claims():
    # These verbs went unmatched entirely, so the guard never even saw a claim.
    for fr in (
        "Launching the classifier now.",
        "Spawning the background job now...",
        "Re-running the failed batch now.",
        "Retrying the connection now.",
        "Restarting the server immediately.",
    ):
        assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is not None, fr


def test_leading_now_and_let_me_announcements_are_claims():
    # "now" leading the clause with a trailing purpose clause, and the
    # "Let me X" imperative — both escaped every pattern before.
    for fr in (
        "Terminal pane focused. Now launching the classifier so you can watch live output.",
        "Let me launch it as a background process so you can watch live output.",
        "Let me start the ingestion and report back.",
    ):
        assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is not None, fr


def test_let_me_know_is_not_a_claim():
    # The inverse of "Let me X": handing control back, not announcing work.
    for fr in (
        "I could not reach the server. Let me know how you want to proceed.",
        "Let me know if you want me to run the migration.",
    ):
        assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is None, fr


def test_intent_announcement_over_failed_tool_is_larp_even_if_failure_mentioned():
    # The transcript case: the model owns the OLD failure, then promises new work
    # it never performed. Acknowledging a failure must not license an intent claim.
    msgs = [_u("run it"), _a("terminal"), _t("terminal", "Error: command timed out after 180s")]
    fr = "The foreground run timed out after 180s. Launching it in the background now."
    assert build_larp_nudge(messages=msgs, final_response=fr, config=CFG) is not None


def test_require_success_false_restores_old_pass_through():
    msgs = [_u("do x"), _a("write_file"), _t("write_file", "Error executing tool 'write_file': denied")]
    cfg = {"larp_detection": {"require_success": False, "exempt_toolsets": []}}
    assert build_larp_nudge(messages=msgs, final_response="I have updated the file.", config=cfg) is None


def test_call_with_no_result_is_treated_as_unsuccessful():
    # An assistant tool_call whose result never arrived grounds nothing.
    msgs = [_u("do x"), _a("write_file")]
    assert build_larp_nudge(messages=msgs, final_response="I have updated the file.", config=CFG) is not None


def test_no_tool_nudge_keeps_its_original_wording():
    # (a) must stay distinguishable from (b) so the model corrects the right thing.
    nudge = build_larp_nudge(messages=[_u("x")], final_response="I have updated the file.", config=CFG)
    assert nudge is not None
    assert "no corresponding tool call was made this turn" in nudge


def test_no_claim_passes():
    assert build_larp_nudge(messages=[_u("hi")], final_response="Here is a summary of the weather.", config=CFG) is None


def test_modal_is_not_a_claim():
    fr = "I should update the file but need confirmation first."
    assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is None


def test_narrate_then_stop_is_larp():
    fr = "Sounds good. I'll now run the ingestion script."
    assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is not None


def test_any_tool_call_passes_by_default():
    # default exempt is empty -> a memory call counts as real work.
    msgs = [_u("x"), _a("memory"), _t("memory", "saved")]
    assert build_larp_nudge(messages=msgs, final_response="I have saved that.", config=CFG) is None


def test_strict_exempt_flags_housekeeping_only_turn():
    msgs = [_u("x"), _a("memory"), _t("memory", "saved")]
    cfg = {"larp_detection": {"exempt_toolsets": ["memory"]}}
    assert build_larp_nudge(messages=msgs, final_response="I have updated the database.", config=cfg) is not None


def test_reprompt_cap():
    assert build_larp_nudge(messages=[_u("x")], final_response="I have updated the file.", config=CFG, attempts=2) is None


def test_disabled_by_default():
    assert larp_detection_enabled({"larp_detection": {"enabled": False}}) is False


def test_env_cannot_override_config(monkeypatch):
    # Enablement is config-only (AGENTS.md: no HERMES_* vars for behavior).
    monkeypatch.setenv("HERMES_LARP_DETECTION", "1")
    assert larp_detection_enabled({"larp_detection": {"enabled": False}}) is False
    assert larp_detection_enabled({"larp_detection": {"enabled": True}}) is True


# ---- tuning from real session strings (narrate-then-stop / terminal action) ----

def test_present_progressive_narrate_is_larp():
    # "I am dispatching ..." — the dominant form the old future-only regex MISSED.
    fr = "The next 5 products to research (Phase 3): ...\n\nI am dispatching the sub-agents now."
    assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is not None


def test_bare_terminal_action_is_larp():
    for fr in ("Executing now.", "Starting Batch 1 now.", "Proceeding with dispatch...",
               "Correcting the script creation now..."):
        assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is not None, fr


def test_proceeding_with_item_now_is_larp():
    fr = "### Project State\n- Total Completed: 36\n\nI am proceeding with PGPx9944 now."
    assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is not None


def test_permission_question_now_is_not_larp():
    # trailing "?" => asking permission, not claiming -> must NOT flag.
    for fr in ("Want me to execute this now?", "Want me to snip Figures 3 and 4 now?",
               "Should I proceed now?"):
        assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is None, fr


def test_waiting_status_is_not_larp():
    fr = "**Current KG state:** 103 products.\n\nWaiting for batch 3 result..."
    assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is None


def test_i_am_state_is_not_larp():
    # "I am ready/unable" are states, not gerund actions -> must NOT match narrate.
    for fr in ("I am ready to help with the next step.", "I am unable to do that right now."):
        assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is None, fr


def test_present_progressive_with_tool_passes():
    # (c) a real tool call this turn backs the announcement -> pass.
    msgs = [_u("x"), _a("delegate_task"), _t("delegate_task", "spawned 5 subagents")]
    fr = "I am dispatching the sub-agents now."
    assert build_larp_nudge(messages=msgs, final_response=fr, config=CFG) is None


def test_post_compaction_window_enables_when_disabled():
    cfg = {"larp_detection": {"enabled": False, "post_compaction_window": 3}}

    class _InWindow:
        _turns_since_compaction = 1

    class _PastWindow:
        _turns_since_compaction = 5

    class _NeverCompacted:
        _turns_since_compaction = None

    assert larp_detection_enabled(cfg) is False                          # no agent -> off
    assert larp_detection_enabled(cfg, agent=_InWindow()) is True        # within window -> on
    assert larp_detection_enabled(cfg, agent=_PastWindow()) is False     # past window -> off
    assert larp_detection_enabled(cfg, agent=_NeverCompacted()) is False # never compacted -> off


def test_post_compaction_window_default_off():
    class _JustCompacted:
        _turns_since_compaction = 0

    # default window 0 -> disabled stays disabled even right after compaction.
    assert larp_detection_enabled({"larp_detection": {"enabled": False}}, agent=_JustCompacted()) is False


# ---- asking the user is NOT claiming ----
# The guard must never re-prompt a turn that stops to ask a question: doing so
# pushes the model to act without the approval it just requested — worse than
# the LARP it is trying to prevent. Regression for the "question + conditional
# future" shape, which the trailing-"?" check alone did not cover.


def test_question_then_conditional_future_is_not_larp():
    for fr in (
        "Which approach do you prefer? Let me know and I'll implement it.",
        "Should I use the staging or prod bucket? Once you confirm, I'll start the migration.",
        "Want me to proceed? If so, I'll run the migration now.",
        "I can do this two ways. Which do you prefer? Just say the word.",
    ):
        assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is None, fr


def test_conditional_lead_without_question_is_not_larp():
    # The question can fall outside the tail window on a long response; an
    # explicit "waiting on you" clause must still suppress the announcement.
    fr = "Here is the full plan.\n\n" + ("Detail line.\n" * 40) + "Once you approve, I'll start the migration."
    assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is None


def test_blocked_asking_for_input_is_not_larp():
    for fr in (
        "I need the API key before I can continue. Where should I read it from?",
        "I'm blocked: the repo has no remote configured. Which remote should I add?",
    ):
        assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is None, fr


def test_past_tense_claim_still_flagged_despite_a_question():
    # A factual past-tense assertion is a claim regardless of any question or
    # confirmation language elsewhere — suppression applies only to the
    # conditional intent-announcement branch.
    fr = "I confirmed the plan with you earlier. I have now deployed the changes."
    assert build_larp_nudge(messages=[_u("x")], final_response=fr, config=CFG) is not None


# --------------------------- Tier-2 judge gating ---------------------------
# The judge is opt-in and fails open; these stub it to assert only WHEN it runs.

JUDGE_CFG = {"larp_detection": {"judge_tier_enabled": True, "exempt_toolsets": []}}


def _stub_judge(monkeypatch, verdict: bool, calls: list):
    import agent.larp_detection as ld

    def fake(messages, final_response, claim, kind="completed"):
        calls.append((claim, kind))
        return verdict

    monkeypatch.setattr(ld, "_judge_ungrounded", fake)


def test_judge_runs_when_tools_failed(monkeypatch):
    # Regression: the judge used to be gated behind any_success, so it switched
    # itself off exactly when every tool was failing.
    calls: list = []
    _stub_judge(monkeypatch, True, calls)
    msgs = [_u("x"), _a("shell"), _t("shell", "Error: connection refused")]
    # require_success off, so the (b) branch can't mask what the judge does.
    cfg = {"larp_detection": {"judge_tier_enabled": True, "require_success": False, "exempt_toolsets": []}}
    fr = "I ran the query and it returned 42 rows, so the import failed cleanly."
    assert build_larp_nudge(messages=msgs, final_response=fr, config=cfg) is not None
    assert calls, "judge should run on a turn whose tools all failed"


def test_judge_runs_on_successful_turn(monkeypatch):
    calls: list = []
    _stub_judge(monkeypatch, True, calls)
    msgs = [_u("x"), _a("shell"), _t("shell", "ok")]
    fr = "I ran the query and it returned 42 rows."
    assert build_larp_nudge(messages=msgs, final_response=fr, config=JUDGE_CFG) is not None
    assert calls


def test_judge_grounded_verdict_passes(monkeypatch):
    calls: list = []
    _stub_judge(monkeypatch, False, calls)
    msgs = [_u("x"), _a("shell"), _t("shell", "ok")]
    fr = "I ran the query and it returned 42 rows."
    assert build_larp_nudge(messages=msgs, final_response=fr, config=JUDGE_CFG) is None
    assert calls


def test_judge_not_consulted_for_vague_claims(monkeypatch):
    # _looks_specific still gates it: no numbers/result words -> no LLM call.
    calls: list = []
    _stub_judge(monkeypatch, True, calls)
    msgs = [_u("x"), _a("shell"), _t("shell", "ok")]
    assert build_larp_nudge(messages=msgs, final_response="I have updated the file.", config=JUDGE_CFG) is None
    assert not calls


def test_judge_off_by_default(monkeypatch):
    calls: list = []
    _stub_judge(monkeypatch, True, calls)
    msgs = [_u("x"), _a("shell"), _t("shell", "ok")]
    fr = "I ran the query and it returned 42 rows."
    assert build_larp_nudge(messages=msgs, final_response=fr, config=CFG) is None
    assert not calls


# ------------------- Tier-2 on intent announcements -------------------
# The gap tier-1 cannot close: a SUCCESSFUL but unrelated tool call satisfies
# rule (c), so "Terminal pane focused. Now launching X." looks grounded.

# A focus call succeeded; the announced launch never happened.
ADJACENT = [
    _u("run it in that terminal"),
    _a("desktop_ui"),
    _t("desktop_ui", '{"ok": true, "focused": "terminal"}'),
]
FR_ADJACENT = "Terminal pane focused. Now launching the classifier so you can watch live output."


def test_intent_claim_with_adjacent_success_is_caught_by_judge(monkeypatch):
    calls: list = []
    _stub_judge(monkeypatch, True, calls)
    nudge = build_larp_nudge(messages=ADJACENT, final_response=FR_ADJACENT, config=JUDGE_CFG)
    assert nudge is not None
    assert "performed the action you announced" in nudge
    assert calls and calls[0][1] == "intent", "judge must be asked the intent question"


def test_intent_claim_judged_grounded_passes(monkeypatch):
    # Present-progressive narration of work a tool really did stays legitimate.
    calls: list = []
    _stub_judge(monkeypatch, False, calls)
    msgs = [_u("x"), _a("delegate_task"), _t("delegate_task", "spawned 5 subagents")]
    fr = "I am dispatching the sub-agents now."
    assert build_larp_nudge(messages=msgs, final_response=fr, config=JUDGE_CFG) is None
    assert calls and calls[0][1] == "intent"


def test_intent_routing_can_be_disabled(monkeypatch):
    calls: list = []
    _stub_judge(monkeypatch, True, calls)
    cfg = {
        "larp_detection": {
            "judge_tier_enabled": True,
            "judge_intent_claims": False,
            "exempt_toolsets": [],
        }
    }
    assert build_larp_nudge(messages=ADJACENT, final_response=FR_ADJACENT, config=cfg) is None
    assert not calls


def test_intent_routing_still_requires_judge_tier(monkeypatch):
    # judge_intent_claims defaults True but must not turn the judge on by itself.
    calls: list = []
    _stub_judge(monkeypatch, True, calls)
    assert build_larp_nudge(messages=ADJACENT, final_response=FR_ADJACENT, config=CFG) is None
    assert not calls


def test_completed_claim_still_gets_the_completed_question(monkeypatch):
    calls: list = []
    _stub_judge(monkeypatch, True, calls)
    msgs = [_u("x"), _a("shell"), _t("shell", "ok")]
    fr = "I ran the query and it returned 42 rows."
    assert build_larp_nudge(messages=msgs, final_response=fr, config=JUDGE_CFG) is not None
    assert calls and calls[0][1] == "completed"


def test_judge_not_reached_when_tier1_already_caught_it(monkeypatch):
    # require_success settles the no-success intent case deterministically;
    # spending an aux call there would be pure latency.
    calls: list = []
    _stub_judge(monkeypatch, True, calls)
    msgs = [_u("x"), _a("terminal"), _t("terminal", "Error: command timed out after 180s")]
    assert build_larp_nudge(messages=msgs, final_response="Launching it now.", config=JUDGE_CFG) is not None
    assert not calls, "tier-1 should have decided this without the judge"
