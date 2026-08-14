"""Post-turn LARP guard: catch "claimed an action but didn't call a tool".

Policy-only (mirrors :mod:`agent.verification_stop`): it inspects the just-finished
turn and, when the model asserts it performed an action but made **no** matching
(substantive) tool call, returns a corrective nudge so the conversation loop
re-prompts instead of finalizing.

Four-way contract:
  (a) action-claim + ZERO substantive tool calls this turn -> TRUE LARP -> re-prompt
  (b) action-claim + substantive calls but NONE succeeded  -> nothing actually
      performed the action. Re-prompt UNLESS the message itself owns the failure
      (see below). Governed by ``larp_detection.require_success`` (default True).
  (c) action-claim + a successful substantive tool          -> pass through
  (d) any of the above + Tier-2 judge says an outcome-specific claim, or an
      INTENT announcement, is ungrounded -> re-prompt. Intent announcements need
      the judge because a successful but UNRELATED tool call satisfies (c):
      "Terminal pane focused. Now launching the classifier." looks grounded to
      every deterministic check, and only a semantic one sees that focusing a
      pane launches nothing.

(b) is the highest-value catch and used to be the guard's blind spot: when every
tool this turn fails, the model tends to stop reporting the failure and start
narrating the next step as though it had run ("Terminal focused. Now launching
X."). The old rule passed that through on the theory that a failed tool is
honest narration — true only when the message SAYS so. So the suppression is
narrowed to exactly that case:

  * an INTENT announcement ("Now launching X") is never grounded by a failed
    tool — it is a promise that ended the turn -> always re-prompt;
  * a past-tense COMPLETION claim ("I updated the file") is honest only if the
    message acknowledges the failure -> pass when it does, re-prompt when it
    silently asserts success a failed tool never delivered.

Disabled by default (opt-in); see ``DEFAULT_CONFIG["larp_detection"]``.
Tier-2 (an LLM judge for outcome-specific claims) is a further opt-in and fails
open (never re-prompts on error). It runs on any turn with tool activity — NOT
only successful ones, or it would switch itself off exactly when the tools are
broken and the LARP risk is highest.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import logging

logger = logging.getLogger(__name__)

_FALSEY = {"0", "false", "no", "off", ""}

# Tool-name tokens whose calls do NOT count as "substantive" work. EMPTY by
# default for the lowest false-positive rate: any real tool call this turn means
# the model "did something", so a claim is not flagged. Users can add tokens via
# ``larp_detection.exempt_toolsets`` to make detection stricter (e.g. so a turn
# that only wrote to memory/todo still counts as a no-op for substantive claims).
_DEFAULT_EXEMPT: set[str] = set()

# Past-tense completion of a substantive action.
_ACTION_VERBS = (
    "updated|created|saved|wrote|written|ingested|added|removed|deleted|ran|"
    "executed|searched|fetched|downloaded|installed|configured|committed|pushed|"
    "sent|applied|fixed|implemented|generated|stored|recorded|registered|modified|"
    "edited|patched|built|deployed|uploaded|inserted|populated|completed"
)

_CLAIM_PATTERNS = [
    # "I have updated ...", "I've saved ...", "I updated ..."
    re.compile(
        r"\bI(?:\s+have|'ve)?\s+(?:just\s+|now\s+|already\s+|successfully\s+)?(?:"
        + _ACTION_VERBS
        + r")\b",
        re.IGNORECASE,
    ),
    # bare completion status
    re.compile(
        r"\b(?:all\s+(?:steps|tasks|items)\s+(?:are\s+)?(?:complete|completed|done)|"
        r"task\s+(?:is\s+)?(?:complete|completed|done)|completed\s+successfully|"
        r"successfully\s+completed)\b",
        re.IGNORECASE,
    ),
]

# Present-progressive / imperative action verbs used to ANNOUNCE (not report)
# work. Local models routinely end a turn with these instead of calling the
# tool: "I am proceeding with X now.", "Executing now.", "Proceeding with
# dispatch...". (Deliberately excludes status words like "waiting"/"processing".)
_ACTION_GERUNDS = (
    r"proceeding|executing|dispatching|initiating|starting|running|creating|"
    r"fixing|correcting|continuing|beginning|generating|building|deploying|"
    r"uploading|downloading|fetching|searching|updating|writing|saving|"
    r"ingesting|installing|committing|pushing|sending|applying|implementing|"
    r"rewriting|recreating|moving|kicking\s+off|"
    # Launch/dispatch family: the dominant real-world form when a tool just
    # failed and the model promises a retry it never issues ("Launching it in
    # the background now.", "Spawning the job now...", "Re-running now.").
    r"launching|relaunching|spawning|restarting|rerunning|re-running|retrying|"
    r"reattempting|re-attempting|opening|scheduling|queueing|queuing|triggering|"
    r"invoking|submitting|syncing|cloning|pulling|refreshing|exporting|importing"
)

# "narrate then stop": the message END announces intent instead of doing it.
# Covers "I'll X" / "I will X" / "I am going to X" AND the present-progressive
# "I am (now) proceeding/dispatching..." / "I'm executing..." — the dominant
# real-world form the earlier future-only pattern missed. Requires a gerund
# after "I am" so states ("I am unable/ready/done/sorry") don't match.
_NARRATE_THEN_STOP = re.compile(
    r"\bI(?:'ll|\s+will)\s+\w+"
    r"|\bI(?:'m|\s+am)\s+(?:now\s+|currently\s+)?(?:going\s+to\s+\w+|\w+ing\b)"
    # "Let me launch it in the background" — first-person imperative intent.
    # "Let me know ..." is the opposite (handing control back), so it is excluded
    # here as well as by _CONDITIONAL_LEAD.
    r"|\blet me\s+(?!know\b|see\s+if\b)\w+"
    # "Now launching the classifier so you can watch." — the leading-"now" form.
    # _TERMINAL_ACTION only matches when the marker TRAILS the sentence, so a
    # purpose clause after the verb ("...so you can watch") used to escape both.
    r"|\bnow\s+(?:" + _ACTION_GERUNDS + r")\b",
    re.IGNORECASE,
)

# Bare terminal action announcement: a sentence STARTING with an action gerund
# Clauses that make a following announcement conditional on the USER, not a
# claim of work done: "Let me know and I'll implement it", "Once you confirm,
# I'll start", "If so, I'll run it now". Used to suppress the intent-announcement
# branch when the model is waiting on approval (see _first_claim). Kept separate
# from the question check so it still fires when the question fell outside the
# tail window of a long response.
_CONDITIONAL_LEAD = re.compile(
    r"\b(?:let me know|just say|say the word|tell me|once you|after you|when you|"
    r"if so|if you|if that|if it|pending your|awaiting your|on your (?:go|approval|confirmation)|"
    r"confirm(?:ed)?|approve|give me the (?:go|green light))\b",
    re.IGNORECASE,
)

# and ENDING the message with "now"/"immediately"/"…" (no trailing question).
# Catches "Executing now.", "Starting Batch 1 now.", "Proceeding with dispatch…".
_TERMINAL_ACTION = re.compile(
    r"(?:^|[.!\n]\s*)(?:" + _ACTION_GERUNDS + r")\b[^?\n]*?"
    r"(?:\bnow\b|\bimmediately\b|\.\.\.|…)[.!…\"'\s]*$",
    re.IGNORECASE,
)

# Modal/conditional words right before a verb that make it NOT a completion claim.
_MODAL_PREFIX = re.compile(r"\b(can|could|should|would|might|may|need to|try to|plan to)\s*$", re.IGNORECASE)

# The message OWNS a failure: it tells the user something went wrong. A past-tense
# claim alongside this is honest narration of a broken tool ("I ran the tests and
# they errored"), not a LARP — so contract (b) passes it through. Deliberately
# does NOT rescue intent announcements: "it timed out, now launching it in the
# background" acknowledges the old failure while still promising unperformed work.
_ACK_FAILURE = re.compile(
    r"\b(?:fail(?:ed|ing|s|ure)?|error(?:ed|s)?|timed\s*out|timeout|unable\s+to|"
    r"could\s*n[o']t|cannot|can'?t|refused|denied|unreachable|not\s+reachable|"
    r"broke|broken|did\s*n[o']t\s+work|no\s+such|rejected|blocked)\b",
    re.IGNORECASE,
)


def _section(config: Optional[dict]) -> dict:
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}
    sec = config.get("larp_detection") if isinstance(config, dict) else None
    return sec if isinstance(sec, dict) else {}


def _flag(sec: dict, key: str, default: bool) -> bool:
    val = sec.get(key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() not in _FALSEY
    return bool(val)


def larp_detection_enabled(config: Optional[dict] = None, agent: Any = None) -> bool:
    """Whether the post-turn LARP guard runs this turn.

    Enablement is config-only (``larp_detection.enabled``): per AGENTS.md,
    behavioral settings belong in ``config.yaml`` and ``.env`` is for secrets,
    so there is deliberately no env-var override.
    """
    sec = _section(config)
    if _flag(sec, "enabled", False):
        return True
    # Opt-in high-risk window: LARPing spikes right after context compaction —
    # the summary reads as completed-action prose with the tool calls stripped,
    # and the model imitates it. When post_compaction_window > 0, run the guard
    # for that many turns after each compaction even if otherwise disabled.
    # Default 0 -> no behavior change.
    window = int(sec.get("post_compaction_window", 0) or 0)
    if window > 0 and agent is not None:
        tsc = getattr(agent, "_turns_since_compaction", None)
        if isinstance(tsc, int) and 0 <= tsc <= window:
            return True
    return False


def _exempt_tokens(config: Optional[dict]) -> set[str]:
    tokens = set(_DEFAULT_EXEMPT)
    raw = _section(config).get("exempt_toolsets")
    if isinstance(raw, (list, tuple, set)):
        tokens |= {str(t).strip().lower() for t in raw if str(t).strip()}
    return tokens


def _is_substantive(name: str, exempt: set[str]) -> bool:
    n = (name or "").strip().lower()
    if not n:
        return False
    return not any(tok in n for tok in exempt)


def _acknowledges_failure(text: str) -> bool:
    """Whether the message tells the user something went wrong this turn."""
    return bool(_ACK_FAILURE.search(text or ""))


def _first_claim_with_kind(text: str) -> Optional[tuple[str, str]]:
    """Like :func:`_first_claim` but also reports which KIND of claim matched:

    ``"completed"`` - past-tense/bare completion assertion ("I updated the file")
    ``"intent"``    - narrate-then-stop / bare terminal action ("Now launching X")

    The distinction matters only for contract (b): a failed tool can make a
    completion claim honest, but never makes an intent announcement performed.
    """
    for pat in _CLAIM_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        prefix = text[max(0, m.start() - 16) : m.start()]
        if _MODAL_PREFIX.search(prefix):
            continue
        return text[m.start() : m.start() + 140].strip(), "completed"
    # Intent-announcement (narrate-then-stop / bare terminal action) counts only
    # at the END of the message — and NOT when the message ends by asking the
    # user ("Want me to X now?" / "... now?"), which is correct stop-to-confirm
    # behavior, not a LARP.
    if text.rstrip().endswith("?"):
        return None
    tail = text[-200:]
    m = _NARRATE_THEN_STOP.search(tail) or _TERMINAL_ACTION.search(tail)
    if not m:
        return None
    # An announcement that is CONDITIONAL on the user is a request for
    # confirmation, not a claim: "Which approach do you prefer? Let me know and
    # I'll implement it." Re-prompting there is actively harmful — it pushes the
    # model to act without the approval it just asked for. Suppress when the
    # announcement is preceded (nearby) by a question, or by an explicit
    # "waiting on you" clause. Past-tense claims above are unaffected: those are
    # factual assertions regardless of any question that follows.
    before = tail[: m.start()]
    if "?" in before or _CONDITIONAL_LEAD.search(before):
        return None
    return tail[m.start() : m.start() + 140].strip(), "intent"


def _first_claim(text: str) -> Optional[str]:
    """Back-compat wrapper: the claim text only, without its kind."""
    found = _first_claim_with_kind(text)
    return found[0] if found else None


def _looks_specific(claim: str) -> bool:
    return bool(re.search(r"\d", claim)) or bool(
        re.search(r"\b(found|contains?|returned|listed|retrieved)\b", claim, re.IGNORECASE)
    )


def _last_user_index(messages: list) -> int:
    synthetic = ("_verification_stop_synthetic", "_larp_reprompt_synthetic", "_empty_recovery_synthetic")
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "user" and not any(m.get(k) for k in synthetic):
            return i
    return -1


def _turn_tool_activity(messages: list, exempt: set[str]) -> tuple[bool, bool, bool]:
    """Return (made_substantive_call, any_success, any_fail) since the last user msg."""
    try:
        from agent.display import _detect_tool_failure
    except Exception:
        _detect_tool_failure = None  # type: ignore[assignment]

    start = _last_user_index(messages)
    made = any_success = any_fail = False
    id_to_name: dict[str, str] = {}
    for m in messages[start + 1 :]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                name = fn.get("name") or (tc.get("name") if isinstance(tc, dict) else "") or ""
                if _is_substantive(name, exempt):
                    made = True
                tcid = tc.get("id") if isinstance(tc, dict) else None
                if tcid:
                    id_to_name[tcid] = name
        elif role == "tool":
            name = m.get("name") or m.get("tool_name") or id_to_name.get(m.get("tool_call_id"), "")
            if not _is_substantive(name, exempt):
                continue
            is_err = False
            if _detect_tool_failure is not None:
                try:
                    is_err, _ = _detect_tool_failure(name, m.get("content"))
                except Exception:
                    is_err = False
            if is_err:
                any_fail = True
            else:
                any_success = True
    return made, any_success, any_fail


def _judge_ungrounded(
    messages: list, final_response: str, claim: str, kind: str = "completed"
) -> bool:
    """Tier-2: one cheap aux-LLM check. Returns True only on a confident UNGROUNDED
    verdict; fails open (False) on any error.

    *kind* selects the question asked. "completed" audits whether a past-tense
    claim is supported by the tool results. "intent" audits whether the ANNOUNCED
    action was actually carried out — a different judgement, because a present
    -progressive narration of work a tool really did ("I am dispatching the
    sub-agents now" + a delegate call) is legitimate, while the same sentence
    with only an adjacent tool call is the LARP.
    """
    try:
        from agent.auxiliary_client import call_llm

        start = _last_user_index(messages)
        tool_lines = []
        for m in messages[start + 1 :]:
            if isinstance(m, dict) and m.get("role") == "tool":
                c = m.get("content")
                tool_lines.append(f"- {m.get('name') or m.get('tool_name')}: {str(c)[:300]}")
        tool_summary = "\n".join(tool_lines[:20]) or "(no tool results this turn)"
        if kind == "intent":
            sys = (
                "You audit whether the action an assistant ANNOUNCED was actually carried "
                "out by one of this turn's tool calls. Narrating work a tool really did is "
                "fine — answer GROUNDED. Announcing work and ending the turn without a tool "
                "call that performs it is a failure — answer UNGROUNDED. A tool call that "
                "only does something ADJACENT to the announced action (opening or focusing "
                "a window, reading a file, checking status) does NOT count as performing it. "
                "Reply with exactly one word: GROUNDED or UNGROUNDED."
            )
            check = f"Specifically check this announced action: {claim[:200]}"
        else:
            sys = (
                "You audit whether an assistant's claims of completed actions are GROUNDED "
                "in the tool results from this turn. Reply with exactly one word: GROUNDED "
                "or UNGROUNDED."
            )
            check = f"Specifically check this claim: {claim[:200]}"
        usr = (
            f"Assistant final message:\n{final_response[:1500]}\n\n"
            f"Tool results this turn:\n{tool_summary[:2000]}\n\n"
            f"{check}"
        )
        resp = call_llm(
            task="larp_detection",
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            max_tokens=8,
            temperature=0.0,
            timeout=20.0,
        )
        verdict = (resp.choices[0].message.content or "").strip().upper()
        return verdict.startswith("UNGROUND")
    except Exception:
        logger.debug("LARP judge failed (fail-open)", exc_info=True)
        return False


def _nudge(claim: str, *, specific: bool, failed: bool = False, announced: bool = False) -> str:
    if specific:
        extra = " Your claim references a specific result that the tool outputs do not support."
    elif announced:
        # The tool calls this turn did something ADJACENT — saying "no tool call
        # was made" would read as false to the model and invite an argument
        # instead of the action.
        extra = " No tool call in this turn performed the action you announced."
    else:
        extra = ""
    # (b) vs (a): distinguish "you never called anything" from "everything you
    # called errored", so the model corrects the right thing — retrying blindly
    # when the tool is broken is exactly the loop we want to avoid.
    grounding = (
        "every tool call this turn failed, so nothing actually performed it"
        if failed
        else "no corresponding tool call was made this turn"
    )
    return (
        "[System: In your previous message you indicated you completed an action "
        f'("{claim[:120]}") but {grounding}.{extra} '
        "Either perform the action now using the appropriate tool, or clearly state that "
        "you did not/cannot do it and why. Do not report actions as done unless a tool "
        "call actually performed them.]"
    )


def build_larp_nudge(
    *,
    messages: list,
    final_response: str,
    agent: Any = None,
    config: Optional[dict] = None,
    attempts: int = 0,
) -> Optional[str]:
    """Return a corrective re-prompt when the turn LARPed, else None."""
    sec = _section(config)
    if attempts >= int(sec.get("max_reprompts", 2) or 2):
        return None
    text = (final_response or "").strip()
    if not text:
        return None
    found = _first_claim_with_kind(text)
    if not found:
        return None
    claim, kind = found

    exempt = _exempt_tokens(config)
    made, any_success, any_fail = _turn_tool_activity(messages, exempt)

    if made:
        # (b): calls were made but NONE succeeded, so nothing performed the
        # claimed action. Only a past-tense claim that OWNS the failure is honest
        # narration; an intent announcement is a promise no failed tool can keep.
        if _flag(sec, "require_success", True) and not any_success:
            if kind == "intent" or not _acknowledges_failure(text):
                return _nudge(claim, specific=False, failed=any_fail)

        # (d): Tier-2 runs on ANY turn with tool activity. Gating it on
        # any_success would disable it precisely when every tool is failing —
        # the highest-LARP-risk state there is.
        #
        # Intent announcements are routed here too (judge_intent_claims): they
        # are the one shape the deterministic tiers cannot settle, because a
        # SUCCESSFUL but unrelated call satisfies (c). "Terminal pane focused.
        # Now launching the classifier." is grounded by the focus call as far as
        # tier-1 can tell; only a semantic check sees that focusing a pane is not
        # launching anything. Costs one aux call per announcement turn, so it is
        # separately switchable for slow/saturated aux backends.
        judge_intent = kind == "intent" and _flag(sec, "judge_intent_claims", True)
        if _flag(sec, "judge_tier_enabled", False) and (_looks_specific(claim) or judge_intent):
            if _judge_ungrounded(messages, text, claim, kind=kind):
                return _nudge(
                    claim,
                    specific=_looks_specific(claim),
                    failed=any_fail and not any_success,
                    announced=kind == "intent",
                )

        # (c): a successful substantive tool backs the turn -> pass.
        return None

    # (a): an action claim with ZERO substantive tool calls this turn = LARP.
    return _nudge(claim, specific=False)


__all__ = ["larp_detection_enabled", "build_larp_nudge"]
