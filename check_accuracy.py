"""
OpsEnv Accuracy Checker
=======================
Runs all 12 incidents through the agent, scores each task, and prints a
detailed breakdown.

Run (rule-based, no API key needed):
  python check_accuracy.py

Run (LLM mode — uses HF_TOKEN / OPENAI_API_KEY):
  python check_accuracy.py --llm
  python check_accuracy.py --llm --model meta-llama/Llama-3.1-8B-Instruct
"""
import sys, os, argparse

# Make sure project root is on path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from server.environment import OpsEnv, _incidents
from models import IncidentAction

# ── Import helpers from inference.py ─────────────────────────────────────────
from inference import (
    _fallback_severity,
    _fallback_root_cause,
    _fallback_response,
    llm_classify_severity,
    llm_identify_root_cause,
    llm_execute_response,
    _ROOT_CAUSE_ETA,
    API_KEY,
    MODEL_NAME,
)

HEADER = "=" * 80

def score_label(s: float) -> str:
    if s >= 1.0:   return "✅ PERFECT"
    if s >= 0.80:  return "🟢 GOOD"
    if s >= 0.50:  return "🟡 PARTIAL"
    if s >= 0.35:  return "🟠 ADJACENT"
    return             "❌ WRONG"


def classify_severity(obs_dict, use_llm):
    if use_llm:
        try:
            return llm_classify_severity(obs_dict)
        except Exception as e:
            print(f"  [WARN] LLM severity failed ({e.__class__.__name__}): using fallback", flush=True)
            return _fallback_severity(obs_dict)
    return _fallback_severity(obs_dict)


def identify_root_cause(obs_dict, use_llm):
    if use_llm:
        try:
            return llm_identify_root_cause(obs_dict)
        except Exception as e:
            print(f"  [WARN] LLM root cause failed ({e.__class__.__name__}): using fallback", flush=True)
            return _fallback_root_cause(obs_dict)
    return _fallback_root_cause(obs_dict)


def execute_response(obs_dict, root_cause_svc, severity, use_llm):
    if use_llm:
        try:
            return llm_execute_response(obs_dict, root_cause_svc, severity)
        except Exception as e:
            print(f"  [WARN] LLM response failed ({e.__class__.__name__}): using fallback", flush=True)
            return _fallback_response(obs_dict, root_cause_svc, severity)
    return _fallback_response(obs_dict, root_cause_svc, severity)


def run(use_llm: bool = False):
    incidents = _incidents()

    mode_label = f"LLM ({MODEL_NAME})" if use_llm else "Rule-Based Fallback"

    print(HEADER)
    print(f"  OpsEnv Accuracy Report — All 12 Incidents  [{mode_label}]")
    print(HEADER)

    if use_llm and not API_KEY:
        print("  ⚠️  WARNING: No HF_TOKEN or OPENAI_API_KEY found.")
        print("  Set the env var and retry, or run without --llm for rule-based mode.")
        sys.exit(1)

    totals = {"severity": [], "root_cause": [], "response": [], "episode": []}

    for inc_data in incidents:
        # ── Build a fresh episode using this specific incident ────────────
        env2 = OpsEnv([inc_data])
        result = env2.reset()

        obs_dict = result.observation.to_dict()

        print(f"\n{inc_data.incident_id} │ {inc_data.title[:62]}")
        print(f"  True:  sev={inc_data.true_severity}  root={inc_data.true_root_cause_service}")

        # Task 1 — classify_severity
        severity_pred = classify_severity(obs_dict, use_llm)
        action1 = IncidentAction(severity=severity_pred)
        r1 = env2.step(action1)
        score_sev = round(r1.reward, 3)
        detail_sev = r1.info.get("details", {})

        # Task 2 — identify_root_cause
        obs_dict2 = r1.observation.to_dict()
        rc = identify_root_cause(obs_dict2, use_llm)
        action2 = IncidentAction(
            root_cause_service=rc["root_cause_service"],
            root_cause_reason=rc["root_cause_reason"],
        )
        r2 = env2.step(action2)
        score_rc = round(r2.reward, 3)
        detail_rc = r2.info.get("details", {})

        # Task 3 — execute_response
        obs_dict3 = r2.observation.to_dict()
        resp = execute_response(obs_dict3, rc["root_cause_service"], severity_pred, use_llm)
        action3 = IncidentAction(
            runbook_id=resp["runbook_id"],
            eta_minutes=resp["eta_minutes"],
            status_update=resp["status_update"],
        )
        r3 = env2.step(action3)
        score_resp = round(r3.reward, 3)
        detail_resp = r3.info.get("details", {})

        episode_total = round(score_sev + score_rc + score_resp, 3)

        totals["severity"].append(score_sev)
        totals["root_cause"].append(score_rc)
        totals["response"].append(score_resp)
        totals["episode"].append(episode_total)

        # ── Print incident block ─────────────────────────────────────────
        print(f"  Pred:  sev={severity_pred}  root={rc['root_cause_service']}")
        print()
        print(f"  Task 1 Severity    {score_sev:.2f}  {score_label(score_sev)}"
              f"  [{detail_sev.get('match','?')}]")
        print(f"  Task 2 Root Cause  {score_rc:.2f}  {score_label(score_rc)}"
              f"  [{detail_rc.get('service_match','?')}]"
              + ("" if detail_rc.get("evidence_cited", "?") == "?" else
                 f"  evidence_cited={detail_rc.get('evidence_cited')}"))
        print(f"  Task 3 Response    {score_resp:.2f}  {score_label(score_resp)}")

        # Response breakdown
        rb_info      = detail_resp.get("runbook", "?")
        eta_info     = detail_resp.get("eta", "?")
        comms_score  = detail_resp.get("comms_score", 0)
        leakage      = detail_resp.get("comms_no_leakage", "?")
        tone         = detail_resp.get("comms_tone", "?")
        words        = detail_resp.get("status_word_count", 0)
        print(f"    runbook={rb_info}")
        print(f"    eta={eta_info}")
        print(f"    comms score={comms_score:.3f}  words={words}  leakage_free={leakage}  tone={tone}")
        if detail_resp.get("comms_over_promise_penalty"):
            print(f"    ⚠️  over-promise penalty applied (-0.15)")
        if detail_resp.get("comms_jargon_penalty"):
            print(f"    ⚠️  jargon penalty applied (-0.08): {detail_resp.get('comms_jargon_note','')}")
        if detail_resp.get("comms_leaked_terms"):
            print(f"    ⚠️  leaked terms: {detail_resp.get('comms_leaked_terms')}")
        print(f"  ───  Episode total: {episode_total:.2f} / 3.00  ({episode_total/3:.0%})")

    # ── Summary table ────────────────────────────────────────────────────
    n = len(incidents)
    avg_sev  = sum(totals["severity"])   / n
    avg_rc   = sum(totals["root_cause"]) / n
    avg_resp = sum(totals["response"])   / n
    avg_ep   = sum(totals["episode"])    / n

    print("\n" + HEADER)
    print("  SUMMARY")
    print(HEADER)
    print(f"  Mode: {mode_label}")
    print(f"  Task 1 — Severity      avg = {avg_sev:.3f} / 1.000  ({avg_sev:.1%})")
    print(f"  Task 2 — Root Cause    avg = {avg_rc:.3f} / 1.000  ({avg_rc:.1%})")
    print(f"  Task 3 — Response      avg = {avg_resp:.3f} / 1.000  ({avg_resp:.1%})")
    print(f"  Episode total          avg = {avg_ep:.3f} / 3.000  ({avg_ep/3:.1%})")
    print()

    # Per-incident totals table
    print("  Per-incident totals:")
    print("  " + "-" * 50)
    print(f"  {'ID':<8} {'Sev':>5} {'RC':>6} {'Resp':>6} {'Total':>7} {'%':>6}")
    print("  " + "-" * 50)
    for i, inc_data in enumerate(incidents):
        s   = totals["severity"][i]
        r   = totals["root_cause"][i]
        re_ = totals["response"][i]
        t   = totals["episode"][i]
        print(f"  {inc_data.incident_id:<8} {s:>5.2f} {r:>6.2f} {re_:>6.2f} {t:>7.2f} {t/3:>6.1%}")
    print("  " + "-" * 50)
    print(f"  {'AVG':<8} {avg_sev:>5.2f} {avg_rc:>6.2f} {avg_resp:>6.2f} {avg_ep:>7.2f} {avg_ep/3:>6.1%}")
    print(HEADER)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpsEnv Accuracy Checker")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM inference (requires HF_TOKEN or OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override MODEL_NAME env var (e.g. meta-llama/Llama-3.1-8B-Instruct)",
    )
    args = parser.parse_args()

    if args.model:
        import inference as _inf
        _inf.MODEL_NAME = args.model
        MODEL_NAME = args.model  # local ref for display

    run(use_llm=args.llm)
