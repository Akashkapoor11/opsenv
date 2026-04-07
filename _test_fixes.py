"""
Quick test: run all 3 episodes in no-llm mode and also simulate
the exact fix for dns-resolver disambiguation + ETA overrides.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from server.environment import OpsEnv, _incidents
from models import IncidentAction

# ── Test 1: Verify the ETA lookup table is imported correctly ─────────────
from inference import _ROOT_CAUSE_ETA, _fallback_severity, _fallback_root_cause, _fallback_response

print("=== ETA Lookup Table ===")
for svc, eta in _ROOT_CAUSE_ETA.items():
    print(f"  {svc}: {eta}min")

# ── Test 2: simulate episodes with corrected root_cause values ───────────
env = OpsEnv()
incidents = _incidents()

# Only test the 3 episodes from the question
test_cases = [
    # (episode_index, true_svc, true_severity, correct_runbook, correct_eta)
    (0, "postgres-payments", "P1", "RB-019", 18),
    (1, "redis-sessions",    "P1", "RB-011", 10),
    (2, "dns-resolver",      "P0", "RB-016", 25),
]

print("\n=== Simulating Episodes with Corrected Answers ===")
total_all = 0.0
for ep_idx, true_svc, true_sev, correct_rb, correct_eta in test_cases:
    env._episode_counter = ep_idx  # set to correct incident
    result = env.reset()
    inc = incidents[ep_idx]
    ep_total = 0.0

    print(f"\n--- Episode {ep_idx+1}: {inc.title} ---")

    # Task 1: classify_severity - use true answer
    r1 = env.step(IncidentAction(severity=true_sev))
    ep_total += r1.reward
    print(f"  classify_severity: {true_sev} -> reward={r1.reward} | {r1.info.get('details',{}).get('match','')}")

    # Task 2: identify_root_cause - use true answer with evidence
    evidence = inc.evidence_signals[0] if inc.evidence_signals else "log evidence"
    r2 = env.step(IncidentAction(
        root_cause_service=true_svc,
        root_cause_reason=f"Evidence: {evidence} from logs/metrics"
    ))
    ep_total += r2.reward
    print(f"  identify_root_cause: {true_svc} -> reward={r2.reward} | {r2.info.get('details',{}).get('service_match','')}")

    # Task 3: execute_response - use correct runbook + ETA from lookup
    actual_eta = _ROOT_CAUSE_ETA.get(true_svc, correct_eta)
    # Use a good status update (P0 or P1 appropriate)
    if true_sev == "P0":
        status = (
            "We are experiencing a critical outage affecting all users and all services. "
            "Our team is urgently working to restore full access. "
            "We will provide updates every 15 minutes and are immediately addressing the issue."
        )
    else:
        feature = "login" if "login" in inc.title.lower() else "checkout"
        status = (
            f"We are aware of an issue impacting {feature} for a significant number of users. "
            f"Our team is actively investigating and working to resolve this as quickly as possible. "
            f"We will provide updates as more information becomes available."
        )
    r3 = env.step(IncidentAction(
        runbook_id=correct_rb,
        eta_minutes=actual_eta,
        status_update=status
    ))
    ep_total += r3.reward
    d3 = r3.info.get('details', {})
    print(f"  execute_response: {correct_rb} eta={actual_eta}m -> reward={r3.reward}")
    print(f"    runbook={d3.get('runbook','?')} eta={d3.get('eta','?')}")
    print(f"    comms_impact={d3.get('comms_impact_area','?')} leakage={d3.get('comms_no_leakage','?')} tone={d3.get('comms_tone','?')}")

    print(f"  >> Episode reward: {ep_total:.2f} / 3.00  ({ep_total/3:.1%})")
    total_all += ep_total

print(f"\n=== Average: {total_all/3:.2f} / 3.00  ({total_all/9:.1%}) ===")
