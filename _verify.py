"""
Standalone verifier: simulates all 3 episodes with the OPTIMAL answers
(as an oracle) and writes the scores to _verify_out.txt.
Also tests the _normalize_service and _ROOT_CAUSE_ETA functions.
"""
import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

output = io.StringIO()

def pr(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, file=output, **kwargs)

try:
    from server.environment import OpsEnv, _incidents
    from models import IncidentAction
    from inference import _ROOT_CAUSE_ETA, _SERVICE_NORMALIZE, _normalize_service
    pr("[OK] Imports successful")
except Exception as e:
    pr(f"[ERROR] Import failed: {e}")
    with open("_verify_out.txt", "w") as f:
        f.write(output.getvalue())
    sys.exit(1)

# ── Test normalization map ────────────────────────────────────────────────
pr("\n=== Service Normalization Tests ===")
tests = [
    ("coredns",     "dns-resolver"),
    ("dns-resolver","dns-resolver"),
    ("redis",       "redis-sessions"),
    ("postgres",    "postgres-payments"),
    ("kafka",       "kafka"),
    ("memcached",   "memcached"),
]
all_ok = True
for inp, expected in tests:
    got = _normalize_service(inp)
    status = "OK" if got == expected else f"FAIL (got {got!r})"
    pr(f"  normalize({inp!r}) = {got!r}  [{status}]")
    if got != expected:
        all_ok = False

# ── Test ETA lookup ───────────────────────────────────────────────────────
pr("\n=== ETA Lookup Tests ===")
eta_tests = [
    ("postgres-payments", 18),
    ("redis-sessions",    10),
    ("dns-resolver",      25),
    ("coredns",           25),
    ("memcached",         15),
    ("kafka",             12),
]
for svc, expected in eta_tests:
    got = _ROOT_CAUSE_ETA.get(svc, 0)
    status = "OK" if got == expected else f"FAIL (got {got})"
    pr(f"  ETA[{svc}] = {got}min  [{status}]")

# ── Simulate episodes with oracle answers ─────────────────────────────────
pr("\n=== Episode Simulation (Oracle Answers) ===")
incidents = _incidents()

# The first 3 incidents match ep1, ep2, ep3 in the original run
episode_configs = [
    # (inc_idx, true_svc,           true_sev, correct_rb, correct_eta, feature)
    (0, "postgres-payments", "P1", "RB-019", 18, "checkout payment"),
    (1, "redis-sessions",    "P1", "RB-011", 10, "login authentication"),
    (2, "dns-resolver",      "P0", "RB-016", 25, "all services all users"),
]

grand_total = 0.0
env = OpsEnv()

for ep_num, (inc_idx, true_svc, true_sev, correct_rb, correct_eta, feature_words) in enumerate(episode_configs, 1):
    # Force exact incident
    env._episode_counter = inc_idx
    result = env.reset()
    inc = incidents[inc_idx]
    ep_total = 0.0
    pr(f"\n--- Episode {ep_num}: {inc.title} ---")

    # Task 1: classify_severity
    r1 = env.step(IncidentAction(severity=true_sev))
    ep_total += r1.reward
    d1 = r1.info.get("details", {})
    pr(f"  classify_severity:    {true_sev} → reward={r1.reward:.2f} ({d1.get('match','')})")

    # Task 2: identify_root_cause (with evidence from signals)
    evidence = inc.evidence_signals[0] if inc.evidence_signals else "key metric"
    reason = f"Evidence: {evidence} confirms {true_svc} is the root cause"
    r2 = env.step(IncidentAction(root_cause_service=true_svc, root_cause_reason=reason))
    ep_total += r2.reward
    d2 = r2.info.get("details", {})
    pr(f"  identify_root_cause:  {true_svc} → reward={r2.reward:.2f} ({d2.get('service_match','')})")
    if d2.get("evidence_cited"):
        pr(f"    evidence_terms_found: {d2.get('evidence_terms_found','')}")

    # Task 3: execute_response — use ETA from lookup table
    actual_eta = _ROOT_CAUSE_ETA.get(true_svc, correct_eta)

    if true_sev == "P0":
        status_msg = (
            "We are experiencing a critical outage affecting all users and all services are currently unavailable. "
            "Our team is urgently working to restore full access immediately. "
            "We will provide updates every 15 minutes and are immediately addressing the issue."
        )
    else:
        status_msg = (
            f"We are aware of an issue impacting {feature_words} for a significant number of users. "
            "Our team is actively investigating and working to resolve this as quickly as possible. "
            "We will continue to provide updates as more information becomes available."
        )

    r3 = env.step(IncidentAction(runbook_id=correct_rb, eta_minutes=actual_eta, status_update=status_msg))
    ep_total += r3.reward
    d3 = r3.info.get("details", {})
    pr(f"  execute_response:     {correct_rb} eta={actual_eta}m → reward={r3.reward:.2f}")
    pr(f"    runbook={d3.get('runbook','?')}  eta={d3.get('eta','?')}")
    pr(f"    impact_area={d3.get('comms_impact_area','?')}  no_leakage={d3.get('comms_no_leakage','?')}  tone={d3.get('comms_tone','?')}")
    if d3.get("comms_over_promise_penalty"):
        pr(f"    [PENALTY] over-promise detected!")
    if d3.get("comms_jargon_penalty"):
        pr(f"    [PENALTY] jargon detected!")
    if not d3.get("comms_no_leakage"):
        pr(f"    [PENALTY] leaked terms: {d3.get('comms_leaked_terms','')}")

    pr(f"  >> Episode {ep_num} reward: {ep_total:.2f} / 3.00  ({ep_total/3:.1%})")
    grand_total += ep_total

pr(f"\n{'='*60}")
pr(f"Grand total: {grand_total:.2f} / 9.00  (avg: {grand_total/3:.2f}/3.00 = {grand_total/9:.1%})")
pr(f"{'='*60}")

# Write to file
with open("_verify_out.txt", "w") as f:
    f.write(output.getvalue())

pr("\n[Done] Results written to _verify_out.txt")
