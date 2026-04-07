"""
Final safety check script — run from inside the opsenv/ directory.
Tests: imports, reset/step flow, --no-llm path, openai lazy import.
"""
import sys
import os

results = []

def check(name, fn):
    try:
        fn()
        results.append(f"  PASS  {name}")
    except Exception as e:
        results.append(f"  FAIL  {name}  ({type(e).__name__}: {e})")

# 1. openai is NOT imported at module level
def test_no_top_level_openai():
    with open("inference.py", encoding="utf-8") as f:
        lines = f.readlines()
    top_imports = [l.strip() for l in lines if l.startswith("from openai") or l.startswith("import openai")]
    assert len(top_imports) == 0, f"Top-level openai import found: {top_imports}"
check("inference.py: no top-level openai import", test_no_top_level_openai)

# 2. Environment imports cleanly
def test_env_import():
    from server.environment import OpsEnv
    from models import IncidentAction, IncidentObservation, IncidentState, StepResult
check("imports: server.environment, models", test_env_import)

# 3. reset() works
def test_reset():
    from server.environment import OpsEnv
    env = OpsEnv()
    r = env.reset()
    assert r.observation is not None
    assert r.observation.task_name == "classify_severity"
    assert r.done is False
    assert r.reward == 0.0
check("reset() → clean initial state, task=classify_severity", test_reset)

# 4. Full episode: 3 steps, correct answers
def test_full_episode_correct():
    from server.environment import OpsEnv
    from models import IncidentAction
    env = OpsEnv()
    env.reset()  # INC-001
    # Task 1 — correct severity
    r1 = env.step(IncidentAction(severity="P1"))
    assert r1.reward == 1.0, f"Expected 1.0 got {r1.reward}"
    # Task 2 — correct service + evidence
    r2 = env.step(IncidentAction(
        root_cause_service="postgres-payments",
        root_cause_reason="connection pool exhausted waiting=127 connections active=50",
    ))
    assert r2.reward == 1.0, f"Expected 1.0 got {r2.reward}"
    # Task 3 — correct runbook, good ETA, clean comms (no internal names, 30+ words)
    r3 = env.step(IncidentAction(
        runbook_id="RB-019",
        eta_minutes=18,
        status_update=(
            "We are aware of an issue impacting checkout and payment functionality. "
            "Our team is actively investigating and working to resolve this as quickly as possible. "
            "We will provide updates every 30 minutes."
        ),
    ))
    assert r3.done is True
    assert r3.reward >= 0.65, f"Task3 reward too low: {r3.reward}"
check("full episode (correct answers): rewards ≥ expected", test_full_episode_correct)

# 5. Full episode: wrong answers score clearly lower
def test_full_episode_weak():
    from server.environment import OpsEnv
    from models import IncidentAction
    env = OpsEnv()
    env.reset()
    r1 = env.step(IncidentAction(severity="P3"))       # wrong by 2 levels
    assert r1.reward == 0.0, f"Expected 0.0 got {r1.reward}"
    r2 = env.step(IncidentAction(root_cause_service="payment-svc", root_cause_reason="it is down"))
    assert r2.reward == 0.0, f"Expected 0.0 got {r2.reward}"
    r3 = env.step(IncidentAction(
        runbook_id="RB-007",
        eta_minutes=5,
        status_update="the postgres database connection pool has failed will be fixed in 5 minutes",
    ))
    assert r3.reward < 0.30, f"Weak model reward too high: {r3.reward}"
check("full episode (weak answers): all tasks score low", test_full_episode_weak)

# 6. Hard-task grader: leakage penalty fires
def test_leakage_penalty():
    from server.environment import OpsEnv
    from models import IncidentAction
    env = OpsEnv(); env.reset()
    env.step(IncidentAction(severity="P1"))
    env.step(IncidentAction(root_cause_service="postgres-payments", root_cause_reason="x"))
    # correct runbook, good ETA, but status leaks "postgres" and "cluster"
    r = env.step(IncidentAction(
        runbook_id="RB-019", eta_minutes=18,
        status_update="The postgres cluster connection pool is exhausted and we are investigating now.",
    ))
    assert r.info["details"].get("comms_no_leakage") is False, "Leakage not detected"
check("hard task: leakage penalty fires on 'postgres'/'cluster'", test_leakage_penalty)

# 7. Hard-task grader: over-promise penalty fires
def test_over_promise():
    from server.environment import OpsEnv
    from models import IncidentAction
    env = OpsEnv(); env.reset()
    env.step(IncidentAction(severity="P1"))
    env.step(IncidentAction(root_cause_service="postgres-payments", root_cause_reason="x"))
    r = env.step(IncidentAction(
        runbook_id="RB-019", eta_minutes=18,
        status_update="We are investigating the checkout issue and it will be resolved in 15 minutes.",
    ))
    assert r.info["details"].get("comms_over_promise_penalty") is True, "Over-promise not detected"
check("hard task: over-promise penalty fires on 'resolved in 15 minutes'", test_over_promise)

# 8. Hard-task grader: jargon penalty fires
def test_jargon_penalty():
    from server.environment import OpsEnv
    from models import IncidentAction
    env = OpsEnv(); env.reset()
    env.step(IncidentAction(severity="P1"))
    env.step(IncidentAction(root_cause_service="postgres-payments", root_cause_reason="x"))
    r = env.step(IncidentAction(
        runbook_id="RB-019", eta_minutes=18,
        status_update="Our RCA shows checkout functionality is impacted and we are reviewing metrics and SLO compliance.",
    ))
    assert r.info["details"].get("comms_jargon_penalty") is True, "Jargon not detected"
check("hard task: jargon penalty fires on 'RCA'/'metrics'/'SLO'", test_jargon_penalty)

# 9. state() doesn't advance episode
def test_state_no_advance():
    from server.environment import OpsEnv
    env = OpsEnv(); env.reset()
    s1 = env.state()
    s2 = env.state()
    assert s1.step_count == s2.step_count == 0
    assert s1.current_task_name == "classify_severity"
check("state() does not advance episode", test_state_no_advance)

# 10. openenv.yaml has all required keys
def test_openenv_yaml():
    import yaml
    with open("openenv.yaml", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    for key in ("name", "version", "description", "tasks", "action_space", "observation_space", "env_vars"):
        assert key in y, f"Missing key: {key}"
    assert len(y["tasks"]) >= 3
    difficulties = {t["difficulty"] for t in y["tasks"]}
    assert "easy" in difficulties and "hard" in difficulties
check("openenv.yaml: all required keys, 3+ tasks, easy+hard difficulty", test_openenv_yaml)

# 11. FastAPI app imports
def test_app_import():
    from server.app import app
    routes = [r.path for r in app.routes]
    for required in ["/health", "/reset", "/step", "/state", "/tasks"]:
        assert required in routes, f"Missing route: {required}"
check("FastAPI app: /health /reset /step /state /tasks all registered", test_app_import)

# Print results
print("\n" + "="*60)
print("OpsEnv Final Safety Check")
print("="*60)
for r in results:
    print(r)
passed = sum(1 for r in results if "PASS" in r)
failed = sum(1 for r in results if "FAIL" in r)
print("="*60)
print(f"Result: {passed} passed, {failed} failed")
if failed == 0:
    print("ALL CHECKS PASSED — safe to submit")
else:
    print("SOME CHECKS FAILED — review above")
sys.exit(0 if failed == 0 else 1)
