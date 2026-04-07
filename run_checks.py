"""
Full automated check for OpsEnv — ASCII output only (Windows-safe).
Run: python run_checks.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

results = []

def check(name, fn):
    try:
        fn()
        results.append(("PASS", name, ""))
    except Exception as e:
        results.append(("FAIL", name, f"{type(e).__name__}: {e}"))

# ------------------------------------------------------------------
# T1: No top-level openai import in inference.py
# ------------------------------------------------------------------
def t1():
    with open("inference.py", encoding="utf-8") as f:
        lines = f.readlines()
    hits = [l.strip() for l in lines if l.startswith("from openai") or l.startswith("import openai")]
    assert len(hits) == 0, f"Top-level openai import found: {hits}"
check("T1  inference.py: no top-level openai import", t1)

# ------------------------------------------------------------------
# T2: Core imports succeed
# ------------------------------------------------------------------
def t2():
    from server.environment import OpsEnv
    from models import IncidentAction, IncidentObservation, IncidentState, StepResult
check("T2  imports: server.environment + models", t2)

# ------------------------------------------------------------------
# T3: reset() produces clean initial state
# ------------------------------------------------------------------
def t3():
    from server.environment import OpsEnv
    env = OpsEnv()
    r = env.reset()
    assert r.observation is not None, "observation is None"
    assert r.observation.task_name == "classify_severity", f"wrong task: {r.observation.task_name}"
    assert r.done is False, "done should be False after reset"
    assert r.reward == 0.0, f"reward should be 0.0, got {r.reward}"
check("T3  reset(): clean state, task=classify_severity, reward=0.0", t3)

# ------------------------------------------------------------------
# T4: Full correct episode — INC-001
# ------------------------------------------------------------------
def t4():
    from server.environment import OpsEnv
    from models import IncidentAction
    env = OpsEnv()
    env.reset()  # INC-001

    r1 = env.step(IncidentAction(severity="P1"))
    assert r1.reward == 1.0, f"Task1 exact match should=1.0, got {r1.reward}"

    r2 = env.step(IncidentAction(
        root_cause_service="postgres-payments",
        root_cause_reason="connection pool exhausted waiting=127 connections active=50",
    ))
    assert r2.reward == 1.0, f"Task2 with evidence should=1.0, got {r2.reward} details={r2.info['details']}"

    r3 = env.step(IncidentAction(
        runbook_id="RB-019",
        eta_minutes=18,
        status_update=(
            "We are aware of an issue impacting checkout and payment functionality. "
            "Our team is actively investigating and working to resolve this as quickly as possible. "
            "We will provide updates every 30 minutes."
        ),
    ))
    assert r3.done is True, "Episode should be done after 3 steps"
    assert r3.reward >= 0.65, f"Task3 should be >=0.65, got {r3.reward} details={r3.info['details']}"
check("T4  full correct episode: T1=1.0, T2=1.0, T3>=0.65, done=True", t4)

# ------------------------------------------------------------------
# T5: Weak answers score low
# ------------------------------------------------------------------
def t5():
    from server.environment import OpsEnv
    from models import IncidentAction
    env = OpsEnv()
    env.reset()

    r1 = env.step(IncidentAction(severity="P3"))  # 2 levels off
    assert r1.reward == 0.0, f"Wrong severity(P3 vs P1) should=0.0, got {r1.reward}"

    r2 = env.step(IncidentAction(root_cause_service="payment-svc", root_cause_reason="it is down"))
    assert r2.reward == 0.0, f"Wrong service should=0.0, got {r2.reward}"

    r3 = env.step(IncidentAction(
        runbook_id="RB-007",
        eta_minutes=5,
        status_update="the postgres database connection pool has failed will be fixed in 5 minutes",
    ))
    assert r3.reward < 0.30, f"Weak Task3 should be <0.30, got {r3.reward} details={r3.info['details']}"
check("T5  weak answers: sev=0.0, root=0.0, response<0.30", t5)

# ------------------------------------------------------------------
# T6: Leakage penalty fires
# ------------------------------------------------------------------
def t6():
    from server.environment import OpsEnv
    from models import IncidentAction
    env = OpsEnv(); env.reset()
    env.step(IncidentAction(severity="P1"))
    env.step(IncidentAction(root_cause_service="postgres-payments", root_cause_reason="x"))
    r = env.step(IncidentAction(
        runbook_id="RB-019", eta_minutes=18,
        status_update="The postgres cluster connection pool is exhausted and we are investigating now.",
    ))
    assert r.info["details"].get("comms_no_leakage") is False, \
        f"Leakage not detected. details={r.info['details']}"
check("T6  leakage penalty: 'postgres'/'cluster' detected", t6)

# ------------------------------------------------------------------
# T7: Over-promise penalty fires
# ------------------------------------------------------------------
def t7():
    from server.environment import OpsEnv
    from models import IncidentAction
    env = OpsEnv(); env.reset()
    env.step(IncidentAction(severity="P1"))
    env.step(IncidentAction(root_cause_service="postgres-payments", root_cause_reason="x"))
    r = env.step(IncidentAction(
        runbook_id="RB-019", eta_minutes=18,
        status_update="We are investigating the checkout issue and it will be resolved in 15 minutes.",
    ))
    assert r.info["details"].get("comms_over_promise_penalty") is True, \
        f"Over-promise not detected. details={r.info['details']}"
check("T7  over-promise penalty: 'resolved in 15 minutes'", t7)

# ------------------------------------------------------------------
# T8: Jargon penalty fires
# ------------------------------------------------------------------
def t8():
    from server.environment import OpsEnv
    from models import IncidentAction
    env = OpsEnv(); env.reset()
    env.step(IncidentAction(severity="P1"))
    env.step(IncidentAction(root_cause_service="postgres-payments", root_cause_reason="x"))
    r = env.step(IncidentAction(
        runbook_id="RB-019", eta_minutes=18,
        status_update="Our RCA shows checkout functionality is impacted and we are reviewing metrics and SLO compliance.",
    ))
    assert r.info["details"].get("comms_jargon_penalty") is True, \
        f"Jargon not detected. details={r.info['details']}"
check("T8  jargon penalty: 'RCA'/'metrics'/'SLO'", t8)

# ------------------------------------------------------------------
# T9: state() does not advance episode
# ------------------------------------------------------------------
def t9():
    from server.environment import OpsEnv
    env = OpsEnv(); env.reset()
    s1 = env.state()
    s2 = env.state()
    assert s1.step_count == s2.step_count == 0, f"step_count changed: {s1.step_count} -> {s2.step_count}"
    assert s1.current_task_name == "classify_severity"
check("T9  state() does not advance episode", t9)

# ------------------------------------------------------------------
# T10: openenv.yaml has all required keys
# ------------------------------------------------------------------
def t10():
    import yaml
    with open("openenv.yaml", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    for key in ("name", "version", "description", "tasks", "action_space", "observation_space", "env_vars"):
        assert key in y, f"Missing key in openenv.yaml: {key}"
    assert len(y["tasks"]) >= 3, f"Need >= 3 tasks, got {len(y['tasks'])}"
    diffs = {t["difficulty"] for t in y["tasks"]}
    assert "easy" in diffs, "No 'easy' difficulty task"
    assert "hard" in diffs, "No 'hard' difficulty task"
check("T10 openenv.yaml: all required keys, 3+ tasks, easy+hard", t10)

# ------------------------------------------------------------------
# T11: FastAPI routes registered
# ------------------------------------------------------------------
def t11():
    from server.app import app
    routes = [r.path for r in app.routes]
    for required in ["/health", "/reset", "/step", "/state", "/tasks"]:
        assert required in routes, f"Missing FastAPI route: {required}"
check("T11 FastAPI: /health /reset /step /state /tasks registered", t11)

# ------------------------------------------------------------------
# T12: Adjacent severity scoring (1 level off = 0.35)
# ------------------------------------------------------------------
def t12():
    from server.environment import OpsEnv
    from models import IncidentAction
    env = OpsEnv(); env.reset()  # INC-001 true=P1
    r = env.step(IncidentAction(severity="P0"))  # 1 level off
    assert r.reward == 0.35, f"Adjacent sev should=0.35, got {r.reward}"
check("T12 adjacent severity = 0.35 (not 0.5)", t12)

# ------------------------------------------------------------------
# T13: grade_task() API
# ------------------------------------------------------------------
def t13():
    from server.environment import OpsEnv
    from models import IncidentAction
    env = OpsEnv(); env.reset()
    env.step(IncidentAction(severity="P1"))
    g = env.grade_task("classify_severity")
    assert g.graded is True, "grade_task() should return graded=True"
    assert g.score == 1.0, f"grade_task() score should=1.0, got {g.score}"
    g2 = env.grade_task("execute_response")
    assert g2.graded is False, "grade_task not-yet-completed should be graded=False"
check("T13 grade_task() API correct", t13)

# ------------------------------------------------------------------
# T14: Family match (same family, wrong service) = 0.40
# ------------------------------------------------------------------
def t14():
    from server.environment import OpsEnv
    from models import IncidentAction
    env = OpsEnv(); env.reset()  # INC-001 true=postgres-payments (database family)
    env.step(IncidentAction(severity="P1"))
    r = env.step(IncidentAction(root_cause_service="mysql-payments", root_cause_reason="database issue"))
    assert r.reward == 0.40, f"Family match should=0.40, got {r.reward} details={r.info['details']}"
check("T14 family match (mysql vs postgres) = 0.40", t14)

# ------------------------------------------------------------------
# T15: done episode step returns 0.0 reward
# ------------------------------------------------------------------
def t15():
    from server.environment import OpsEnv
    from models import IncidentAction
    env = OpsEnv(); env.reset()
    env.step(IncidentAction(severity="P1"))
    env.step(IncidentAction(root_cause_service="postgres-payments", root_cause_reason="x"))
    env.step(IncidentAction(runbook_id="RB-019", eta_minutes=18, status_update="We are investigating the checkout and payment issue. Team is working to resolve it as quickly as possible. Updates will follow."))
    # Episode done — extra step should return 0
    r = env.step(IncidentAction(severity="P0"))
    assert r.reward == 0.0, f"Step after done should=0.0, got {r.reward}"
check("T15 step() after episode done returns reward=0.0", t15)

# ------------------------------------------------------------------
# Print results
# ------------------------------------------------------------------
print()
print("=" * 65)
print("OpsEnv Full Check Results")
print("=" * 65)
passed = 0
failed = 0
for status, name, err in results:
    if status == "PASS":
        print(f"  [PASS]  {name}")
        passed += 1
    else:
        print(f"  [FAIL]  {name}")
        print(f"          {err}")
        failed += 1
print("=" * 65)
print(f"Result: {passed} passed, {failed} failed out of {passed+failed} checks")
if failed == 0:
    print("ALL CHECKS PASSED -- safe to submit")
else:
    print("SOME CHECKS FAILED -- review above")
print("=" * 65)
sys.exit(0 if failed == 0 else 1)
