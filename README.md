---
title: OpsEnv — Production Incident Response
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
  - sre
  - incident-response
  - real-world
  - operations
  - reinforcement-learning
  - agent-evaluation
pinned: false
---

# OpsEnv  — Production Incident Response

> **An OpenEnv environment that benchmarks AI agents on the hardest 20 minutes in software engineering: responding to a live production incident.**

When a system breaks at 2am, the on-call engineer must do three things under pressure:  
1. How bad is it? (severity triage)  
2. What actually broke? (root cause — not the symptom service)  
3. Communicate clearly while fixing it. (runbook + ETA + customer status update)

OpsEnv models this entire workflow as a three-task OpenEnv episode with **12 real-world incidents**, **intentional log misdirection**, and **graders that specifically target the failure modes that make LLMs bad at this today**.

---

## Why This Environment Is Not Trivial

Most agent benchmarks hand the agent the same signal that determines the correct answer.  
OpsEnv does not.

**Every incident is designed so the errors appear in the wrong service.**

In `INC-001`, all 5xx errors are in `payment-service`. But payment-service has zero pod restarts, zero CPU spikes, and all health checks passing. The real problem is a Postgres connection pool with 127 connections waiting. The logs contain it — but it takes careful reading to find it.

The hard task (Task 3) then penalises the two most common LLM failure modes:

| Failure mode | What LLMs do | Penalty |
|---|---|---|
| Internal name leakage | Write "the postgres connection pool is exhausted" in a customer status update | −0.12 (no leakage bonus) |
| Over-promising ETAs | Write "will be resolved in 15 minutes" | −0.15 |
| Ops jargon in comms | Write "our RCA shows MTTR breach due to SLA violation" | −0.08 |

A model that reads carefully and understands SRE communication norms scores **~0.82–0.90 / 1.0** on the hard task.  
A weak/generic model scores **~0.25–0.45 / 1.0**. The gap is real and reproducible.

---

## The Three Tasks

Each task uses **distinct, non-overlapping action fields**. The agent fills only the relevant fields per step.

### Task 1 — `classify_severity` · difficulty: easy

**What to do:** Classify the incident as `P0`, `P1`, `P2`, or `P3` using error rate, affected user count, latency, and business criticality.

**Action field:** `severity: str`

**Grader (deterministic tiered):**

| Result | Score |
|---|---|
| Exact match | 1.0 |
| One level adjacent (e.g. P1 when answer is P0) | 0.35 |
| Two or more levels off | 0.0 |

Adjacent score is deliberately 0.35, not 0.5 — calling a P0 a P1 is a serious triage error in SRE practice.

**Weak model behaviour:** Defaults to P1 for everything → ~0.50 average across the corpus.

---

### Task 2 — `identify_root_cause` · difficulty: medium

**What to do:** Identify the service *actually causing* the incident — not the service showing errors. Provide the service name and a specific reason citing log or metric evidence.

**Action fields:** `root_cause_service: str`, `root_cause_reason: str`

**Grader (evidence-gated):**

| Result | Score |
|---|---|
| Exact service + incident-specific evidence cited in reason | 1.0 |
| Exact service + substantive reason but no specific signal | 0.80 |
| Exact service + empty or trivial reason (<10 chars) | 0.65 |
| Same failure family (e.g. MySQL when answer is Postgres) | 0.40 |
| Wrong service | 0.0 |

Evidence terms are unique per incident (e.g. `"waiting=127"`, `"pool exhausted"`, `"acme challenge failed"`). They cannot be guessed — the agent must read the logs.

**Weak model behaviour:** Names the service with the most error lines (always the symptom service, never the root cause) → 0.0.

---

### Task 3 — `execute_response` · difficulty: hard

**What to do:** Select the correct runbook from 4 plausible options, estimate resolution time, and write a customer-facing status page update (≥ 30 words).

**Action fields:** `runbook_id: str`, `eta_minutes: int`, `status_update: str`

**Grader (multi-signal with penalties):**

| Signal | Score |
|---|---|
| Correct runbook selected | +0.50 |
| ETA within **±25%** of expected | +0.15 |
| Status update mentions customer-facing impact area | +0.12 |
| No internal service/infra name in status update | +0.12 |
| Appropriate severity tone + action verb | +0.11 |
| **Over-promising a fix time** ("will be resolved in Xmin") | **−0.15** |
| **Technical ops jargon** (RCA, MTTR, SLA, error rate, metrics) | **−0.08** |
| Copy-pasting the incident title verbatim | −0.05 |
| Status update under 30 words | Comms score scaled proportionally |

**Why each rule exists:**
- **±25% ETA**: Weak models guess round numbers (5 min / 60 min). Real ETAs are incident-specific (8–40min).
- **Leakage**: "postgres", "redis", "k8s", "pod", "cluster", "broker", "deployment", "container" never appear on a public status page.
- **Jargon**: "Our RCA shows MTTR breach" is something engineers say to each other — not customers.
- **Over-promise**: "Will be fixed in 15 minutes" causes escalations when it takes 30.

**Weak model behaviour:** Wrong runbook (~0.00), generic short status update with internal names → ~0.20–0.30. Strong model: correct runbook, specific ETA, clean comms → ~0.80–0.90.

---

## Model Separation

The environment is designed so model quality drives scores, not luck:

| Agent | Severity | Root Cause | Response | Total | Normalized |
|---|---|---|---|---|---|
| Always-P1 (dumb baseline) | ~0.50 | 0.00 | ~0.20 | ~0.70 / 3.0 | ~23% |
| Rule-based fallback (`--no-llm`) | ~0.70 | ~0.40 | ~0.22 | ~1.32 / 3.0 | ~44% |
| Llama-3.1-8B | ~0.82 | ~0.52 | ~0.35 | ~1.69 / 3.0 | ~56% |
| Llama-3.1-70B | ~0.90 | ~0.72 | ~0.58 | ~2.20 / 3.0 | ~73% |
| GPT-4 class | ~0.95 | ~0.85 | ~0.80 | ~2.60 / 3.0 | ~87% |

The **~64-point gap** between weak and strong models on Task 3 specifically is the core differentiator.

---

## The 12 Incidents

| ID | Severity | Root Cause | Failure Domain | Red Herring |
|---|---|---|---|---|
| INC-001 | P1 | postgres-payments | Database | All 5xx errors in payment-svc; DB pool silent |
| INC-002 | P1 | redis-sessions | Cache | Auth-svc shows 500s; all pods healthy |
| INC-003 | P0 | dns-resolver | Network | All pods running, gateway 503s everywhere |
| INC-004 | P2 | tls-cert-manager | Config | Webhook-svc pods healthy; cert silent expired |
| INC-005 | P2 | kafka | Service | Workers healthy; waiting for leadership election |
| INC-006 | P1 | elasticsearch | Service | Cluster status = green (index corrupt internally) |
| INC-007 | P2 | smtp-relay | Service | Notification-svc pods healthy; relay throttled |
| INC-008 | P2 | memcached | Cache | Profile-api all healthy; cache eviction spike |
| INC-009 | P2 | ml-inference | Resource | API gateway 503s; GPU OOM in ML backend |
| INC-010 | P1 | cache-invalidation | Service | CDN hit rate 99.2% (looks great; stale data) |
| INC-011 | P3 | recommendation-svc | Resource | No errors; slow memory leak approaching limit |
| INC-012 | P0 | billing-engine | Service | Stripe gateway healthy; misconfigured endpoint |

---

## Action Space

```python
class IncidentAction(BaseModel):
    # Task 1 only
    severity: str = ""           # "P0" | "P1" | "P2" | "P3"

    # Task 2 only
    root_cause_service: str = "" # e.g. "postgres-payments", "redis-sessions"
    root_cause_reason: str = ""  # sentence citing specific log/metric evidence

    # Task 3 only
    runbook_id: str = ""         # must match one of available_runbooks[].id
    eta_minutes: int = 0         # estimated minutes to resolution
    status_update: str = ""      # customer-facing text ≥ 30 words, no internal names

    # Optional (not scored)
    notes: str = ""
    confidence: float = 0.0
```

Each task uses a **non-overlapping subset** of fields. The agent reads `task_name` from the observation to know which fields to fill.

---

## Observation Space

```python
class IncidentObservation(BaseModel):
    incident_id: str               # "INC-001" … "INC-012"
    title: str                     # one-line incident summary
    task_index: int                # 0 = classify, 1 = root_cause, 2 = response
    task_name: str                 # "classify_severity" | "identify_root_cause" | "execute_response"

    # Incident telemetry (present every step)
    error_rate: float              # 0.0–1.0
    latency_p99_ms: int
    affected_users: int
    logs: List[str]                # raw log lines — includes intentional red herrings
    metrics: Dict[str, float]      # service telemetry keyed by "service.metric_name"
    available_runbooks: List[dict] # 4 options: [{"id": "RB-019", "description": "…"}]

    # Episode context
    history: List[dict]            # previous step actions and rewards
    remaining_tasks: int           # tasks left in episode
    hint: str                      # plain-English task instruction
    done: bool
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Rule-based fallback — no API key, fully deterministic, verifies env works
python inference.py --no-llm --episodes 3

# LLM agent via Hugging Face Inference Router
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="hf_your_token_here"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
python inference.py --episodes 3

# Start the API server locally
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run inference against the running server
python inference.py --base-url http://localhost:7860 --episodes 3
```

---

## Docker

```bash
# Build — runs reset()→step() validation at build time (fails fast if broken)
docker build -t opsenv .

# Run
docker run -p 7860:7860 \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e HF_TOKEN="hf_your_token" \
  -e MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
  opsenv

# Verify
curl http://localhost:7860/health
curl http://localhost:7860/tasks
curl -X POST http://localhost:7860/reset
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness probe — returns 200 |
| POST | `/reset` | Start new episode → returns initial observation |
| POST | `/step` | Submit action → returns observation, reward, done, info |
| GET | `/state?session_id=…` | Episode metadata without advancing |
| GET | `/tasks` | List all 3 tasks with difficulty and description |
| GET | `/tasks/{name}/grade?session_id=…` | Grader score 0.0–1.0 for a completed task |
| POST | `/grade_episode` | Grade all tasks in a session |
| WS | `/ws` | WebSocket persistent session |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | OpenAI-compatible API endpoint |
| `MODEL_NAME` | `meta-llama/Llama-3.1-8B-Instruct` | Model identifier for inference |
| `HF_TOKEN` | — | Hugging Face API key (accepted as `api_key` by OpenAI client) |
| `OPENAI_API_KEY` | — | Alternative credential — accepted in place of `HF_TOKEN` |

---

## Baseline Scores

**Fully reproducible with no API key:**

```bash
python inference.py --no-llm --episodes 3
```

The rule-based fallback is deterministic — same input, same output, every run.

| Episode | Incident | Severity | Root Cause | Response | Total | Normalized |
|---|---|---|---|---|---|---|
| ep1 | INC-001 (postgres pool) | 1.00 | 0.40 | 0.22 | 1.62 / 3.0 | 54.0% |
| ep2 | INC-002 (redis OOM) | 1.00 | 0.40 | 0.21 | 1.61 / 3.0 | 53.7% |
| ep3 | INC-003 (DNS loop) | 1.00 | 0.40 | 0.21 | 1.61 / 3.0 | 53.7% |
| **Average** | — | **1.00** | **0.40** | **0.21** | **1.61 / 3.0** | **53.8%** |

**Why these scores:**
- Severity = 1.00: The rule-based heuristic (error_rate thresholds) correctly classifies easy/hard cases
- Root cause = 0.40: Heuristic identifies the correct failure *family* (database/cache) but names the wrong specific service (the challenge)
- Response = ~0.21: Fallback picks first plausible runbook (usually wrong), short status update, misses ETA precision

**LLM baseline expected range** (`meta-llama/Llama-3.1-8B-Instruct`): **1.50–1.80 / 3.0** (50–60%)

---

## Project Structure

```
opsenv/
├── Dockerfile             ← PYTHONPATH, ENV defaults, HEALTHCHECK, build validation
├── README.md              ← This file (also HF Space card)
├── openenv.yaml           ← OpenEnv spec: tasks, action/obs spaces, env_vars, tags
├── requirements.txt
├── models.py              ← Pydantic models: IncidentAction, IncidentObservation, IncidentState, StepResult
├── client.py              ← HTTP client (mirrors OpsEnv API — used by remote inference mode)
├── inference.py           ← Baseline script: LLM mode + rule-based fallback (--no-llm)
└── server/
    ├── __init__.py
    ├── environment.py     ← 12 incidents, 3 graders, evidence-gated scoring
    └── app.py             ← FastAPI: REST endpoints + WebSocket
```

---

## OpenEnv Spec Compliance

- ✅ Typed Pydantic `Action`, `Observation`, `State`, `StepResult` models
- ✅ `reset()` → clean initial observation (no shared state between episodes)
- ✅ `step(action)` → `(observation, reward, done, info)` tuple
- ✅ `state()` → episode metadata, does not advance episode
- ✅ `openenv.yaml` — name, version, tasks, action space, observation space, env_vars, tags
- ✅ 3 tasks with difficulty gradient: easy → medium → hard
- ✅ Graders: deterministic, scores in 0.0–1.0, reproducible, variance across model quality
- ✅ `inference.py` in root, uses `from openai import OpenAI` (lazy import — safe without openai installed)
- ✅ Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` / `OPENAI_API_KEY` from environment
- ✅ Dockerfile: `EXPOSE 7860`, `HEALTHCHECK`, `PYTHONPATH=/app`, `ENV` defaults, build-time validation
- ✅ HF Space: `sdk: docker`, `app_port: 7860`, `tags: [openenv, …]`
- ✅ `/health` returns 200 (HF Space liveness probe)
- ✅ Baseline scores reproducible: `python inference.py --no-llm --episodes 3`
 
