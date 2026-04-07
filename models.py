from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IncidentAction(BaseModel):
    """
    Action submitted by the agent at each step.
    Each task uses a distinct, non-overlapping subset of fields.

    Task 1 (classify_severity)  → use: severity
    Task 2 (identify_root_cause) → use: root_cause_service, root_cause_reason
    Task 3 (execute_response)   → use: runbook_id, eta_minutes, status_update
    """
    # Task 1
    severity: str = ""                # "P0" | "P1" | "P2" | "P3"

    # Task 2
    root_cause_service: str = ""      # e.g. "postgres-payments", "redis-sessions"
    root_cause_reason: str = ""       # brief explanation citing log evidence

    # Task 3
    runbook_id: str = ""              # must match one of available_runbooks ids
    eta_minutes: int = 0              # estimated time to resolution
    status_update: str = ""           # customer-facing status page text (≥20 words)

    # Optional
    notes: str = ""
    confidence: float = 0.0


class IncidentObservation(BaseModel):
    """Observation returned by the environment after each step / reset."""
    incident_id: str
    title: str
    task_index: int
    task_name: str

    # Incident telemetry
    error_rate: float                       # 0.0–1.0
    latency_p99_ms: int
    affected_users: int
    logs: List[str]                         # raw log lines (may contain red herrings)
    metrics: Dict[str, float]               # service metrics
    available_runbooks: List[Dict[str, str]]  # [{"id": "RB-019", "description": "..."}]

    # Episode context
    history: List[Dict[str, Any]]
    remaining_tasks: int
    hint: str
    done: bool = False
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class IncidentState(BaseModel):
    """Episode metadata returned by state()."""
    episode_id: str
    step_count: int
    total_reward: float
    current_task_index: int
    current_task_name: str
    incident_id: str
    completed_tasks: List[str]
    done: bool
    notes: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class StepResult(BaseModel):
    """OpenEnv-style return: (observation, reward, done, info)."""
    observation: IncidentObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "observation": self.observation.to_dict(),
            "reward": float(self.reward),
            "done": bool(self.done),
            "info": self.info,
        }
        if self.session_id is not None:
            payload["session_id"] = self.session_id
        return payload


class TaskInfo(BaseModel):
    name: str
    difficulty: str
    description: str
    score_range: str = "0.0 – 1.0"


class TaskGradeResult(BaseModel):
    task_name: str
    score: float
    max_score: float = 1.0
    graded: bool
    details: Dict[str, Any] = Field(default_factory=dict)
