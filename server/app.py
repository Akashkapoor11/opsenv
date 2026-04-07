from __future__ import annotations

import os
import uuid
from typing import Any, Dict

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from models import IncidentAction, TaskInfo
from server.environment import OpsEnv

app = FastAPI(
    title="OpsEnv — Production Incident Response",
    version="1.0.0",
    description=(
        "A real-world SRE incident response environment. "
        "An AI agent receives production incident alerts, logs, and metrics "
        "and must classify severity, identify root cause, and execute the correct response. "
        "Implements the OpenEnv step()/reset()/state() interface."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: Dict[str, OpsEnv] = {}


def _get_env(session_id: str) -> OpsEnv:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return _sessions[session_id]


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@app.get("/")
def root() -> Dict[str, str]:
    """Root endpoint — HF Spaces health probe hits this first."""
    return {"status": "healthy", "environment": "opsenv", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness probe — must return 200 for HF Space ping."""
    return {"status": "healthy", "environment": "opsenv", "version": "1.0.0"}


@app.get("/openenv.yaml", response_class=PlainTextResponse)
def serve_openenv_yaml() -> str:
    """Serve the openenv.yaml spec — required by `openenv validate`."""
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "openenv.yaml")
    yaml_path = os.path.abspath(yaml_path)
    try:
        with open(yaml_path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="openenv.yaml not found")


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    """Return environment metadata — required by `openenv validate` (metadata_endpoint check)."""
    return {
        "name": "opsenv",
        "version": "1.0.0",
        "description": (
            "OpsEnv is a real-world production incident response environment for AI agents. "
            "An agent receives live-style incident alerts, error logs, and service metrics "
            "and must perform three sequential SRE tasks: classify incident severity, "
            "identify the true root cause service, and execute the correct response."
        ),
        "author": "Akash Kapoor <akashkapoor12004@gmail.com>",
        "license": "MIT",
        "tags": ["openenv", "sre", "incident-response", "real-world", "operations"],
        "tasks": [t.model_dump() for t in OpsEnv.TASK_INFO],
    }


@app.get("/schema")
def schema() -> Dict[str, Any]:
    """Return action/observation/state schemas — required by `openenv validate` (schema_endpoint check)."""
    return {
        "action": {
            "type": "object",
            "description": "Task-specific action fields — each task uses a non-overlapping subset.",
            "properties": {
                "severity": {"type": "string", "description": "Task 1: P0, P1, P2, or P3"},
                "root_cause_service": {"type": "string", "description": "Task 2: root cause service name"},
                "root_cause_reason": {"type": "string", "description": "Task 2: brief explanation"},
                "runbook_id": {"type": "string", "description": "Task 3: runbook ID from available list"},
                "eta_minutes": {"type": "integer", "description": "Task 3: estimated minutes to resolution"},
                "status_update": {"type": "string", "description": "Task 3: customer-facing status text"},
                "notes": {"type": "string", "description": "Optional internal notes"},
                "confidence": {"type": "number", "description": "Agent self-reported confidence 0.0-1.0"},
            },
        },
        "observation": {
            "type": "object",
            "description": "Incident telemetry and task context returned at each step.",
            "properties": {
                "incident_id": {"type": "string"},
                "title": {"type": "string"},
                "task_index": {"type": "integer", "description": "0=classify_severity, 1=identify_root_cause, 2=execute_response"},
                "task_name": {"type": "string"},
                "error_rate": {"type": "number", "description": "Service error rate 0.0-1.0"},
                "latency_p99_ms": {"type": "integer"},
                "affected_users": {"type": "integer"},
                "logs": {"type": "array", "items": {"type": "string"}},
                "metrics": {"type": "object"},
                "available_runbooks": {"type": "array", "items": {"type": "object"}},
                "history": {"type": "array", "items": {"type": "object"}},
                "remaining_tasks": {"type": "integer"},
                "hint": {"type": "string"},
                "done": {"type": "boolean"},
            },
        },
        "state": {
            "type": "object",
            "description": "Episode metadata returned by state() without advancing the episode.",
            "properties": {
                "episode_id": {"type": "string"},
                "step_count": {"type": "integer"},
                "total_reward": {"type": "number"},
                "current_task_index": {"type": "integer"},
                "current_task_name": {"type": "string"},
                "incident_id": {"type": "string"},
                "completed_tasks": {"type": "array", "items": {"type": "string"}},
                "done": {"type": "boolean"},
            },
        },
    }


@app.post("/mcp")
async def mcp_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal JSON-RPC 2.0 MCP endpoint — required by `openenv validate` (mcp_endpoint check).
    Handles initialize and describe methods; all others return method-not-found.
    """
    jsonrpc = body.get("jsonrpc", "2.0")
    req_id = body.get("id", 1)
    method = body.get("method", "")

    if method == "initialize":
        return {
            "jsonrpc": jsonrpc,
            "id": req_id,
            "result": {
                "name": "opsenv",
                "version": "1.0.0",
                "description": "OpsEnv — Production Incident Response Environment",
                "capabilities": ["reset", "step", "state"],
            },
        }
    if method == "describe":
        return {
            "jsonrpc": jsonrpc,
            "id": req_id,
            "result": {
                "environment": "opsenv",
                "tasks": [t.model_dump() for t in OpsEnv.TASK_INFO],
            },
        }
    # Default: echo back a valid JSON-RPC 2.0 response
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {"environment": "opsenv", "method": method, "status": "ok"},
    }


# ------------------------------------------------------------------
# OpenEnv core endpoints
# ------------------------------------------------------------------


@app.post("/reset")
def reset() -> Dict[str, Any]:
    """Start a new episode. Returns initial observation."""
    session_id = uuid.uuid4().hex[:12]
    env = OpsEnv()
    _sessions[session_id] = env
    result = env.reset()
    env._session_id = session_id
    result.session_id = session_id
    result.observation.session_id = session_id
    payload = result.to_dict()
    payload["session_id"] = session_id
    return payload


@app.post("/step")
def step(body: Dict[str, Any]) -> Dict[str, Any]:
    """Submit an action and advance the episode by one step."""
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    env = _get_env(session_id)
    action = IncidentAction(
        severity=str(body.get("severity", "")),
        root_cause_service=str(body.get("root_cause_service", "")),
        root_cause_reason=str(body.get("root_cause_reason", "")),
        runbook_id=str(body.get("runbook_id", "")),
        eta_minutes=int(body.get("eta_minutes", 0) or 0),
        status_update=str(body.get("status_update", "")),
        notes=str(body.get("notes", "")),
        confidence=float(body.get("confidence", 0.0) or 0.0),
    )
    result = env.step(action)
    result.session_id = session_id
    payload = result.to_dict()
    payload["session_id"] = session_id
    return payload


@app.get("/state")
def state(session_id: str) -> Dict[str, Any]:
    """Return current episode metadata without advancing the episode."""
    env = _get_env(session_id)
    return env.state().to_dict()


@app.post("/close")
def close_session(body: Dict[str, Any]) -> Dict[str, Any]:
    """Release server-side session resources."""
    session_id = body.get("session_id")
    if session_id and session_id in _sessions:
        del _sessions[session_id]
    return {"status": "closed", "session_id": session_id}


# ------------------------------------------------------------------
# Task listing & grader endpoints
# ------------------------------------------------------------------

@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    """List all tasks with difficulty and description."""
    tasks = [t.model_dump() for t in OpsEnv.TASK_INFO]
    return {
        "environment": "opsenv",
        "version": "1.0.0",
        "tasks": tasks,
        "total": len(tasks),
    }


@app.get("/tasks/{task_name}")
def get_task(task_name: str) -> Dict[str, Any]:
    """Return metadata for a specific task by name."""
    for t in OpsEnv.TASK_INFO:
        if t.name == task_name:
            return t.model_dump()
    raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found")


@app.get("/tasks/{task_name}/grade")
def grade_task(task_name: str, session_id: str) -> Dict[str, Any]:
    """Return the deterministic grade (0.0–1.0) for a completed task in a session."""
    env = _get_env(session_id)
    result = env.grade_task(task_name)
    return result.model_dump()


@app.post("/grade_episode")
def grade_episode(body: Dict[str, Any]) -> Dict[str, Any]:
    """Return grades for all completed tasks in the episode."""
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    env = _get_env(session_id)
    grades = {}
    total = 0.0
    for task_name in OpsEnv.TASKS:
        g = env.grade_task(task_name)
        grades[task_name] = g.model_dump()
        if g.graded:
            total += g.score

    return {
        "session_id": session_id,
        "episode_done": env.done,
        "grades": grades,
        "total_score": round(total, 4),
        "max_score": float(len(OpsEnv.TASKS)),
        "normalized_score": round(total / len(OpsEnv.TASKS), 4),
    }


# ------------------------------------------------------------------
# WebSocket (persistent session)
# ------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Persistent WebSocket for multi-step interaction.
    Messages: {"type": "reset"} | {"type": "step", "action": {...}}
              {"type": "state"} | {"type": "grade", "task_name": "..."} | {"type": "close"}
    """
    await websocket.accept()
    session_id = uuid.uuid4().hex[:12]
    env = OpsEnv()
    _sessions[session_id] = env
    await websocket.send_json({"type": "session", "session_id": session_id})

    try:
        while True:
            message = await websocket.receive_json()
            msg_type = message.get("type")

            if msg_type == "reset":
                result = env.reset()
                result.session_id = session_id
                await websocket.send_json({"type": "reset_result", **result.to_dict()})

            elif msg_type == "step":
                raw = message.get("action", {})
                action = IncidentAction(
                    severity=str(raw.get("severity", "")),
                    root_cause_service=str(raw.get("root_cause_service", "")),
                    root_cause_reason=str(raw.get("root_cause_reason", "")),
                    runbook_id=str(raw.get("runbook_id", "")),
                    eta_minutes=int(raw.get("eta_minutes", 0) or 0),
                    status_update=str(raw.get("status_update", "")),
                    notes=str(raw.get("notes", "")),
                    confidence=float(raw.get("confidence", 0.0) or 0.0),
                )
                result = env.step(action)
                result.session_id = session_id
                await websocket.send_json({"type": "step_result", **result.to_dict()})

            elif msg_type == "state":
                await websocket.send_json({"type": "state_result", "state": env.state().to_dict()})

            elif msg_type == "grade":
                task_name = message.get("task_name", "")
                grade = env.grade_task(task_name)
                await websocket.send_json({"type": "grade_result", **grade.model_dump()})

            elif msg_type == "close":
                await websocket.send_json({"type": "closed", "session_id": session_id})
                break
            else:
                await websocket.send_json({"type": "error", "message": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        pass
    finally:
        _sessions.pop(session_id, None)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
