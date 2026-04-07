from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from models import IncidentAction, IncidentObservation, IncidentState, StepResult


def _parse_obs(d: Dict[str, Any]) -> IncidentObservation:
    return IncidentObservation(**d)


def _parse_result(d: Dict[str, Any]) -> StepResult:
    return StepResult(
        observation=_parse_obs(d["observation"]),
        reward=float(d.get("reward", 0.0)),
        done=bool(d.get("done", False)),
        info=dict(d.get("info", {})),
        session_id=d.get("session_id"),
    )


class OpsClient:
    """
    Lightweight HTTP client for OpsEnv.
    Implements the same reset()/step()/state() interface as OpsEnv
    so inference.py can use either interchangeably.
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id: Optional[str] = None

    def reset(self) -> StepResult:
        r = requests.post(f"{self.base_url}/reset", json={}, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()
        self.session_id = payload.get("session_id")
        return _parse_result(payload)

    def step(self, action: Any) -> StepResult:
        if not self.session_id:
            raise RuntimeError("Call reset() before step().")
        if isinstance(action, dict):
            payload = {"session_id": self.session_id, **action}
        else:
            payload = {"session_id": self.session_id, **action.model_dump()}
        r = requests.post(f"{self.base_url}/step", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return _parse_result(r.json())

    def state(self) -> IncidentState:
        if not self.session_id:
            raise RuntimeError("Call reset() before state().")
        r = requests.get(f"{self.base_url}/state",
                         params={"session_id": self.session_id},
                         timeout=self.timeout)
        r.raise_for_status()
        return IncidentState(**r.json())

    def close(self) -> None:
        if self.session_id:
            try:
                requests.post(f"{self.base_url}/close",
                              json={"session_id": self.session_id},
                              timeout=self.timeout)
            except Exception:
                pass
        self.session_id = None
