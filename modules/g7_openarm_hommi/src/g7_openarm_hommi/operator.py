from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from enum import Enum

from g7_openarm_idl import UIRequest, UIResponse
from g7_openarm_utils import build_pub, build_sub

REQUEST_PERIOD = 0.5  # s; the monitor drops requests not seen for a few seconds


class OperatorState(Enum):
    WAITING = "waiting"  # warmed up, not commanding the robot yet
    RUNNING = "running"
    PAUSED = "paused"  # holding the last commanded target


_PROMPTS = {
    OperatorState.WAITING: ("HoMMI ready (warmed up) - press Enter to start", "enter"),
    OperatorState.RUNNING: ("HoMMI running - press Space to pause", "space"),
    OperatorState.PAUSED: ("HoMMI paused, holding pose - press Enter to resume", "enter"),
}
_TRANSITIONS = {
    (OperatorState.WAITING, "enter"): OperatorState.RUNNING,
    (OperatorState.RUNNING, "space"): OperatorState.PAUSED,
    (OperatorState.PAUSED, "enter"): OperatorState.RUNNING,
}


class OperatorControl:
    """Start / pause / resume driven by monitor prompts (rt/ui/request|response).

    The current prompt is republished periodically, so the monitor can start
    later than the node. ``generation`` increments on every transition, letting
    workers discard results computed under a previous state.
    """

    def __init__(self, on_change: Callable[[OperatorState], None]) -> None:
        self._lock = threading.Lock()
        self._session = uuid.uuid4().hex[:8]
        self._seq = 0
        self._state = OperatorState.WAITING
        self._on_change = on_change
        self._stop = threading.Event()
        self._pub = build_pub("rt/ui/request", UIRequest)
        self._sub = build_sub("rt/ui/response", UIResponse, self._on_response)
        self._thread = threading.Thread(
            target=self._publish_loop, name="hommi-operator", daemon=True
        )
        self._thread.start()

    @property
    def state(self) -> OperatorState:
        with self._lock:
            return self._state

    @property
    def generation(self) -> int:
        with self._lock:
            return self._seq

    def _request_id(self) -> str:
        return f"hommi-{self._session}-{self._seq}"

    def _publish_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                prompt, keys = _PROMPTS[self._state]
                msg = UIRequest(self._request_id(), "hommi", prompt, keys)
            if self._pub is not None:
                self._pub.Write(msg)
            self._stop.wait(REQUEST_PERIOD)

    def _on_response(self, msg: UIResponse) -> None:
        with self._lock:
            if msg.id != self._request_id():
                return
            new_state = _TRANSITIONS.get((self._state, msg.key))
            if new_state is None:
                return
            self._state = new_state
            self._seq += 1
        print(f"HoMMI operator: {new_state.value}", flush=True)
        self._on_change(new_state)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
