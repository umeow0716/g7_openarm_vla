import os
import select
import signal
import sys
import termios
import threading
import time
import tty
from typing import Any

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_

from g7_openarm_idl import Odom, Target, UIRequest, UIResponse, WBCLowCmd
from g7_openarm_utils import ChannelFactoryInitialize, build_pub, build_sub

from .config import config

# A request disappears when its owner stops republishing it for this long.
REQUEST_TTL = 3.0  # s
KEY_NAMES = {"\r": "enter", "\n": "enter", " ": "space"}


class HzMonitor:
    def __init__(self, topic: str, msg_type: Any) -> None:
        self.topic = topic
        self.msg_type = msg_type
        self._count = 0
        self._start = time.monotonic()
        self._lock = threading.Lock()
        self._hz = 0.0
        self._sub = build_sub(self.topic, self.msg_type, self.callback)

    def callback(self, _: Any) -> None:
        with self._lock:
            self._count += 1

    def read_and_reset(self) -> int:
        with self._lock:
            count = self._count
            self._count = 0
            return count

    def get_hz(self) -> float:
        elapsed = time.monotonic() - self._start
        if elapsed < 1.0:
            return self._hz
        count = self.read_and_reset()
        self._hz = count / elapsed
        self._start = time.monotonic()

        return self._hz


class RequestBoard:
    """Operator prompts from other modules (rt/ui/request -> rt/ui/response)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # id -> (first_seen, last_seen, request)
        self._requests: dict[str, tuple[float, float, UIRequest]] = {}
        self._answered: set[str] = set()
        self._pub = build_pub("rt/ui/response", UIResponse)
        self._sub = build_sub("rt/ui/request", UIRequest, self._on_request)

    def _on_request(self, msg: UIRequest) -> None:
        now = time.monotonic()
        with self._lock:
            if msg.id in self._answered:
                return
            first = self._requests.get(msg.id, (now, now, msg))[0]
            self._requests[msg.id] = (first, now, msg)

    def active(self) -> list[UIRequest]:
        now = time.monotonic()
        with self._lock:
            stale = [r for r, (_, seen, _) in self._requests.items() if now - seen > REQUEST_TTL]
            for rid in stale:
                del self._requests[rid]
            return [req for _, _, req in sorted(self._requests.values(), key=lambda v: v[0])]

    def press(self, key: str) -> UIRequest | None:
        """Answer the oldest active request that accepts ``key``."""
        for req in self.active():
            if key in {k.strip() for k in req.keys.split(",")}:
                with self._lock:
                    self._requests.pop(req.id, None)
                    self._answered.add(req.id)
                if self._pub is not None:
                    self._pub.Write(UIResponse(req.id, key))
                return req
        return None


class TerminalKeys:
    """Non-blocking key reader on the controlling terminal (works from spawned
    processes whose stdin is /dev/null). Restores the terminal on exit."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._saved: list[Any] | None = None
        if not sys.stdout.isatty():
            # Only take keys from a terminal that is actually showing the
            # monitor; a launcher started in the background (output redirected)
            # must not grab the keyboard of whatever terminal it inherited.
            return
        try:
            self._fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)  # no line buffering / echo; Ctrl+C still works
        except OSError:
            self._fd = None

    @property
    def available(self) -> bool:
        return self._fd is not None

    def read(self) -> list[str]:
        if self._fd is None:
            return []
        keys: list[str] = []
        while select.select([self._fd], [], [], 0)[0]:
            try:
                chunk = os.read(self._fd, 64).decode(errors="ignore")
            except BlockingIOError:
                break
            if not chunk:
                break
            keys.extend(KEY_NAMES.get(ch, ch) for ch in chunk)
        return keys

    def restore(self) -> None:
        if self._fd is not None and self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            os.close(self._fd)
            self._fd = None


def render(monitors: list[HzMonitor], requests: list[UIRequest], keys_ok: bool, note: str) -> str:
    lines = ["\033[H\033[2J\033[3J", "== topics " + "=" * 50]
    lines += [f"{m.topic}: {m.get_hz():.2f} Hz" for m in monitors]
    lines.append("== requests " + "=" * 48)
    if not requests:
        lines.append("(none)")
    for req in requests:
        keys = " / ".join(k.strip().capitalize() for k in req.keys.split(","))
        lines.append(f"[{req.source}] {req.prompt}  <{keys}>")
    if not keys_ok:
        lines.append("(no terminal: requests are shown but cannot be answered here)")
    if note:
        lines.append(note)
    return "\n".join(lines) + "\n"


def main() -> None:
    ChannelFactoryInitialize(config.dds.domain_id, config.dds.interface)

    monitors = [
        HzMonitor("rt/target", Target),
        HzMonitor("rt/imustate", IMUState_),
        HzMonitor("rt/lowstate", LowState_),
        HzMonitor("rt/lowcmd", LowCmd_),
        HzMonitor("rt/odom", Odom),
        HzMonitor("rt/wbclowcmd", WBCLowCmd),
    ]
    board = RequestBoard()
    keys = TerminalKeys()

    def stop(_signum: int, _frame: object) -> None:
        keys.restore()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)

    note = ""
    next_draw = 0.0
    shown: list[str] = []
    try:
        while True:
            for key in keys.read():
                answered = board.press(key)
                if answered is not None:
                    note = f"-> answered [{answered.source}] with {key}"
                    next_draw = 0.0
            now = time.monotonic()
            requests = board.active()
            ids = [f"{r.id}:{r.prompt}" for r in requests]
            if now >= next_draw or ids != shown:
                sys.stdout.write(render(monitors, requests, keys.available, note))
                sys.stdout.flush()
                shown = ids
                next_draw = now + config.interval
            time.sleep(0.05)
    finally:
        keys.restore()


if __name__ == "__main__":
    main()
