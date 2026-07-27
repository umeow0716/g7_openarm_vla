# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import collections
import json
import select
import socket
import threading
import time
from typing import Any

from .udp_response import VRUDPResponse


class JsonUdpReceiver:
    """Receive VR JSON packets in a background thread and retain the freshest one."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5006, buf_size: int = 4096) -> None:
        self._host = host
        self._port = port
        self._buf_size = buf_size
        self._lock = threading.Lock()
        self._latest: VRUDPResponse | None = None
        self._recv_ts: collections.deque[int] = collections.deque(maxlen=512)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def latest(self) -> VRUDPResponse | None:
        with self._lock:
            return self._latest

    def drain_recv_timestamps(self) -> list[int]:
        """Return and clear the arrival timestamps (ns) collected since last call."""
        with self._lock:
            items = list(self._recv_ts)
            self._recv_ts.clear()
            return items

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)

    @staticmethod
    def _parse_packet(data: bytes) -> VRUDPResponse | None:
        try:
            line = data.decode("utf-8", errors="replace").strip()
            if not line:
                return None
            payload: Any = json.loads(line)
            if not isinstance(payload, dict):
                return None
            return VRUDPResponse.from_mapping(payload)
        except (json.JSONDecodeError, ValueError):
            return None

    def _loop(self) -> None:
        while self._running:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
                    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    server.bind((self._host, self._port))
                    server.settimeout(1.0)

                    while self._running:
                        try:
                            data, _ = server.recvfrom(self._buf_size)
                            recv_ns = time.time_ns()
                            last_message = self._parse_packet(data)
                            arrivals = [recv_ns] if last_message is not None else []

                            while select.select([server], [], [], 0.0)[0]:
                                data, _ = server.recvfrom(self._buf_size)
                                recv_ns = time.time_ns()
                                parsed = self._parse_packet(data)
                                if parsed is not None:
                                    arrivals.append(recv_ns)
                                    last_message = parsed

                            with self._lock:
                                self._recv_ts.extend(arrivals)
                                if last_message is not None:
                                    self._latest = last_message
                        except TimeoutError:
                            continue
                        except OSError:
                            if self._running:
                                time.sleep(0.1)
                            break
            except OSError:
                if self._running:
                    time.sleep(1.0)
