"""Take turns on the GPU between the policy and the ZED stereo process.

Both TensorRT workloads run in parallel by default, and the GPU time-slices them:
on the Thor, FoundationStereo then slows the policy from ~90 ms to ~400 ms per
chunk. With turns, each runs alone at full speed, and when both are waiting they
alternate, so the policy gets a fresh head depth before every chunk and the stereo
model is never starved. When only one side wants the GPU it runs back to back.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.context import BaseContext
from typing import Literal

Party = Literal["policy", "stereo"]
_INDEX: dict[Party, int] = {"policy": 0, "stereo": 1}
_FREE = -1


class GpuTurns:
    """A two-party mutex shared across processes that hands over to a waiting peer."""

    def __init__(self, ctx: BaseContext | None = None) -> None:
        ctx = ctx or mp.get_context("spawn")
        self._cond = ctx.Condition(ctx.Lock())
        self._owner = ctx.RawValue("i", _FREE)
        self._turn = ctx.RawValue("i", _FREE)  # who goes next when both wait
        self._waiting = ctx.RawArray("b", 2)

    @contextmanager
    def turn(self, party: Party) -> Iterator[None]:
        me = _INDEX[party]
        other = 1 - me
        with self._cond:
            self._waiting[me] = 1
            while self._owner.value != _FREE or (
                self._waiting[other] and self._turn.value == other
            ):
                self._cond.wait()
            self._waiting[me] = 0
            self._owner.value = me
        try:
            yield
        finally:
            with self._cond:
                self._owner.value = _FREE
                self._turn.value = other
                self._cond.notify_all()
