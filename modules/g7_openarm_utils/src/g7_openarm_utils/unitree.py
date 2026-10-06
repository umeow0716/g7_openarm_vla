from  __future__ import annotations

import uuid

from typing import Any, Callable
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.utils.hz_sample import RecurrentThread


def build_pub(
    topic: str,
    type: Any,
    enabled=True
):
    if not enabled:
        return None

    pub = ChannelPublisher(topic, type)
    pub.Init()
    return pub


def build_sub(
    topic: str,
    type: Any,
    handler: Callable,
    enabled=True
):
    if not enabled:
        return None

    sub = ChannelSubscriber(topic, type)
    sub.Init(handler, 0)
    return sub


def build_thread(
    hz: int | float,
    target: Callable,
    enabled: bool = True
):
    if hz <= 0.0:
        raise RuntimeError("hz must be greater than 0")

    if not enabled:
        return None

    interval = 1.0 / hz

    thread = RecurrentThread(
        name=str(uuid.uuid4()),
        target=target,
        interval=interval,
    )
    thread.Start()

    return thread
