from dataclasses import dataclass

import cyclonedds.idl.annotations as annotate
from cyclonedds.idl import IdlStruct


@dataclass
@annotate.final
@annotate.autoid("sequential")
class UIRequest(IdlStruct, typename="UIRequest"):
    """An operator prompt shown by the monitor (rt/ui/request).

    Requesters republish their active request periodically; the monitor drops
    requests it has not seen for a few seconds, so no explicit cancel is needed.
    """

    id: str  # unique per request instance; responses echo it
    source: str  # requesting module, e.g. "hommi"
    prompt: str
    keys: str  # comma-separated accepted keys, e.g. "enter" or "space"


@dataclass
@annotate.final
@annotate.autoid("sequential")
class UIResponse(IdlStruct, typename="UIResponse"):
    """The operator's answer to a UIRequest (rt/ui/response)."""

    id: str
    key: str


def UIRequest_default() -> UIRequest:
    return UIRequest("", "", "", "")


def UIResponse_default() -> UIResponse:
    return UIResponse("", "")
