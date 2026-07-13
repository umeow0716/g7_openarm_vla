from __future__ import annotations

from cyclonedds.idl import IdlStruct
from cyclonedds.internal import SampleInfo

from .amr_cmd import AMRCmd
from .openarm_cmd import OpenArmCmd


class WBCLowCmd(IdlStruct):
    amr: AMRCmd
    openarm: OpenArmCmd

    sample_info: SampleInfo

    def __init__(
        self,
        amr: AMRCmd,
        openarm: OpenArmCmd,
    ) -> None:
        ...


def WBCLowCmd_default() -> WBCLowCmd:
    ...
