from dataclasses import dataclass

import cyclonedds.idl.annotations as annotate
from cyclonedds.idl import IdlStruct

from .amr_cmd import AMRCmd, AMRCmd_default
from .openarm_cmd import OpenArmCmd, OpenArmCmd_default


@dataclass
@annotate.final
@annotate.autoid("sequential")
class WBCLowCmd(IdlStruct, typename="WBCLowCmd"):
    amr: AMRCmd
    openarm: OpenArmCmd


def WBCLowCmd_default():
    return WBCLowCmd(AMRCmd_default(), OpenArmCmd_default())  # type: ignore
