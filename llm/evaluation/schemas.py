from pydantic import BaseModel
from typing import List, Literal
from training.fault_injection import STATE_NAMES

VALID_FAULT_TYPES = Literal[STATE_NAMES]


class FaultDiagnosis(BaseModel):
    faulty_sensors: List[str]
    fault_type: VALID_FAULT_TYPES
    confidence: Literal["high", "medium", "low"]
    reasoning: str
