from pydantic import BaseModel
from typing import List, Literal, Optional


class FaultDiagnosis(BaseModel):
    is_faulty: bool
    faulty_sensors: List[str]
    fault_type: str
    confidence: Literal["high", "medium", "low"]
    reasoning: str
