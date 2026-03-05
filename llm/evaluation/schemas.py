from pydantic import BaseModel
from typing import List, Literal

VALID_FAULT_TYPES = Literal[
    "sensor_drift", "signal_spike", "flatline",
    "cluster_fault", "inconsistency", "normal"
]


class FaultDiagnosis(BaseModel):
    faulty_sensors: List[str]
    fault_type: VALID_FAULT_TYPES
    confidence: Literal["high", "medium", "low"]
    reasoning: str
