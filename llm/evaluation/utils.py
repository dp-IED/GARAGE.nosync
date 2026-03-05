import numpy as np
from typing import Dict, List

from llm.evaluation.schemas import FaultDiagnosis


def sensor_labels_to_window_label(sensor_labels: np.ndarray) -> int:
    """
    Convert sensor-level labels to window-level sensor-indexed label.

    Args:
        sensor_labels: (num_sensors,) binary array - which sensors are faulty

    Returns:
        int: 0 if no fault, 1-8 if fault detected (1-indexed sensor index)
    """
    faulty_indices = np.where(sensor_labels > 0)[0]
    if len(faulty_indices) == 0:
        return 0
    return int(faulty_indices[0]) + 1


def parse_structured_response(raw_json: str, sensor_cols: List[str]) -> Dict:
    sensor_lookup = {s.lower(): s for s in sensor_cols}
    try:
        data = FaultDiagnosis.model_validate_json(raw_json)
    except Exception:
        return {
            "sensor_labels": [],
            "fault_type": "unknown",
            "confidence": "low",
            "reasoning": "parse_error",
        }
    validated_sensors = [
        sensor_lookup[s.lower()]
        for s in data.faulty_sensors
        if s.lower() in sensor_lookup
    ]
    return {
        "sensor_labels": validated_sensors,
        "fault_type": data.fault_type,
        "confidence": data.confidence,
        "reasoning": data.reasoning,
    }


def parsed_to_prediction(parsed: Dict, sensor_names: List[str]) -> Dict:
    sensor_labels = np.zeros(len(sensor_names), dtype=np.float32)
    for s in parsed["sensor_labels"]:
        if s in sensor_names:
            sensor_labels[sensor_names.index(s)] = 1.0
        else:
            sensor_clean = s.replace(" ()", "").strip()
            for i, name in enumerate(sensor_names):
                if name.replace(" ()", "").strip() == sensor_clean:
                    sensor_labels[i] = 1.0
                    break
    window_label = sensor_labels_to_window_label(sensor_labels)
    return {
        "window_label": window_label,
        "sensor_labels": sensor_labels,
        "sensor_labels_raw": sensor_labels.copy(),
        "sensor_labels_root_only": sensor_labels.copy(),
        "root_cause_sensors": parsed["sensor_labels"],
        "affected_sensors": [],
        "fault_type": parsed["fault_type"] if parsed["fault_type"] != "unknown" else None,
        "reasoning": parsed["reasoning"],
    }
