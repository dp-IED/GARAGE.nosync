import numpy as np
from typing import Dict, List, Any, Optional

from llm.evaluation.schemas import FaultDiagnosis
from llm.inference import create_json_schema_response_format
from training.fault_injection import STATE_NAMES as _VALID_FAULT_TYPES


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


def _normalize_sensor_name(name: str) -> str:
    """Normalize for fuzzy matching: lowercase, strip () suffix and surrounding whitespace."""
    return name.replace(" ()", "").replace("()", "").strip().lower()


def parse_structured_response(raw_json: str, sensor_cols: List[str]) -> Dict:

    sensor_lookup = {s.lower(): s for s in sensor_cols}
    normalized_lookup = {_normalize_sensor_name(s): s for s in sensor_cols}

    try:
        data = FaultDiagnosis.model_validate_json(raw_json)
    except Exception:
        return {
            "is_faulty": False,
            "sensor_labels": [],
            "fault_type": "normal",
            "confidence": "low",
            "reasoning": "parse_error",
        }
    validated_sensors = []
    if data.is_faulty:
        for s in data.faulty_sensors:
            if s.lower() in sensor_lookup:
                validated_sensors.append(sensor_lookup[s.lower()])
            else:
                norm = _normalize_sensor_name(s)
                if norm in normalized_lookup:
                    validated_sensors.append(normalized_lookup[norm])

    # Preserve the LLM's direct is_faulty verdict independently of sensor validation.
    # A window is faulty if the LLM said so — even if sensor name matching failed.
    is_faulty = data.is_faulty

    # Normalise fault_type: always a non-None string.
    # If the window is not faulty, use "normal". If faulty, use the LLM's value (normalised).
    llm_fault_type = data.fault_type.strip() if data.fault_type else ""
    if not is_faulty:
        fault_type = "normal"
    elif llm_fault_type.lower() in {ft.lower() for ft in _VALID_FAULT_TYPES}:
        fault_type = next(
            ft for ft in _VALID_FAULT_TYPES if ft.lower() == llm_fault_type.lower()
        )
    else:
        import warnings
        fault_type = llm_fault_type if llm_fault_type else "unknown"
        warnings.warn(
            f"LLM returned unrecognised fault_type {llm_fault_type!r} (not in VALID_FAULT_TYPES); "
            f"storing as {fault_type!r}. Check prompt or model output."
        )

    return {
        "is_faulty": is_faulty,
        "sensor_labels": validated_sensors,
        "fault_type": fault_type,
        "confidence": data.confidence,
        "reasoning": data.reasoning,
    }


def call_llm_fault_diagnosis(
    client: Any,
    model_name: str,
    messages: List[Dict[str, str]],
    sensor_names: List[str],
    *,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Dict:
    """
    Call LLM with structured JSON schema output (same method as serialised KG).
    Uses FaultDiagnosis schema for reliable parsing.

    temperature: sampling temperature (0.0 = greedy-ish; eval scripts should pass this explicitly).
    max_tokens: cap generation length; None uses client default, or 2048 if the client has no default.
    """
    _schema_dict = FaultDiagnosis.model_json_schema()
    response_format = create_json_schema_response_format(
        schema_name="FaultDiagnosis",
        schema_dict=_schema_dict,
        strict=True,
    )

    try:
        cc_kwargs: Dict[str, Any] = {
            "messages": messages,
            "response_format": response_format,
            "temperature": temperature,
        }
        if max_tokens is not None:
            cc_kwargs["max_tokens"] = max_tokens
        elif getattr(client, "config", None) is not None and client.config.max_tokens is not None:
            cc_kwargs["max_tokens"] = client.config.max_tokens
        else:
            cc_kwargs["max_tokens"] = 2048
        response = client.chat_completions_create(**cc_kwargs)
        raw_json = response["choices"][0]["message"]["content"]
    except Exception as e:
        return {
            "is_faulty": False,
            "window_label": 0,
            "sensor_labels": np.zeros(len(sensor_names), dtype=np.float32),
            "sensor_labels_raw": np.zeros(len(sensor_names), dtype=np.float32),
            "sensor_labels_root_only": np.zeros(len(sensor_names), dtype=np.float32),
            "root_cause_sensors": [],
            "affected_sensors": [],
            "fault_type": "normal",
            "reasoning": f"Error: {str(e)}",
        }
    parsed = parse_structured_response(raw_json, sensor_names)
    return parsed_to_prediction(parsed, sensor_names)


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

    # Use the LLM's direct is_faulty verdict for the binary window label.
    # Fall back to sensor-derived label when is_faulty is not present (e.g. legacy callers).
    is_faulty: bool = parsed.get("is_faulty", sensor_labels_to_window_label(sensor_labels) > 0)

    # Coerce inconsistency: if the LLM claimed fault but named no recognised sensors,
    # treat the window as normal so that window_label and is_faulty stay in agreement.
    if is_faulty and sensor_labels.sum() == 0:
        is_faulty = False

    window_label = sensor_labels_to_window_label(sensor_labels) if is_faulty else 0

    # fault_type is always a non-None string: "normal", a valid fault type, or "unknown".
    fault_type: str = parsed.get("fault_type") or ("normal" if not is_faulty else "unknown")
    if not is_faulty:
        fault_type = "normal"

    return {
        "is_faulty": is_faulty,
        "window_label": window_label,
        "sensor_labels": sensor_labels,
        "sensor_labels_raw": sensor_labels.copy(),
        "sensor_labels_root_only": sensor_labels.copy(),
        "root_cause_sensors": parsed["sensor_labels"],
        "affected_sensors": [],
        "fault_type": fault_type,
        "reasoning": parsed["reasoning"],
    }
