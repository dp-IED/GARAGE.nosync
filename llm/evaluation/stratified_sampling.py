"""
Stratified subsampling of evaluation windows by fault type.

Used when capping the number of windows (--limit) so class proportions stay close to
the full dataset instead of taking only the first N rows in file order. Eval CLIs default
to this behavior; pass --no-stratify-limit for first-N rows without fault_types.
"""

import numpy as np

_INVALID_FAULT_TYPE_LABELS = frozenset({"normal", "unknown", "faulty", ""})


def validate_fault_types_for_stratification(
    fault_types: np.ndarray,
    sensor_labels: np.ndarray,
) -> None:
    """
    Every window marked faulty in sensor_labels must have a non-placeholder fault_types entry.
    Raises ValueError otherwise (stratified quotas need a label per stratum).
    """
    faulty_mask = sensor_labels.sum(axis=1) != 0
    faulty_indices = np.where(faulty_mask)[0]
    if len(faulty_indices) == 0:
        return
    for i in faulty_indices:
        ft = fault_types[i]
        ft_str = (str(ft).strip() if ft is not None else "")
        if ft is None or ft_str in _INVALID_FAULT_TYPE_LABELS:
            raise ValueError(
                f"Stratified limit requires valid fault_types for every faulty window. "
                f"Window {i} has sensor_labels.sum(axis=1) != 0 but fault_types[{i}] = {ft!r} "
                f"(None, empty, 'normal', 'unknown', or 'faulty' are invalid for faulty windows)."
            )


def stratified_sample_indices(
    fault_types: np.ndarray,
    limit: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Sample row indices stratified by fault type, preserving full-dataset proportions.
    Returns indices sorted ascending for stable iteration order.
    """
    n = len(fault_types)
    if limit >= n:
        return np.arange(n)

    ft_str = np.array(
        [(str(ft).strip() or "normal") if ft is not None else "normal" for ft in fault_types]
    )
    unique = np.unique(ft_str)
    stratum_indices = {ft: np.where(ft_str == ft)[0] for ft in unique}
    stratum_sizes = {ft: len(idxs) for ft, idxs in stratum_indices.items()}
    total = sum(stratum_sizes.values())

    targets = {}
    for ft in unique:
        raw = limit * stratum_sizes[ft] / total
        targets[ft] = max(0, min(stratum_sizes[ft], int(np.floor(raw))))

    current_sum = sum(targets.values())
    remainder = limit - current_sum
    if remainder > 0:
        fracs = [(limit * stratum_sizes[ft] / total - targets[ft], ft) for ft in unique]
        fracs.sort(key=lambda x: -x[0])
        for _, ft in fracs:
            if remainder <= 0:
                break
            add = min(remainder, stratum_sizes[ft] - targets[ft])
            targets[ft] += add
            remainder -= add

    rng = np.random.default_rng(random_state)
    sampled = []
    for ft in unique:
        idxs = stratum_indices[ft]
        k = min(targets[ft], len(idxs))
        chosen = rng.choice(idxs, size=k, replace=False)
        sampled.extend(chosen.tolist())

    return np.sort(np.array(sampled))
