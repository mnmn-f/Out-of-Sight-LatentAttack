import os
from typing import Dict, List, Optional, Sequence, Set, Tuple


SEQUENTIAL_EDGE_BY_SOURCE_ROLE = {
    "planner": "planner->critic",
    "critic": "critic->refiner",
    "refiner": "refiner->judger",
}


def parse_roles(raw: str) -> Optional[List[str]]:
    raw = (raw or "").strip().lower()
    if not raw or raw == "all":
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def edge_label_for_role(role: str, edge_map: str) -> Optional[str]:
    if edge_map == "none":
        return None
    if edge_map != "sequential":
        raise ValueError(f"Unsupported edge_map: {edge_map}")
    return SEQUENTIAL_EDGE_BY_SOURCE_ROLE.get(role)


def list_sample_dirs(trace_root: str) -> List[str]:
    if not os.path.isdir(trace_root):
        raise FileNotFoundError(f"Trace directory not found: {trace_root}")
    sample_dirs = []
    for name in sorted(os.listdir(trace_root)):
        full_path = os.path.join(trace_root, name)
        if os.path.isdir(full_path) and name.startswith("sample_"):
            sample_dirs.append(full_path)
    return sample_dirs


def sample_index_from_dir(sample_dir: str) -> Optional[int]:
    name = os.path.basename(sample_dir)
    if not name.startswith("sample_"):
        return None
    raw_idx = name[len("sample_") :]
    try:
        return int(raw_idx)
    except ValueError:
        return None


def find_role_trace(sample_dir: str, role: str) -> Optional[str]:
    prefix = f"{role}_"
    for name in sorted(os.listdir(sample_dir)):
        if name.startswith(prefix) and name.endswith(".pt"):
            return os.path.join(sample_dir, name)
    return None


def collect_trace_pairs(
    clean_root: str,
    attacked_root: str,
    roles: Optional[Sequence[str]],
    allowed_sample_indices: Optional[Set[int]] = None,
) -> Dict[str, List[Tuple[str, str]]]:
    clean_samples = {os.path.basename(path): path for path in list_sample_dirs(clean_root)}
    attacked_samples = {os.path.basename(path): path for path in list_sample_dirs(attacked_root)}
    common_sample_names = sorted(set(clean_samples) & set(attacked_samples))
    pairs: Dict[str, List[Tuple[str, str]]] = {}
    role_set = set(roles) if roles else None

    for sample_name in common_sample_names:
        clean_dir = clean_samples[sample_name]
        attacked_dir = attacked_samples[sample_name]
        sample_idx = sample_index_from_dir(clean_dir)
        if allowed_sample_indices is not None and sample_idx not in allowed_sample_indices:
            continue

        candidate_roles = set()
        for name in os.listdir(clean_dir):
            if name.endswith(".pt") and "_" in name:
                candidate_roles.add(name.split("_", 1)[0])
        for name in os.listdir(attacked_dir):
            if name.endswith(".pt") and "_" in name:
                candidate_roles.add(name.split("_", 1)[0])

        for role in sorted(candidate_roles):
            if role_set is not None and role not in role_set:
                continue
            clean_file = find_role_trace(clean_dir, role)
            attacked_file = find_role_trace(attacked_dir, role)
            if clean_file and attacked_file:
                pairs.setdefault(role, []).append((clean_file, attacked_file))

    return pairs
