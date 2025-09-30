from typing import List, Tuple


def add_jitter_to_events(events: List[Tuple[float, int]], eps: float = 1e-6) -> List[Tuple[float, int]]:
    events = sorted(events, key=lambda x: (x[0], x[1]))
    out: List[Tuple[float, int]] = []
    last_t = None
    for t, i in events:
        if last_t is not None and t <= last_t:
            t = last_t + eps
        out.append((float(t), int(i)))
        last_t = t
    return out


