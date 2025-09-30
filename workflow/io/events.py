import json
from typing import List, Tuple


def save_events_json(path: str, events: List[Tuple[float, int]]):
    data = [{"t": float(t), "i": int(i)} for t, i in events]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_events_json(path: str) -> List[Tuple[float, int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = [(float(d["t"]), int(d["i"])) for d in data]
    events.sort(key=lambda x: x[0])
    return events


