import json
from pathlib import Path


def record_metrics(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, allow_nan=False) + "\n"
        )
