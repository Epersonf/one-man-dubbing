from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf8")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf8"))


def read_all_json(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    results = []
    for path in sorted(directory.glob("*.json")):
        data = read_json(path)
        if data is not None:
            results.append(data)
    return results


def delete_json(path: Path) -> None:
    path.unlink(missing_ok=True)
