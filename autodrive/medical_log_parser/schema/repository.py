from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .model import Schema


class SchemaRepository:
    """
    简单的 JSON 文件仓库，将来可以换成 DB / 远程配置中心。
    """

    def save(self, schema: Schema, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = schema.to_dict()
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str) -> Schema:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")
        text = p.read_text(encoding="utf-8")
        data: Dict[str, Any] = json.loads(text)
        return Schema.from_dict(data)
