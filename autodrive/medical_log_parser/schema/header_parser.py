from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

from .model import Schema, StructSchema, UnionSchema, FieldSchema

logger = logging.getLogger(__name__)


class HeaderSchemaExtractor:
    STRUCT_PATTERN = re.compile(
        r"typedef\s+struct\s*\{(?P<body>.*?)\}\s*(?P<name>\w+)\s*;",
        re.DOTALL,
    )
    UNION_PATTERN = re.compile(
        r"typedef\s+union\s*\{(?P<body>.*?)\}\s*(?P<name>\w+)\s*;",
        re.DOTALL,
    )
    FIELD_PATTERN = re.compile(
        r"(?P<ctype>[_a-zA-Z][\w\s\*]*)\s+"
        r"(?P<name>[_a-zA-Z]\w*)"
        r"(?:\[(?P<length>\d+)\])?\s*;",
        re.MULTILINE,
    )

    def extract_from_directory(self, directory: str) -> Schema:
        root = Path(directory)
        if not root.exists():
            raise FileNotFoundError(f"Header directory not found: {directory}")

        structs_raw: Dict[str, List[Tuple[str, str, Optional[int]]]] = {}
        unions_raw: Dict[str, List[Tuple[str, str, Optional[int]]]] = {}

        for header in root.rglob("*.h"):
            try:
                text = header.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to read header %s: %s", header, e)
                continue

            for m in self.STRUCT_PATTERN.finditer(text):
                body = m.group("body") or ""
                name = m.group("name").strip()
                fields = self._parse_fields_from_body(body, header)
                structs_raw[name] = fields

            for m in self.UNION_PATTERN.finditer(text):
                body = m.group("body") or ""
                name = m.group("name").strip()
                fields = self._parse_fields_from_body(body, header)
                unions_raw[name] = fields

        if not structs_raw and not unions_raw:
            raise ValueError(f"No struct/union definitions found under {directory}")

        schema = Schema()

        for uname, fields in unions_raw.items():
            u_fields = [
                self._field_to_schema(ctype, fname, length, unions_raw)
                for (ctype, fname, length) in fields
            ]
            schema.unions[uname] = UnionSchema(name=uname, fields=u_fields)

        for sname, fields in structs_raw.items():
            s_fields: List[FieldSchema] = []
            unions_meta: Dict[str, List[str]] = {}
            for ctype, fname, length in fields:
                is_union = ctype.strip() in unions_raw
                field_schema = self._field_to_schema(ctype, fname, length, unions_raw)
                s_fields.append(field_schema)

                if is_union:
                    union_def = schema.unions.get(ctype.strip())
                    if not union_def:
                        continue
                    group_fields: List[str] = []
                    for uf in union_def.fields:
                        group_fields.append(f"{fname}_{uf.name}")
                    unions_meta[fname] = group_fields

            schema.structs[sname] = StructSchema(
                name=sname,
                fields=s_fields,
                unions=unions_meta,
            )

        return schema

    def _parse_fields_from_body(
        self, body: str, header_path: Path
    ) -> List[Tuple[str, str, Optional[int]]]:
        fields: List[Tuple[str, str, Optional[int]]] = []
        for line in body.splitlines():
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith("//") or line_stripped.startswith("/*"):
                continue
            m = self.FIELD_PATTERN.search(line_stripped)
            if not m:
                logger.warning("[%s] Unrecognized field line: %r", header_path.name, line_stripped)
                continue
            ctype = (m.group("ctype") or "").strip()
            name = (m.group("name") or "").strip()
            length_str = m.group("length")
            length = int(length_str) if length_str is not None else None
            if not ctype or not name:
                logger.warning("[%s] Empty ctype or name: %r", header_path.name, line_stripped)
                continue
            fields.append((ctype, name, length))
        return fields

    def _field_to_schema(
        self,
        ctype: str,
        name: str,
        length: Optional[int],
        unions_raw: Dict[str, List[Tuple[str, str, Optional[int]]]],
    ) -> FieldSchema:
        ctype_clean = " ".join(ctype.split())
        card: Literal["scalar", "array"] = "array" if length not in (None, 1) else "scalar"
        length_val = 1 if length is None else int(length)

        is_char_array = False
        if ctype_clean.startswith("char") and card == "array":
            is_char_array = True

        is_union_type = ctype_clean in unions_raw

        return FieldSchema(
            name=name,
            ctype=ctype_clean if not is_char_array else f"char[{length_val}]",
            cardinality=card,
            length=length_val,
            is_char_array=is_char_array,
            is_union=is_union_type,
        )
