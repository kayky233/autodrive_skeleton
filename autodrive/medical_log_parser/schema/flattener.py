from __future__ import annotations

from typing import Dict, List, Tuple

from .model import Schema, StructSchema


class SchemaFlattener:
    """
    负责把 Schema 中某个根 struct 扁平化为列名列表 + union 分组信息。
    """

    def build_flat_columns(
        self,
        schema: Schema,
        root_struct: str,
        prefix: str = "",
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        if root_struct not in schema.structs:
            raise KeyError(f"Root struct {root_struct!r} not found in schema.")

        flat_cols: List[str] = []
        union_groups: Dict[str, List[str]] = {}

        def _walk_struct(struct: StructSchema, pfx: str) -> None:
            for field in struct.fields:
                fname = field.name
                ftype = field.ctype
                card = field.cardinality
                length = field.length
                is_char_array = field.is_char_array
                is_union = field.is_union

                base_name = f"{pfx}{fname}" if not pfx else f"{pfx}_{fname}"

                if is_union and ftype in schema.unions:
                    union_def = schema.unions[ftype]
                    group_fields: List[str] = []
                    for uf in union_def.fields:
                        member_name = uf.name
                        col = f"{base_name}_{member_name}"
                        flat_cols.append(col)
                        group_fields.append(col)
                    union_groups[base_name] = group_fields
                    continue

                if is_char_array:
                    flat_cols.append(base_name)
                    continue

                if card == "array":
                    if ftype in schema.structs:
                        sub_struct = schema.structs[ftype]
                        for i in range(length):
                            sub_pfx = f"{base_name}_{i}"
                            _walk_struct(sub_struct, sub_pfx)
                    elif ftype in schema.unions:
                        union_def = schema.unions[ftype]
                        for i in range(length):
                            for uf in union_def.fields:
                                member_name = uf.name
                                col = f"{base_name}_{i}_{member_name}"
                                flat_cols.append(col)
                    else:
                        for i in range(length):
                            flat_cols.append(f"{base_name}_{i}")
                    continue

                if ftype in schema.structs:
                    sub_struct = schema.structs[ftype]
                    _walk_struct(sub_struct, base_name)
                else:
                    flat_cols.append(base_name)

        _walk_struct(schema.structs[root_struct], prefix)
        return flat_cols, union_groups
