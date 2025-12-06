from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class UnionResolutionStrategy(ABC):
    """
    决定 union 的哪个成员在当前行有效。
    """

    @abstractmethod
    def pick_member(
        self,
        struct_name: str,
        union_base_name: str,
        row_dict: Dict[str, Any],
        group_fields: List[str],
    ) -> Optional[str]:
        """
        返回 group_fields 中选中的列名；如果 None 表示无法确定。
        """
        ...


class DefaultUnionResolutionStrategy(UnionResolutionStrategy):
    """
    默认策略：如果有 tag 字段，则按 tag；否则取第一个非空字段。
    示例：
        union_base_name = "value"
        group_fields = ["value_int_value", "value_float_value", "value_str_value"]
        tag 字段尝试："value_type" 或 "value_tag"
    """

    TAG_SUFFIXES = ["_type", "_tag", "_kind"]

    def pick_member(
        self,
        struct_name: str,
        union_base_name: str,
        row_dict: Dict[str, Any],
        group_fields: List[str],
    ) -> Optional[str]:
        for suffix in self.TAG_SUFFIXES:
            tag_col = f"{union_base_name}{suffix}"
            if tag_col in row_dict and row_dict[tag_col] is not None:
                tag_val = str(row_dict[tag_col]).lower()
                for field in group_fields:
                    if "int" in field and tag_val.startswith("int"):
                        return field
                    if "float" in field and tag_val.startswith("float"):
                        return field
                    if "str" in field and "str" in tag_val:
                        return field
                break

        for field in group_fields:
            v = row_dict.get(field)
            if v is not None and v != "":
                return field
        return None


class ValueTransformStrategy(ABC):
    """
    值转换策略，比如枚举映射、单位换算等。
    """

    @abstractmethod
    def transform(self, struct_name: str, member_name: str, value: Any) -> Any:
        ...


class DefaultValueTransformStrategy(ValueTransformStrategy):
    """
    默认：支持简单枚举映射 + list/dict 序列化。
    """

    def __init__(self, enum_map: Optional[Dict[str, Dict[int, str]]] = None) -> None:
        self.enum_map = enum_map or {}

    def transform(self, struct_name: str, member_name: str, value: Any) -> Any:
        if isinstance(value, (list, dict, tuple)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, (int, float)) and struct_name in self.enum_map:
            iv = int(value)
            return self.enum_map[struct_name].get(iv, value)
        return value
