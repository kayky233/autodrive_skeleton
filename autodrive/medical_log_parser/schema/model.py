from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Literal, Any


Cardinality = Literal["scalar", "array"]


@dataclass
class FieldSchema:
    name: str
    ctype: str
    cardinality: Cardinality = "scalar"
    length: int = 1
    is_char_array: bool = False
    is_union: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "FieldSchema":
        return FieldSchema(**data)


@dataclass
class StructSchema:
    name: str
    kind: Literal["struct"] = "struct"
    fields: List[FieldSchema] = field(default_factory=list)
    unions: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "fields": [f.to_dict() for f in self.fields],
            "unions": self.unions,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "StructSchema":
        fields = [FieldSchema.from_dict(f) for f in data.get("fields", [])]
        unions = data.get("unions", {})
        return StructSchema(
            name=data["name"],
            kind=data.get("kind", "struct"),
            fields=fields,
            unions=unions,
        )


@dataclass
class UnionSchema:
    name: str
    kind: Literal["union"] = "union"
    fields: List[FieldSchema] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "fields": [f.to_dict() for f in self.fields],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "UnionSchema":
        fields = [FieldSchema.from_dict(f) for f in data.get("fields", [])]
        return UnionSchema(
            name=data["name"],
            kind=data.get("kind", "union"),
            fields=fields,
        )


@dataclass
class Schema:
    structs: Dict[str, StructSchema] = field(default_factory=dict)
    unions: Dict[str, UnionSchema] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structs": {name: st.to_dict() for name, st in self.structs.items()},
            "unions": {name: un.to_dict() for name, un in self.unions.items()},
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Schema":
        structs = {
            name: StructSchema.from_dict(st)
            for name, st in data.get("structs", {}).items()
        }
        unions = {
            name: UnionSchema.from_dict(un)
            for name, un in data.get("unions", {}).items()
        }
        return Schema(structs=structs, unions=unions)

    def get_struct(self, name: str) -> StructSchema:
        return self.structs[name]

    def get_union(self, name: str) -> UnionSchema:
        return self.unions[name]
