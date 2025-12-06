from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import pandas as pd

from medical_log_parser.schema.model import Schema
from medical_log_parser.schema.flattener import SchemaFlattener
from .strategies import UnionResolutionStrategy, ValueTransformStrategy

logger = logging.getLogger(__name__)


class LogParser:
    def __init__(
        self,
        schema: Schema,
        flattener: SchemaFlattener,
        union_strategy: UnionResolutionStrategy,
        value_strategy: ValueTransformStrategy,
    ) -> None:
        self.schema = schema
        self.flattener = flattener
        self.union_strategy = union_strategy
        self.value_strategy = value_strategy

    def parse_dataframe(
        self,
        df: pd.DataFrame,
        root_struct: str = "MedicalLog",
        log_name_col: str = "log_name",
        struct_name_col: str = "StructName",
    ) -> Tuple[pd.DataFrame, List[str]]:
        if df is None or df.empty:
            return (
                pd.DataFrame(columns=["StructName", "LogName", "MemberName", "Value"]),
                [],
            )

        flat_cols, union_groups = self.flattener.build_flat_columns(self.schema, root_struct)
        known_set = set(flat_cols)
        ignore_cols = {log_name_col, struct_name_col}

        unknown_columns = sorted(
            c for c in df.columns if c not in known_set and c not in ignore_cols
        )
        if unknown_columns:
            logger.warning("Unknown columns ignored: %s", unknown_columns)

        if log_name_col not in df.columns:
            df = df.copy()
            df[log_name_col] = [f"log_{i+1}" for i in range(len(df))]

        records: List[Dict[str, object]] = []

        for idx, row in df.iterrows():
            struct_name = row.get(struct_name_col, root_struct) or root_struct
            log_name = row.get(log_name_col, f"log_{idx+1}") or f"log_{idx+1}"

            row_dict = row.to_dict()

            for col, value in row_dict.items():
                if col in ignore_cols:
                    continue
                if col not in known_set:
                    continue
                if pd.isna(value):
                    continue

                value = self.value_strategy.transform(struct_name, col, value)

                records.append(
                    {
                        "StructName": struct_name,
                        "LogName": str(log_name),
                        "MemberName": col,
                        "Value": value,
                    }
                )

        parsed_df = pd.DataFrame.from_records(
            records, columns=["StructName", "LogName", "MemberName", "Value"]
        )
        return parsed_df, unknown_columns

    @staticmethod
    def to_wide_dataframe(parsed_df: pd.DataFrame) -> pd.DataFrame:
        if parsed_df.empty:
            return pd.DataFrame()
        required = {"StructName", "LogName", "MemberName", "Value"}
        if not required.issubset(parsed_df.columns):
            raise ValueError(f"parsed_df must contain columns: {required}")
        wide_df = parsed_df.pivot_table(
            index=["StructName", "LogName"],
            columns="MemberName",
            values="Value",
            aggfunc="first",
        ).reset_index()
        wide_df.columns.name = None
        return wide_df
