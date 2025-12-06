from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .schema.header_parser import HeaderSchemaExtractor
from .schema.repository import SchemaRepository
from .schema.flattener import SchemaFlattener
from .schema.model import Schema
from .parsing.strategies import (
    DefaultUnionResolutionStrategy,
    DefaultValueTransformStrategy,
)
from .parsing.log_parser import LogParser
from .io.file_readers import FileReaderRegistry, ExcelFileReader, CsvFileReader

logger = logging.getLogger(__name__)


class MedicalLogFacade:
    """
    对外统一入口：你只需要跟这个类交互。
    """

    def __init__(self) -> None:
        self.header_extractor = HeaderSchemaExtractor()
        self.schema_repo = SchemaRepository()
        self.flattener = SchemaFlattener()
        self.file_readers = FileReaderRegistry()
        self.file_readers.register(ExcelFileReader())
        self.file_readers.register(CsvFileReader())

    def build_schema_from_headers(self, header_dir: str, out_path: str) -> Schema:
        schema = self.header_extractor.extract_from_directory(header_dir)
        self.schema_repo.save(schema, out_path)
        return schema

    def load_schema(self, path: str) -> Schema:
        return self.schema_repo.load(path)

    def parse_log_directory(
        self,
        log_dir: str,
        schema: Schema,
        root_struct: str = "MedicalLog",
        pattern: str = "*.xlsx",
        log_name_col: str = "log_name",
        struct_name_col: str = "StructName",
        enum_map: Dict[str, Dict[int, str]] | None = None,
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        log_root = Path(log_dir)
        if not log_root.exists():
            raise FileNotFoundError(f"Log directory not found: {log_dir}")

        union_strategy = DefaultUnionResolutionStrategy()
        value_strategy = DefaultValueTransformStrategy(enum_map=enum_map)
        log_parser = LogParser(schema, self.flattener, union_strategy, value_strategy)

        all_parsed: List[pd.DataFrame] = []
        unknown_map: Dict[str, List[str]] = {}

        for file in sorted(log_root.rglob(pattern)):
            try:
                df = self.file_readers.read(file)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to read log file %s: %s", file, e)
                continue

            if df.empty:
                logger.warning("Log file %s is empty.", file)
                continue

            parsed_df, unknown_cols = log_parser.parse_dataframe(
                df,
                root_struct=root_struct,
                log_name_col=log_name_col,
                struct_name_col=struct_name_col,
            )
            if parsed_df.empty:
                logger.warning("Log file %s produced no parsed rows.", file)
                continue

            parsed_df = parsed_df.copy()
            parsed_df["SourceFile"] = str(file)
            all_parsed.append(parsed_df)
            unknown_map[str(file)] = unknown_cols

        if not all_parsed:
            return (
                pd.DataFrame(columns=["StructName", "LogName", "MemberName", "Value", "SourceFile"]),
                unknown_map,
            )

        merged = pd.concat(all_parsed, ignore_index=True)
        return merged, unknown_map
