from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


class FileReaderStrategy(ABC):
    @abstractmethod
    def can_handle(self, path: Path) -> bool:
        ...

    @abstractmethod
    def read(self, path: Path) -> pd.DataFrame:
        ...


class ExcelFileReader(FileReaderStrategy):
    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in {".xlsx", ".xls"}

    def read(self, path: Path) -> pd.DataFrame:
        return pd.read_excel(path)


class CsvFileReader(FileReaderStrategy):
    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".csv"

    def read(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)


class FileReaderRegistry:
    def __init__(self) -> None:
        self._strategies: List[FileReaderStrategy] = []

    def register(self, strategy: FileReaderStrategy) -> None:
        self._strategies.append(strategy)

    def read(self, path: Path) -> pd.DataFrame:
        for s in self._strategies:
            if s.can_handle(path):
                return s.read(path)
        raise ValueError(f"No file reader registered for {path.suffix}")
