"""Simple in-memory storage for datasets (MVP implementation)."""

import threading
from typing import Dict, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetStorage:
    """In-memory dataset storage for API (MVP implementation). In production, replace with database."""

    def __init__(self):
        """Initialize dataset storage."""
        self._datasets: Dict[str, pd.DataFrame] = {}
        self._lock = threading.Lock()

    def store(self, dataset_id: str, df: pd.DataFrame) -> None:
        """Store a dataset.

        Args:
            dataset_id (str): Dataset identifier.
            df (pd.DataFrame): DataFrame to store.
        """
        with self._lock:
            self._datasets[dataset_id] = df
            logger.debug(f"Stored dataset {dataset_id} with {len(df)} rows")

    def get(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Get a dataset.

        Args:
            dataset_id (str): Dataset identifier.

        Returns:
            Optional[pd.DataFrame]: DataFrame or None if not found.
        """
        with self._lock:
            return self._datasets.get(dataset_id)

    def delete(self, dataset_id: str) -> bool:
        """Delete a dataset.

        Args:
            dataset_id (str): Dataset identifier.

        Returns:
            bool: True if deleted, False if not found.
        """
        with self._lock:
            if dataset_id in self._datasets:
                del self._datasets[dataset_id]
                logger.debug(f"Deleted dataset {dataset_id}")
                return True
            return False


_dataset_storage = DatasetStorage()


def get_dataset_storage() -> DatasetStorage:
    """Get the global dataset storage instance.

    Returns:
        DatasetStorage: Storage instance.
    """
    return _dataset_storage
