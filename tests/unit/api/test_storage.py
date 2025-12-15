"""Unit tests for dataset storage."""

import threading
from unittest.mock import patch

import pandas as pd

from src.api.storage import DatasetStorage, get_dataset_storage


class TestDatasetStorageDelete:
    """Tests for DatasetStorage.delete method."""

    def test_delete_successful_deletion(self):
        """Test successful deletion of a dataset."""
        storage = DatasetStorage()
        dataset_id = "test_dataset_1"
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        storage.store(dataset_id, df)
        assert storage.get(dataset_id) is not None

        result = storage.delete(dataset_id)

        assert result is True
        assert storage.get(dataset_id) is None

    def test_delete_non_existent_dataset(self):
        """Test deletion of non-existent dataset."""
        storage = DatasetStorage()
        non_existent_id = "non_existent_dataset"

        result = storage.delete(non_existent_id)

        assert result is False
        assert storage.get(non_existent_id) is None

    @patch("src.api.storage.logger")
    def test_delete_logs_deletion(self, mock_logger):
        """Test that deletion is logged."""
        storage = DatasetStorage()
        dataset_id = "test_dataset_2"
        df = pd.DataFrame({"col1": [1]})

        storage.store(dataset_id, df)
        storage.delete(dataset_id)

        assert mock_logger.debug.call_count >= 1
        call_args_list = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any(f"Deleted dataset {dataset_id}" in msg for msg in call_args_list)

    def test_delete_thread_safety(self):
        """Test thread safety with concurrent access."""
        storage = DatasetStorage()
        dataset_ids = [f"dataset_{i}" for i in range(10)]
        dfs = [pd.DataFrame({"col": [i]}) for i in range(10)]

        for dataset_id, df in zip(dataset_ids, dfs):
            storage.store(dataset_id, df)

        def delete_datasets(ids):
            for dataset_id in ids:
                storage.delete(dataset_id)

        thread1 = threading.Thread(target=delete_datasets, args=(dataset_ids[:5],))
        thread2 = threading.Thread(target=delete_datasets, args=(dataset_ids[5:],))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        for dataset_id in dataset_ids:
            assert storage.get(dataset_id) is None

    def test_delete_multiple_times_same_id(self):
        """Test deleting the same dataset multiple times."""
        storage = DatasetStorage()
        dataset_id = "test_dataset_3"
        df = pd.DataFrame({"col1": [1]})

        storage.store(dataset_id, df)
        assert storage.delete(dataset_id) is True
        assert storage.delete(dataset_id) is False
        assert storage.delete(dataset_id) is False


class TestGetDatasetStorage:
    """Tests for get_dataset_storage function."""

    def test_get_dataset_storage_returns_singleton(self):
        """Test that get_dataset_storage returns the same instance."""
        storage1 = get_dataset_storage()
        storage2 = get_dataset_storage()

        assert storage1 is storage2

    def test_get_dataset_storage_is_shared(self):
        """Test that operations on one reference affect all references."""
        storage1 = get_dataset_storage()
        storage2 = get_dataset_storage()

        dataset_id = "shared_dataset"
        df = pd.DataFrame({"col": [1, 2, 3]})

        storage1.store(dataset_id, df)
        assert storage2.get(dataset_id) is not None
        assert storage2.get(dataset_id).equals(df)
