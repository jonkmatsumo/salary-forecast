"""Tests for preprocessing encoder functionality (caching removed for performance)."""

import pandas as pd

from src.xgboost.preprocessing import RankedCategoryEncoder


class TestPreprocessingEncoders:
    """Test suite for preprocessing encoder functionality."""

    def test_ranked_category_encoder_basic_transform(self) -> None:
        """Test RankedCategoryEncoder transforms categories correctly."""
        mapping = {"A": 1, "B": 2, "C": 3}
        encoder = RankedCategoryEncoder(mapping=mapping)

        data = pd.Series(["A", "B", "C", "A"])
        result = encoder.transform(data)

        assert result.tolist() == [1, 2, 3, 1]

    def test_ranked_category_encoder_different_data(self) -> None:
        """Test RankedCategoryEncoder handles different data correctly."""
        mapping = {"A": 1, "B": 2, "C": 3}
        encoder = RankedCategoryEncoder(mapping=mapping)

        data1 = pd.Series(["A", "B"])
        data2 = pd.Series(["C", "A"])

        result1 = encoder.transform(data1)
        result2 = encoder.transform(data2)

        assert result1.tolist() == [1, 2]
        assert result2.tolist() == [3, 1]

    def test_ranked_category_encoder_different_mapping(self) -> None:
        """Test RankedCategoryEncoder uses correct mapping."""
        mapping1 = {"A": 1, "B": 2}
        mapping2 = {"A": 10, "B": 20}

        encoder1 = RankedCategoryEncoder(mapping=mapping1)
        encoder2 = RankedCategoryEncoder(mapping=mapping2)

        data = pd.Series(["A", "B"])
        result1 = encoder1.transform(data)
        result2 = encoder2.transform(data)

        assert result1.tolist() == [1, 2]
        assert result2.tolist() == [10, 20]

    def test_ranked_category_encoder_with_dataframe(self) -> None:
        """Test RankedCategoryEncoder with DataFrame input."""
        mapping = {"X": 1, "Y": 2}
        encoder = RankedCategoryEncoder(mapping=mapping)

        df = pd.DataFrame({"col": ["X", "Y", "X"]})
        result = encoder.transform(df)

        assert result.tolist() == [1, 2, 1]

    def test_ranked_category_encoder_with_list(self) -> None:
        """Test RankedCategoryEncoder with list input."""
        mapping = {"A": 1, "B": 2}
        encoder = RankedCategoryEncoder(mapping=mapping)

        data = ["A", "B", "A"]
        result = encoder.transform(data)

        assert result.tolist() == [1, 2, 1]

    def test_ranked_category_encoder_unknown_values(self) -> None:
        """Test RankedCategoryEncoder handles unknown values correctly."""
        mapping = {"A": 1, "B": 2}
        encoder = RankedCategoryEncoder(mapping=mapping)

        data = pd.Series(["A", "B", "C", "D"])
        result = encoder.transform(data)

        assert result.tolist() == [1, 2, -1, -1]

    def test_ranked_category_encoder_consistency(self) -> None:
        """Test RankedCategoryEncoder produces consistent results."""
        mapping = {"A": 1, "B": 2}
        encoder = RankedCategoryEncoder(mapping=mapping)

        data1 = pd.Series(["A", "B"])
        data2 = pd.Series(["A", "B", "A"])

        result1 = encoder.transform(data1)
        result2 = encoder.transform(data2)

        assert result1.tolist() == [1, 2]
        assert result2.tolist() == [1, 2, 1]

    def test_multiple_encoders_independent(self) -> None:
        """Test multiple encoders work independently."""
        encoder1 = RankedCategoryEncoder(mapping={"X": 1, "Y": 2})
        encoder2 = RankedCategoryEncoder(mapping={"P": 10, "Q": 20})

        data1 = pd.Series(["X", "Y"])
        data2 = pd.Series(["P", "Q"])

        result1 = encoder1.transform(data1)
        result2 = encoder2.transform(data2)

        assert result1.tolist() == [1, 2]
        assert result2.tolist() == [10, 20]
