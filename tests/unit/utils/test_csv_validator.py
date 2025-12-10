import pytest
import io
import pandas as pd
from src.utils.csv_validator import validate_csv

def test_valid_csv():
    data = "Col1,Col2\n1,2\n3,4"
    f = io.BytesIO(data.encode('utf-8'))
    is_valid, err, df = validate_csv(f)
    assert is_valid
    assert err is None
    assert len(df) == 2

def test_empty_file():
    f = io.BytesIO(b"")
    is_valid, err, df = validate_csv(f)
    assert not is_valid
    assert "File is empty" in err

def test_too_few_columns():
    data = "Col1\n1\n2"
    f = io.BytesIO(data.encode('utf-8'))
    is_valid, err, df = validate_csv(f)
    assert not is_valid
    assert "must have at least 2 columns" in err

def test_parsing_error():
    """Verify proper error handling for binary or corrupted files."""
    f = io.BytesIO(b"\xff\xff")
    is_valid, err, df = validate_csv(f)
    assert not is_valid
    assert "Failed to parse CSV" in err


def test_csv_with_missing_values():
    """Verify missing values are acceptable in CSV data."""
    data = "Col1,Col2\n1,2\n,4\n3,"
    f = io.BytesIO(data.encode('utf-8'))
    is_valid, err, df = validate_csv(f)
    assert is_valid
    assert err is None
    assert len(df) == 3


def test_csv_single_row():
    """Verify CSV validation requires at least one data row."""
    data = "Col1,Col2"
    f = io.BytesIO(data.encode('utf-8'))
    is_valid, err, df = validate_csv(f)
    if not is_valid:
        assert err is not None
    else:
        assert len(df) == 0


def test_csv_unicode_characters():
    """Verify CSV validator handles international characters correctly."""
    data = "Name,Salary\nJosé,100000\nMüller,200000"
    f = io.BytesIO(data.encode('utf-8'))
    is_valid, err, df = validate_csv(f)
    assert is_valid
    assert err is None
    assert len(df) == 2
