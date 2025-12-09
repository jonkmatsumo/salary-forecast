"""
Analysis tools for LangGraph agents.

These tools are used by agents to explore and analyze data during the
configuration generation workflow.
"""

import re
import json
from typing import Any, Optional
import pandas as pd
import numpy as np
from langchain_core.tools import tool


@tool
def compute_correlation_matrix(df_json: str, columns: Optional[str] = None) -> str:
    """
    Compute pairwise Pearson correlation coefficients between numeric columns.
    
    Args:
        df_json: JSON string representation of the DataFrame (use df.to_json()).
        columns: Optional comma-separated list of column names to include.
                 If not provided, uses all numeric columns.
    
    Returns:
        JSON string of correlation matrix with column pairs and their correlations.
    """
    df = pd.read_json(df_json)
    
    if columns:
        col_list = [c.strip() for c in columns.split(",")]
        # Filter to only existing numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        col_list = [c for c in col_list if c in numeric_cols]
    else:
        col_list = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(col_list) < 2:
        return json.dumps({"error": "Need at least 2 numeric columns for correlation"})
    
    corr_matrix = df[col_list].corr()
    
    # Convert to a more readable format
    correlations = []
    for i, col1 in enumerate(col_list):
        for col2 in col_list[i+1:]:
            correlations.append({
                "column_1": col1,
                "column_2": col2,
                "correlation": round(corr_matrix.loc[col1, col2], 4)
            })
    
    return json.dumps({
        "columns_analyzed": col_list,
        "correlations": correlations,
        "full_matrix": corr_matrix.to_dict()
    }, indent=2)


@tool
def get_column_statistics(df_json: str, column: str) -> str:
    """
    Get detailed statistics for a specific column.
    
    Args:
        df_json: JSON string representation of the DataFrame.
        column: Name of the column to analyze.
    
    Returns:
        JSON string with statistics including dtype, nulls, unique count,
        and numeric stats (mean, std, min, max, quartiles) if applicable.
    """
    df = pd.read_json(df_json)
    
    if column not in df.columns:
        return json.dumps({"error": f"Column '{column}' not found in DataFrame"})
    
    col_data = df[column]
    
    stats = {
        "column": column,
        "dtype": str(col_data.dtype),
        "total_count": len(col_data),
        "null_count": int(col_data.isnull().sum()),
        "null_percentage": round(col_data.isnull().sum() / len(col_data) * 100, 2),
        "unique_count": int(col_data.nunique()),
    }
    
    # Add numeric statistics if applicable
    if pd.api.types.is_numeric_dtype(col_data):
        stats["numeric_stats"] = {
            "mean": round(col_data.mean(), 4) if not col_data.isnull().all() else None,
            "std": round(col_data.std(), 4) if not col_data.isnull().all() else None,
            "min": round(col_data.min(), 4) if not col_data.isnull().all() else None,
            "max": round(col_data.max(), 4) if not col_data.isnull().all() else None,
            "median": round(col_data.median(), 4) if not col_data.isnull().all() else None,
            "q25": round(col_data.quantile(0.25), 4) if not col_data.isnull().all() else None,
            "q75": round(col_data.quantile(0.75), 4) if not col_data.isnull().all() else None,
        }
    
    # Add sample values
    non_null = col_data.dropna()
    if len(non_null) > 0:
        sample_values = non_null.head(5).tolist()
        stats["sample_values"] = [str(v) for v in sample_values]
    
    return json.dumps(stats, indent=2)


@tool
def get_unique_value_counts(df_json: str, column: str, limit: int = 20) -> str:
    """
    Get unique values and their counts for a column.
    
    Args:
        df_json: JSON string representation of the DataFrame.
        column: Name of the column to analyze.
        limit: Maximum number of unique values to return (default 20).
    
    Returns:
        JSON string with value counts, sorted by frequency.
    """
    df = pd.read_json(df_json)
    
    if column not in df.columns:
        return json.dumps({"error": f"Column '{column}' not found in DataFrame"})
    
    value_counts = df[column].value_counts()
    total_unique = len(value_counts)
    
    # Limit output
    value_counts = value_counts.head(limit)
    
    result = {
        "column": column,
        "total_unique_values": total_unique,
        "showing_top": min(limit, total_unique),
        "value_counts": [
            {"value": str(val), "count": int(count), "percentage": round(count / len(df) * 100, 2)}
            for val, count in value_counts.items()
        ]
    }
    
    return json.dumps(result, indent=2)


@tool
def detect_ordinal_patterns(df_json: str, column: str) -> str:
    """
    Detect if a column contains ordinal patterns (e.g., "Level 1", "L2", "Senior", "Junior").
    
    Checks for:
    - Numeric patterns in strings (e.g., "Level 1", "L2", "Grade 3")
    - Common ordinal keywords (junior, senior, entry, mid, etc.)
    - Roman numerals (I, II, III, IV, V)
    
    Args:
        df_json: JSON string representation of the DataFrame.
        column: Name of the column to analyze.
    
    Returns:
        JSON string with detected patterns and suggested ordinal mapping.
    """
    df = pd.read_json(df_json)
    
    if column not in df.columns:
        return json.dumps({"error": f"Column '{column}' not found in DataFrame"})
    
    unique_values = df[column].dropna().unique().tolist()
    unique_values = [str(v) for v in unique_values]
    
    result = {
        "column": column,
        "unique_values": unique_values,
        "patterns_detected": [],
        "is_ordinal": False,
        "suggested_mapping": {}
    }
    
    # Pattern 1: Numeric extraction (Level 1, L2, Grade 3, etc.)
    numeric_pattern = re.compile(r'(\d+)')
    values_with_numbers = []
    for val in unique_values:
        match = numeric_pattern.search(val)
        if match:
            values_with_numbers.append((val, int(match.group(1))))
    
    if len(values_with_numbers) > len(unique_values) * 0.5:
        result["patterns_detected"].append("numeric_in_string")
        result["is_ordinal"] = True
        # Sort by extracted number
        sorted_values = sorted(values_with_numbers, key=lambda x: x[1])
        result["suggested_mapping"] = {v[0]: i for i, v in enumerate(sorted_values)}
    
    # Pattern 2: Common ordinal keywords
    ordinal_keywords = {
        "intern": 0, "entry": 1, "junior": 2, "associate": 3,
        "mid": 4, "senior": 5, "staff": 6, "principal": 7,
        "lead": 8, "director": 9, "vp": 10, "executive": 11
    }
    
    keyword_matches = []
    for val in unique_values:
        val_lower = val.lower()
        for keyword, rank in ordinal_keywords.items():
            if keyword in val_lower:
                keyword_matches.append((val, rank, keyword))
                break
    
    if len(keyword_matches) > len(unique_values) * 0.3 and not result["suggested_mapping"]:
        result["patterns_detected"].append("ordinal_keywords")
        result["is_ordinal"] = True
        sorted_matches = sorted(keyword_matches, key=lambda x: x[1])
        result["suggested_mapping"] = {v[0]: i for i, v in enumerate(sorted_matches)}
        result["matched_keywords"] = [(v[0], v[2]) for v in sorted_matches]
    
    # Pattern 3: Roman numerals
    roman_pattern = re.compile(r'\b(I{1,3}|IV|V|VI{0,3}|IX|X)\b', re.IGNORECASE)
    roman_values = {
        'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
        'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10
    }
    
    values_with_roman = []
    for val in unique_values:
        match = roman_pattern.search(val)
        if match:
            roman = match.group(1).upper()
            if roman in roman_values:
                values_with_roman.append((val, roman_values[roman]))
    
    if len(values_with_roman) > len(unique_values) * 0.3 and not result["suggested_mapping"]:
        result["patterns_detected"].append("roman_numerals")
        result["is_ordinal"] = True
        sorted_values = sorted(values_with_roman, key=lambda x: x[1])
        result["suggested_mapping"] = {v[0]: i for i, v in enumerate(sorted_values)}
    
    return json.dumps(result, indent=2)


@tool
def detect_column_dtype(df_json: str, column: str) -> str:
    """
    Infer the semantic data type of a column.
    
    Categories:
    - numeric_continuous: Float values, likely continuous measurements
    - numeric_discrete: Integer values, possibly counts or IDs
    - categorical: String/object with limited unique values
    - datetime: Date or timestamp values
    - text: String with high cardinality (possibly free text)
    - identifier: Likely an ID column (unique values, numeric pattern)
    - boolean: True/False or binary values
    
    Args:
        df_json: JSON string representation of the DataFrame.
        column: Name of the column to analyze.
    
    Returns:
        JSON string with inferred semantic type and reasoning.
    """
    df = pd.read_json(df_json)
    
    if column not in df.columns:
        return json.dumps({"error": f"Column '{column}' not found in DataFrame"})
    
    col_data = df[column]
    dtype = str(col_data.dtype)
    unique_count = col_data.nunique()
    total_count = len(col_data)
    unique_ratio = unique_count / total_count if total_count > 0 else 0
    
    result = {
        "column": column,
        "pandas_dtype": dtype,
        "unique_count": unique_count,
        "unique_ratio": round(unique_ratio, 4),
        "semantic_type": None,
        "reasoning": []
    }
    
    # Check for datetime
    if pd.api.types.is_datetime64_any_dtype(col_data):
        result["semantic_type"] = "datetime"
        result["reasoning"].append("Column has datetime dtype")
        return json.dumps(result, indent=2)
    
    # Check for boolean
    if pd.api.types.is_bool_dtype(col_data) or set(col_data.dropna().unique()).issubset({True, False, 0, 1, "True", "False", "true", "false"}):
        result["semantic_type"] = "boolean"
        result["reasoning"].append("Column contains only boolean-like values")
        return json.dumps(result, indent=2)
    
    # Check for numeric types
    if pd.api.types.is_numeric_dtype(col_data):
        # Check if it's likely an ID column
        col_name_lower = column.lower()
        if unique_ratio > 0.9 and ("id" in col_name_lower or "key" in col_name_lower or "index" in col_name_lower):
            result["semantic_type"] = "identifier"
            result["reasoning"].append("High uniqueness ratio and column name suggests ID")
        elif pd.api.types.is_float_dtype(col_data):
            result["semantic_type"] = "numeric_continuous"
            result["reasoning"].append("Float dtype suggests continuous values")
        elif pd.api.types.is_integer_dtype(col_data):
            if unique_count < 20:
                result["semantic_type"] = "categorical"
                result["reasoning"].append("Integer with few unique values, likely categorical")
            else:
                result["semantic_type"] = "numeric_discrete"
                result["reasoning"].append("Integer dtype with many unique values")
        return json.dumps(result, indent=2)
    
    # String/object types
    if pd.api.types.is_object_dtype(col_data) or pd.api.types.is_string_dtype(col_data):
        # Try to parse as datetime
        try:
            parsed = pd.to_datetime(col_data.dropna().head(10), errors='coerce')
            if parsed.notna().sum() > 5:
                result["semantic_type"] = "datetime"
                result["reasoning"].append("String values parseable as dates")
                return json.dumps(result, indent=2)
        except:
            pass
        
        # Check uniqueness for categorical vs text
        if unique_ratio > 0.8:
            # Check if looks like ID
            col_name_lower = column.lower()
            if "id" in col_name_lower or "key" in col_name_lower or "name" in col_name_lower:
                result["semantic_type"] = "identifier"
                result["reasoning"].append("High uniqueness and name suggests identifier")
            else:
                result["semantic_type"] = "text"
                result["reasoning"].append("High cardinality string, likely free text")
        else:
            result["semantic_type"] = "categorical"
            result["reasoning"].append(f"String with {unique_count} unique values ({unique_ratio:.1%} unique)")
    
    if result["semantic_type"] is None:
        result["semantic_type"] = "unknown"
        result["reasoning"].append("Could not determine semantic type")
    
    return json.dumps(result, indent=2)


def get_all_tools():
    """Return list of all analysis tools for use with agents."""
    return [
        compute_correlation_matrix,
        get_column_statistics,
        get_unique_value_counts,
        detect_ordinal_patterns,
        detect_column_dtype,
    ]

