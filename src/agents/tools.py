"""Analysis tools for LangGraph agents used to explore and analyze data during the configuration generation workflow."""

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from src.utils.json_utils import parse_df_json_safely

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


@tool
def compute_correlation_matrix(df_json: str, columns: Optional[str] = None) -> str:
    """Compute pairwise Pearson correlation coefficients between numeric columns. Args: df_json (str): JSON DataFrame representation. columns (Optional[str]): Comma-separated column names. Returns: str: JSON correlation matrix."""
    from src.utils.logger import get_logger

    logger = get_logger(__name__)

    logger.debug(f"compute_correlation_matrix called with columns: {columns}")
    logger.debug(f"df_json type: {type(df_json)}, length: {len(df_json) if df_json else 0}")

    try:
        data_dict = parse_df_json_safely(df_json)
        logger.debug("Successfully parsed df_json for correlation matrix")
    except ValueError as e:
        logger.error(f"Failed to parse df_json as JSON in compute_correlation_matrix: {e}")
        logger.error(f"df_json preview: {df_json[:500] if df_json else 'None'}")
        try:
            error_detail = json.loads(str(e))
            return json.dumps(error_detail, indent=2)
        except Exception:
            return json.dumps(
                {
                    "error": f"Invalid JSON in df_json parameter: {str(e)}",
                    "error_type": "json_parse_error",
                }
            )

    try:
        df = pd.DataFrame.from_dict(data_dict)
        logger.debug(f"Created DataFrame for correlation (shape: {df.shape})")
    except Exception as e:
        logger.error(
            f"Failed to create DataFrame in compute_correlation_matrix: {e}", exc_info=True
        )
        return json.dumps({"error": f"Failed to create DataFrame: {str(e)}"})

    if columns:
        col_list = [c.strip() for c in columns.split(",")]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        col_list = [c for c in col_list if c in numeric_cols]
    else:
        col_list = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(col_list) < 2:
        return json.dumps({"error": "Need at least 2 numeric columns for correlation"})

    corr_matrix = df[col_list].corr()

    correlations = []
    for i, col1 in enumerate(col_list):
        for col2 in col_list[i + 1 :]:
            correlations.append(
                {
                    "column_1": col1,
                    "column_2": col2,
                    "correlation": round(corr_matrix.loc[col1, col2], 4),
                }
            )

    return json.dumps(
        {
            "columns_analyzed": col_list,
            "correlations": correlations,
            "full_matrix": corr_matrix.to_dict(),
        },
        indent=2,
    )


@tool
def get_column_statistics(df_json: str, column: str) -> str:
    """Get detailed statistics for a specific column. Args: df_json (str): JSON DataFrame representation. column (str): Column name. Returns: str: JSON statistics."""
    from src.utils.logger import get_logger

    logger = get_logger(__name__)

    logger.debug(f"get_column_statistics called with column: {column}")

    try:
        data_dict = parse_df_json_safely(df_json)
        logger.debug("Successfully parsed df_json for column statistics")
    except ValueError as e:
        logger.error(f"Failed to parse df_json as JSON in get_column_statistics: {e}")
        logger.error(f"df_json preview: {df_json[:500] if df_json else 'None'}")
        try:
            error_detail = json.loads(str(e))
            return json.dumps(error_detail, indent=2)
        except Exception:
            return json.dumps(
                {
                    "error": f"Invalid JSON in df_json parameter: {str(e)}",
                    "error_type": "json_parse_error",
                }
            )

    try:
        df = pd.DataFrame.from_dict(data_dict)
        logger.debug(f"Created DataFrame for column statistics (shape: {df.shape})")
    except Exception as e:
        logger.error(f"Failed to create DataFrame in get_column_statistics: {e}", exc_info=True)
        return json.dumps({"error": f"Failed to create DataFrame: {str(e)}"})

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

    if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):

        def safe_round(val: Any) -> Optional[float]:
            """Safely round values, handling numpy types. Args: val (Any): Value to round. Returns: Optional[float]: Rounded value or None."""
            if val is None or pd.isna(val):
                return None
            try:
                return float(round(float(val), 4))
            except (TypeError, ValueError):
                return float(val) if not pd.isna(val) else None

        stats["numeric_stats"] = {
            "mean": safe_round(col_data.mean()) if not col_data.isnull().all() else None,
            "std": safe_round(col_data.std()) if not col_data.isnull().all() else None,
            "min": safe_round(col_data.min()) if not col_data.isnull().all() else None,
            "max": safe_round(col_data.max()) if not col_data.isnull().all() else None,
            "median": safe_round(col_data.median()) if not col_data.isnull().all() else None,
            "q25": safe_round(col_data.quantile(0.25)) if not col_data.isnull().all() else None,
            "q75": safe_round(col_data.quantile(0.75)) if not col_data.isnull().all() else None,
        }
    elif pd.api.types.is_bool_dtype(col_data):
        stats["boolean_stats"] = {
            "true_count": int(col_data.sum()),
            "false_count": int((~col_data).sum()),
            "null_count": int(col_data.isnull().sum()),
        }

    non_null = col_data.dropna()
    if len(non_null) > 0:
        sample_values = non_null.head(5).tolist()
        stats["sample_values"] = [
            (
                str(v)
                if not isinstance(v, (np.integer, np.floating, np.bool_))
                else str(int(v) if isinstance(v, np.bool_) else v)
            )
            for v in sample_values
        ]

    return json.dumps(stats, indent=2)


@tool
def get_unique_value_counts(df_json: str, column: str, limit: int = 20) -> str:
    """Get unique values and their counts for a column. Args: df_json (str): JSON DataFrame representation. column (str): Column name. limit (int): Max unique values to return. Returns: str: JSON value counts."""
    from src.utils.logger import get_logger

    logger = get_logger(__name__)

    try:
        data_dict = parse_df_json_safely(df_json)
        df = pd.DataFrame.from_dict(data_dict)
    except ValueError as e:
        logger.error(f"Failed to parse df_json as JSON in get_unique_value_counts: {e}")
        try:
            error_detail = json.loads(str(e))
            return json.dumps(error_detail, indent=2)
        except Exception:
            return json.dumps(
                {
                    "error": f"Invalid JSON in df_json parameter: {str(e)}",
                    "error_type": "json_parse_error",
                }
            )

    if column not in df.columns:
        return json.dumps({"error": f"Column '{column}' not found in DataFrame"})

    value_counts = df[column].value_counts()
    total_unique = len(value_counts)
    value_counts = value_counts.head(limit)

    result = {
        "column": column,
        "total_unique_values": total_unique,
        "showing_top": min(limit, total_unique),
        "value_counts": [
            {"value": str(val), "count": int(count), "percentage": round(count / len(df) * 100, 2)}
            for val, count in value_counts.items()
        ],
    }

    return json.dumps(result, indent=2)


@tool
def detect_ordinal_patterns(df_json: str, column: str) -> str:
    """Detect if a column contains ordinal patterns. Args: df_json (str): JSON DataFrame representation. column (str): Column name. Returns: str: JSON with detected patterns and suggested mapping."""
    from src.utils.logger import get_logger

    logger = get_logger(__name__)

    try:
        data_dict = parse_df_json_safely(df_json)
        df = pd.DataFrame.from_dict(data_dict)
    except ValueError as e:
        logger.error(f"Failed to parse df_json as JSON in detect_ordinal_patterns: {e}")
        try:
            error_detail = json.loads(str(e))
            return json.dumps(error_detail, indent=2)
        except Exception:
            return json.dumps(
                {
                    "error": f"Invalid JSON in df_json parameter: {str(e)}",
                    "error_type": "json_parse_error",
                }
            )

    if column not in df.columns:
        return json.dumps({"error": f"Column '{column}' not found in DataFrame"})

    unique_values = df[column].dropna().unique().tolist()
    unique_values = [str(v) for v in unique_values]

    result: Dict[str, Any] = {
        "column": column,
        "unique_values": unique_values,
        "patterns_detected": [],
        "is_ordinal": False,
        "suggested_mapping": {},
    }

    numeric_pattern = re.compile(r"(\d+)")
    values_with_numbers = []
    for val in unique_values:
        match = numeric_pattern.search(val)
        if match:
            values_with_numbers.append((val, int(match.group(1))))

    if len(values_with_numbers) > len(unique_values) * 0.5:
        result["patterns_detected"].append("numeric_in_string")
        result["is_ordinal"] = True
        sorted_values = sorted(values_with_numbers, key=lambda x: x[1])
        result["suggested_mapping"] = {v[0]: i for i, v in enumerate(sorted_values)}

    ordinal_keywords = {
        "intern": 0,
        "entry": 1,
        "junior": 2,
        "associate": 3,
        "mid": 4,
        "senior": 5,
        "staff": 6,
        "principal": 7,
        "lead": 8,
        "director": 9,
        "vp": 10,
        "executive": 11,
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
    roman_pattern = re.compile(r"\b(I{1,3}|IV|V|VI{0,3}|IX|X)\b", re.IGNORECASE)
    roman_values = {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "VII": 7,
        "VIII": 8,
        "IX": 9,
        "X": 10,
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
    """Infer the semantic data type of a column. Args: df_json (str): JSON DataFrame representation. column (str): Column name. Returns: str: JSON with inferred semantic type and reasoning."""
    from src.utils.logger import get_logger

    logger = get_logger(__name__)

    logger.debug(f"detect_column_dtype called with column: {column}")
    logger.debug(f"df_json type: {type(df_json)}, length: {len(df_json) if df_json else 0}")
    logger.debug(f"df_json preview: {df_json[:200] if df_json else 'None'}")

    try:
        data_dict = parse_df_json_safely(df_json)
        logger.debug(
            f"Successfully parsed df_json, keys: {list(data_dict.keys())[:10] if isinstance(data_dict, dict) else 'not a dict'}"
        )
    except ValueError as e:
        logger.error(f"Failed to parse df_json as JSON: {e}")
        logger.error(f"df_json content (first 500 chars): {df_json[:500] if df_json else 'None'}")
        try:
            error_detail = json.loads(str(e))
            return json.dumps(error_detail, indent=2)
        except Exception:
            return json.dumps(
                {
                    "error": f"Invalid JSON in df_json parameter: {str(e)}",
                    "error_type": "json_parse_error",
                }
            )

    try:
        df = pd.DataFrame.from_dict(data_dict)
        logger.debug(f"Created DataFrame with shape: {df.shape}, columns: {df.columns.tolist()}")
    except Exception as e:
        logger.error(f"Failed to create DataFrame from dict: {e}", exc_info=True)
        return json.dumps({"error": f"Failed to create DataFrame: {str(e)}"})

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
        "reasoning": [],
    }

    if pd.api.types.is_datetime64_any_dtype(col_data):
        result["semantic_type"] = "datetime"
        result["reasoning"].append("Column has datetime dtype")
        return json.dumps(result, indent=2)

    if pd.api.types.is_bool_dtype(col_data) or set(col_data.dropna().unique()).issubset(
        {True, False, 0, 1, "True", "False", "true", "false"}
    ):
        result["semantic_type"] = "boolean"
        result["reasoning"].append("Column contains only boolean-like values")
        return json.dumps(result, indent=2)

    if pd.api.types.is_numeric_dtype(col_data):
        col_name_lower = column.lower()
        if unique_ratio > 0.9 and (
            "id" in col_name_lower or "key" in col_name_lower or "index" in col_name_lower
        ):
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
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, message=".*Could not infer format.*"
                )
                parsed = pd.to_datetime(col_data.dropna().head(10), errors="coerce")
            if parsed.notna().sum() > 5:
                result["semantic_type"] = "datetime"
                result["reasoning"].append("String values parseable as dates")
                return json.dumps(result, indent=2)
        except Exception:
            pass

        # Check if column might be location/geographic data
        col_name_lower = column.lower()
        location_keywords = [
            "location",
            "city",
            "address",
            "region",
            "state",
            "country",
            "place",
            "area",
            "zone",
        ]
        if any(keyword in col_name_lower for keyword in location_keywords):
            result["semantic_type"] = "location"
            result["reasoning"].append(f"Column name suggests geographic/location data: {column}")
            return json.dumps(result, indent=2)

        # Check uniqueness for categorical vs text
        if unique_ratio > 0.8:
            if "id" in col_name_lower or "key" in col_name_lower or "name" in col_name_lower:
                result["semantic_type"] = "identifier"
                result["reasoning"].append("High uniqueness and name suggests identifier")
            else:
                result["semantic_type"] = "text"
                result["reasoning"].append("High cardinality string, likely free text")
        else:
            result["semantic_type"] = "categorical"
            result["reasoning"].append(
                f"String with {unique_count} unique values ({unique_ratio:.1%} unique)"
            )

    if result["semantic_type"] is None:
        result["semantic_type"] = "unknown"
        result["reasoning"].append("Could not determine semantic type")

    return json.dumps(result, indent=2)


def get_all_tools() -> List["BaseTool"]:
    """Return list of all analysis tools for use with agents. Returns: List[BaseTool]: List of tool functions."""
    return [
        compute_correlation_matrix,
        get_column_statistics,
        get_unique_value_counts,
        detect_ordinal_patterns,
        detect_column_dtype,
    ]
