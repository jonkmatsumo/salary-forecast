import pandas as pd
from typing import Tuple, Optional, Any

def validate_csv(file_buffer: Any) -> Tuple[bool, Optional[str], Optional[pd.DataFrame]]:
    """Validate a CSV file buffer. Args: file_buffer (Any): File-like object. Returns: Tuple[bool, Optional[str], Optional[pd.DataFrame]]: (is_valid, error_message, dataframe)."""
    try:
        file_buffer.seek(0)
        first_byte = file_buffer.read(1)
        if not first_byte:
            return False, "File is empty", None
        
        file_buffer.seek(0)
        df = pd.read_csv(file_buffer)
        
        if df.empty:
            return False, "CSV contains no data rows", None
            
        if len(df.columns) < 2:
            return False, "CSV must have at least 2 columns", None
            
        return True, None, df
        
    except Exception as e:
        return False, f"Failed to parse CSV: {str(e)}", None
    finally:
        file_buffer.seek(0)
