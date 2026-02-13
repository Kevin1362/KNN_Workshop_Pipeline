from __future__ import annotations
import requests
import pandas as pd

def load_api_json(url: str, records_path: str | None = None):
    """Load JSON from an API endpoint and return a DataFrame.

    If `records_path` is provided, it should be a key containing a list of records in the JSON.
    """
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    if records_path:
        if records_path not in data:
            raise ValueError(f"records_path '{records_path}' not found in JSON keys: {list(data.keys())}")
        data = data[records_path]
    return pd.DataFrame(data)
