from __future__ import annotations
import pandas as pd
from sqlalchemy import create_engine

def load_table(connection_uri: str, table_name: str):
    """Load a table from a relational DB (SQLite/Postgres/etc.) into a DataFrame."""
    engine = create_engine(connection_uri)
    with engine.connect() as conn:
        df = pd.read_sql_table(table_name, conn)
    return df
