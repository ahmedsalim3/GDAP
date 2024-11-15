import sqlite3
import pandas as pd


def save_to_db(df, db_path, table_name):
    """Save a DataFrame to an SQLite database table."""
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"Data saved to table '{table_name}' in database '{db_path}' successfully.")


def save_df_to_csv(df, file_path):
    """Save a DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)

def load_from_db(db_path, table_name):
    """Load a table from an SQLite database and return it as a DataFrame."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    return df
