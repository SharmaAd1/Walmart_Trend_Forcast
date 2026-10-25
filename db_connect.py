import os
from pathlib import Path

import mysql.connector
import pandas as pd
from mysql.connector import Error


TABLE_NAME = "walmart_sales"
COLUMNS = (
    "Store",
    "Date",
    "Weekly_Sales",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
)
ENV_FILE = Path(__file__).resolve().with_name(".env")


class DatabaseConnectionError(RuntimeError):
    pass


def load_local_environment() -> None:
    if not ENV_FILE.is_file():
        return

    try:
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DatabaseConnectionError(f"Could not read the environment file: {error}") from error

    for line_number, line in enumerate(lines, start=1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        key, separator, value = entry.partition("=")
        if not separator or not key.strip():
            raise DatabaseConnectionError(
                f"Invalid entry in the environment file on line {line_number}."
            )
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def fetch_walmart_sales() -> pd.DataFrame:
    load_local_environment()
    password = os.getenv("MYSQL_PASSWORD")
    if not password:
        raise DatabaseConnectionError(
            f"MYSQL_PASSWORD is not set. Add it to {ENV_FILE} or set it in your environment."
        )

    connection = None
    cursor = None
    try:
        connection = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=password,
            database=os.getenv("MYSQL_DATABASE", "cusNew"),
        )
        cursor = connection.cursor(dictionary=True)
        columns = ", ".join(f"`{column}`" for column in COLUMNS)
        cursor.execute(f"SELECT {columns} FROM `{TABLE_NAME}`")
        rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=COLUMNS)
    except (Error, ValueError) as error:
        raise DatabaseConnectionError(
            f"Could not read the `{TABLE_NAME}` table from MySQL: {error}"
        ) from error
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()
