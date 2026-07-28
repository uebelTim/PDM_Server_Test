"""
db_export.py — SQL Server export + decode helpers (canonical historical pipeline).

This is the same export code used for historical backtesting data prep. The live
monitor reuses it verbatim so the live and historical paths pull and shape data
identically:

    export_table_to_csv(use_decode=True, decode_function="CPDecode")
        -> export_query_to_file(..., file_format="csv")   # CROSS APPLY CPDecode(Delta)
    format_channel_df(df)                                  # long (X,Y) -> wide channels

The decoded long rows carry (DateTime, CPScanIdx, X, Y, Z0, DeltaZ); only X and Y
matter downstream. After format_channel_df the frame is ~80-100 numerically-named
channel columns indexed by DateTime — exactly what process_all_channels expects.

Only change vs. the standalone script: the pyarrow import is lazy (inside the
parquet branch) so the CSV path works without pyarrow installed.
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import URL
import pyodbc
from tqdm import tqdm


# ----------------------------
# Connection builders
# ----------------------------
def build_connection_url(
    server,
    database,
    username=None,
    password=None,
    driver="ODBC Driver 18 for SQL Server",
    trusted=False,
    encrypt="yes",
    trust_cert="yes",
    port=None,
):
    """Build a SQLAlchemy URL for SQL Server via the mssql+pyodbc dialect."""
    host = server if port is None else f"{server},{port}"
    query = {"driver": driver, "Encrypt": encrypt, "TrustServerCertificate": trust_cert}
    if trusted:
        query["Trusted_Connection"] = "yes"
        return URL.create("mssql+pyodbc", host=host, database=database, query=query)
    return URL.create(
        "mssql+pyodbc",
        username=username,
        password=password,
        host=host,
        database=database,
        query=query,
    )


def build_conn_str(server, database, username=None, password=None,
                   driver="ODBC Driver 18 for SQL Server", trusted=False,
                   encrypt="yes", trust_cert="yes", port=None):
    """Build a DSN-less ODBC connection string for pyodbc."""
    host = server if port is None else f"{server},{port}"
    if trusted:
        return (f"DRIVER={{{driver}}};SERVER={host};DATABASE={database};"
                f"Trusted_Connection=Yes;Encrypt={encrypt};TrustServerCertificate={trust_cert}")
    return (f"DRIVER={{{driver}}};SERVER={host};DATABASE={database};UID={username};PWD={password};"
            f"Encrypt={encrypt};TrustServerCertificate={trust_cert}")


# ----------------------------
# SQL builder
# ----------------------------
def build_sql_for_export(table_or_join, use_decode, decode_function, decode_schema,
                         datetime_filter, blob_column, sort_column="X"):
    """
    Build the export SQL. With use_decode, joins the Blobs table to its Values
    partner on DateTime and CROSS APPLYs the decode function over the blob column.
    """
    params = {}
    blob_table = table_or_join
    values_table = table_or_join.replace("Blobs", "Values")

    func_prefix = f"{decode_schema}." if decode_schema else ""
    full_decode_func = f"{func_prefix}{decode_function}"

    if use_decode:
        sql = f"""
        SELECT
            v.DateTime,
            cp.* FROM (
            SELECT * FROM {values_table}
        ) AS v
        JOIN {blob_table} AS b
          ON v.DateTime = b.DateTime
        CROSS APPLY {full_decode_func}(b.{blob_column}) AS cp
        """
    else:
        sql = f"SELECT * FROM {table_or_join}"

    if datetime_filter:
        filter_col = "v.DateTime" if use_decode else "DateTime"
        sql += f" WHERE {filter_col} >= :dt_filter"
        params["dt_filter"] = datetime_filter

    if use_decode:
        sql += f"\nORDER BY v.DateTime, cp.{sort_column}"
    else:
        sql += "\nORDER BY DateTime"

    return sql, params


# ----------------------------
# Exporters
# ----------------------------
def export_query_to_file(server, database, sql, params, output_base_path,
                         file_format="parquet", username=None, password=None,
                         driver="ODBC Driver 18 for SQL Server", trusted=False,
                         encrypt="yes", trust_cert="yes", port=None,
                         chunksize=1_000_000, test_mode=False, sort_column="X"):
    """
    Execute a SQL query via SQLAlchemy and stream results directly to a Parquet or
    CSV file. Appends the correct extension and returns the final path.
    """
    file_format = file_format.lower()
    if file_format not in ["parquet", "csv"]:
        raise ValueError("file_format must be either 'parquet' or 'csv'")

    final_path = f"{output_base_path}.{file_format}"

    url = build_connection_url(server, database, username, password, driver,
                               trusted, encrypt, trust_cert, port)
    engine = create_engine(url, fast_executemany=True)

    chunk_msg = f"Chunks von {chunksize} Zeilen" if chunksize else "einem einzigen Durchlauf (kein Chunking)"
    print(f"Starte High-Speed Download ({file_format.upper()}) in {chunk_msg}...")
    if test_mode:
        print("ACHTUNG: Test-Modus aktiv! Der Download bricht nach dem ersten Batch ab.")

    pq_writer = None
    csv_file_handle = None
    first_chunk = True

    with engine.connect() as conn:
        try:
            if file_format == "csv":
                csv_file_handle = open(final_path, "w", encoding="utf-8", newline="")

            query_result = pd.read_sql_query(text(sql), conn, params=params, chunksize=chunksize)

            # chunksize=None -> pandas returns one DataFrame; wrap so the loop iterates rows.
            if chunksize is None:
                query_result = [query_result]

            with tqdm(unit=" Zeilen") as pbar:
                for chunk in query_result:
                    if sort_column in chunk.columns:
                        chunk[sort_column] = chunk[sort_column].fillna(-1).astype(int)

                    if file_format == "parquet":
                        import pyarrow as pa                # lazy: only needed for parquet
                        import pyarrow.parquet as pq
                        table = pa.Table.from_pandas(chunk)
                        if pq_writer is None:
                            pq_writer = pq.ParquetWriter(final_path, table.schema, compression="snappy")
                        pq_writer.write_table(table)
                    elif file_format == "csv":
                        chunk.to_csv(csv_file_handle, index=False, header=first_chunk)
                        first_chunk = False

                    pbar.update(len(chunk))
                    if test_mode:
                        print("\n[TEST-MODUS] Erster Batch gespeichert. Breche ab.")
                        break
        finally:
            if pq_writer:
                pq_writer.close()
            if csv_file_handle:
                csv_file_handle.close()

    print(f"\nDownload erfolgreich! Datei gespeichert unter: {final_path}")
    return final_path


def export_query_to_csv_pyodbc(server, database, sql, params, output_csv_path,
                               username=None, password=None,
                               driver="ODBC Driver 18 for SQL Server", trusted=False,
                               encrypt="yes", trust_cert="yes", port=None,
                               chunksize=None, excel_friendly=False):
    """Execute a SQL query via pyodbc and export results to CSV."""
    conn_str = build_conn_str(server, database, username, password, driver, trusted, encrypt, trust_cert, port)
    encoding = "utf-8-sig" if excel_friendly else "utf-8"

    if isinstance(params, dict) and ":dt" in sql or " :dt" in sql or "=:dt" in sql:
        sql_pyodbc = sql.replace(":dt", "?")
        params_pyodbc = [params.get("dt")]
    else:
        sql_pyodbc = sql
        params_pyodbc = list(params) if isinstance(params, (list, tuple)) else None

    with pyodbc.connect(conn_str) as conn:
        conn.setencoding(encoding="utf-8")
        conn.setdecoding(pyodbc.SQL_CHAR, encoding="utf-8")
        conn.setdecoding(pyodbc.SQL_WCHAR, encoding="utf-8")

        if chunksize:
            first = True
            with open(output_csv_path, "w", encoding=encoding, newline="") as f:
                for chunk in pd.read_sql_query(sql_pyodbc, conn, params=params_pyodbc, chunksize=chunksize):
                    chunk.to_csv(f, index=False, header=first)
                    first = False
        else:
            df = pd.read_sql_query(sql_pyodbc, conn, params=params_pyodbc)
            df.to_csv(output_csv_path, index=False, encoding=encoding)

    return output_csv_path


# ----------------------------
# Router with decode toggle
# ----------------------------
def export_table_to_csv(server, database, output_csv_path, username=None, password=None,
                        backend="sqlalchemy", driver="ODBC Driver 18 for SQL Server",
                        trusted=False, encrypt="yes", trust_cert="yes", port=None,
                        chunksize=None, excel_friendly=False,
                        use_decode=False, decode_function="CPDecode", decode_schema=None,
                        datetime_filter=None, table=None, blob_column="Delta",
                        test_mode=False, sort_column="X", file_format="csv"):
    """
    High-level router: build the (optionally decoding) SQL, then export via SQLAlchemy
    (streams to CSV/Parquet) or pyodbc. Returns the output path.
    """
    sql, params = build_sql_for_export(
        table_or_join=table or "",
        use_decode=use_decode,
        decode_function=decode_function,
        decode_schema=decode_schema,
        datetime_filter=datetime_filter,
        blob_column=blob_column,
        sort_column=sort_column,
    )

    if backend.lower() == "sqlalchemy":
        return export_query_to_file(
            server=server, database=database, sql=sql, params=params,
            output_base_path=output_csv_path, username=username, password=password,
            driver=driver, trusted=trusted, encrypt=encrypt, trust_cert=trust_cert,
            port=port, chunksize=chunksize, test_mode=test_mode, sort_column=sort_column,
            file_format=file_format,
        )
    elif backend.lower() == "pyodbc":
        return export_query_to_csv_pyodbc(
            server=server, database=database, sql=sql, params=params,
            output_csv_path=output_csv_path, username=username, password=password,
            driver=driver, trusted=trusted, encrypt=encrypt, trust_cert=trust_cert,
            port=port, chunksize=chunksize, excel_friendly=excel_friendly,
        )
    else:
        raise ValueError("backend must be 'sqlalchemy' or 'pyodbc'")


# ----------------------------
# Formatting: long -> wide channels
# ----------------------------
EXPECTED_COLS = ["DateTime", "CPScanIdx", "X", "Y", "Z0", "DeltaZ"]


def format_channel_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform ionization-chamber data from long to wide. Only X and Y matter:
    index=DateTime, columns=X (chamber), values=Y. Returns a DateTime-indexed frame
    whose columns are the numerically-named channels (~80-100 of them).
    """
    required = {"DateTime", "X", "Y"}
    missing = required - set(df.columns)
    if missing:
        print(f"Error: Missing required columns: {missing}")
        return None

    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")

    x_numeric = pd.to_numeric(df["X"], errors="coerce")
    if x_numeric.isna().any():
        df["X_clean"] = df["X"].astype(str)
    else:
        df["X_clean"] = x_numeric.astype("Int64")

    df_pivoted = df.pivot(index="DateTime", columns="X_clean", values="Y")

    try:
        df_pivoted = df_pivoted.sort_index().sort_index(axis=1)
    except Exception:
        pass

    df_pivoted = df_pivoted.rename_axis("Chamber", axis="columns")
    return df_pivoted


def format_long_data(file_path: str) -> pd.DataFrame:
    """
    Load a long-format CSV (locating the real header line) and pivot to wide channels.
    Kept for parity with the standalone export script.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return None

    header_line_number = None
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if "DateTime" in line:
                    cols = [c.strip() for c in line.split(",")]
                    if cols[: len(EXPECTED_COLS)] == EXPECTED_COLS:
                        header_line_number = i
                        break
        if header_line_number is None:
            print("Error: Could not find a valid header row matching expected columns.")
            return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    try:
        df = pd.read_csv(file_path, header=header_line_number, parse_dates=["DateTime"])
    except ValueError as e:
        if "Missing column provided to 'parse_dates'" in str(e):
            df = pd.read_csv(file_path, header=header_line_number)
            if "DateTime" not in df.columns:
                print("Error: 'DateTime' column not found even after using detected header.")
                return None
            df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
        else:
            raise

    return format_channel_df(df)
