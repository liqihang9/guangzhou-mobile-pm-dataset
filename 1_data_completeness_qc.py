#!/usr/bin/env python3
"""Generate data-completeness and quality-control results for the Guangzhou dataset.

The script reads the three published 15-second mobile files and the three
published hourly fixed-station files. It verifies their schemas, summarizes
temporal coverage and variable-level QC results, and generates the tables and
supporting results used in the corresponding Technical Validation section.
Published data are read only and are never modified.

Dependencies: Python >=3.10, DuckDB, pandas, and NumPy.
"""

import argparse
import csv
import gzip
from pathlib import Path
from time import perf_counter

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths and runtime settings
# ============================================================

# Edit these paths when running the script directly in an IDE. Command-line
# arguments are optional and, when supplied, override the values below.
DATA_DIR = Path(".")
RESULT_DIR = Path("results") / "data_completeness_qc"
TABLE_DIR = RESULT_DIR / "tables"
ANALYSIS_DATA_DIR = RESULT_DIR / "analysis_data"

# DuckDB may use this directory if an aggregation cannot fit in memory.
TEMP_DIR = Path("duckdb_tmp_technical_validation_01")

MONTH_FILES = {
    month: DATA_DIR / f"Guangzhou_mobile_{month}_15s_QC.csv.gz"
    for month in ("2023-03", "2023-08", "2023-11")
}

STATION_MONTH_FILES = {
    month: DATA_DIR / f"Guangzhou_station_{month}_1h_QC.csv.gz"
    for month in ("2023-03", "2023-08", "2023-11")
}

# False prevents accidental replacement of completed results.
OVERWRITE_EXISTING = False

DUCKDB_THREADS = 4
DUCKDB_MEMORY_LIMIT = "8GB"
DUCKDB_MAX_TEMP_SIZE = "500GB"


# ============================================================
# 2. Released fields and QC definitions
# ============================================================

EXPECTED_COLUMNS = [
    "DEVICE_ID",
    "TIME_POINT",
    "PM25",
    "PM10",
    "LONGITUDE",
    "LATITUDE",
    "TEMPERATURE",
    "HUMIDITY",
    "DEVICE_TIME_QC",
    "PM25_QC",
    "PM10_QC",
    "GPS_QC",
    "TEMPERATURE_QC",
    "HUMIDITY_QC",
    "ROW_QC",
    "QC_FLAG",
]

STATION_EXPECTED_COLUMNS = [
    "RECORD_ID",
    "STATION_CODE",
    "TIME_POINT",
    "PM25",
    "PM10",
    "STATION_TIME_QC",
    "PM25_QC",
    "PM10_QC",
    "ROW_QC",
    "QC_FLAG",
]

STATION_NUMERIC_QC_FIELDS = [
    "STATION_TIME_QC",
    "PM25_QC",
    "PM10_QC",
    "ROW_QC",
]

# Variables displayed in the valid-record percentage figure.
# A record is valid for a variable only when its own QC field equals 0.
VARIABLE_QC_FIELDS = {
    "PM2.5": "PM25_QC",
    "PM10": "PM10_QC",
    "GPS": "GPS_QC",
    "Temperature": "TEMPERATURE_QC",
    "Relative humidity": "HUMIDITY_QC",
}

# All numeric QC fields included in the full QC-level statistics.
NUMERIC_QC_FIELDS = [
    "DEVICE_TIME_QC",
    "PM25_QC",
    "PM10_QC",
    "GPS_QC",
    "TEMPERATURE_QC",
    "HUMIDITY_QC",
    "ROW_QC",
]

# ============================================================
# 3. Input validation and DuckDB source
# ============================================================

def sql_path(path: Path) -> str:
    """Escape a filesystem path for use in a DuckDB SQL string."""
    return str(path).replace("'", "''")


def read_csv_header(path: Path) -> list[str]:
    """Read only the header row of a gzip-compressed CSV file."""
    with gzip.open(
        path,
        mode="rt",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return next(csv.reader(file))


def validate_input_files() -> None:
    """Confirm that all mobile and fixed-station release files are present."""
    for month, path in MONTH_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing release file for {month}: {path}"
            )

        observed_columns = read_csv_header(path)
        if observed_columns != EXPECTED_COLUMNS:
            raise RuntimeError(
                f"Unexpected fields in {path.name}.\n"
                f"Expected: {EXPECTED_COLUMNS}\n"
                f"Observed: {observed_columns}"
            )

    for month, path in STATION_MONTH_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing fixed-station release file for {month}: {path}"
            )

        observed_columns = read_csv_header(path)
        if observed_columns != STATION_EXPECTED_COLUMNS:
            raise RuntimeError(
                f"Unexpected fields in {path.name}.\n"
                f"Expected: {STATION_EXPECTED_COLUMNS}\n"
                f"Observed: {observed_columns}"
            )


def csv_source(path: Path) -> str:
    """Return a typed DuckDB expression for one released CSV.GZ file."""
    return f"""
        read_csv(
            '{sql_path(path)}',
            header = true,
            compression = 'gzip',
            nullstr = '',
            timestampformat = '%Y-%m-%d %H:%M:%S',
            columns = {{
                'DEVICE_ID': 'VARCHAR',
                'TIME_POINT': 'TIMESTAMP',
                'PM25': 'DOUBLE',
                'PM10': 'DOUBLE',
                'LONGITUDE': 'DOUBLE',
                'LATITUDE': 'DOUBLE',
                'TEMPERATURE': 'DOUBLE',
                'HUMIDITY': 'DOUBLE',
                'DEVICE_TIME_QC': 'UTINYINT',
                'PM25_QC': 'UTINYINT',
                'PM10_QC': 'UTINYINT',
                'GPS_QC': 'UTINYINT',
                'TEMPERATURE_QC': 'UTINYINT',
                'HUMIDITY_QC': 'UTINYINT',
                'ROW_QC': 'UTINYINT',
                'QC_FLAG': 'VARCHAR'
            }}
        )
    """


def station_csv_source(path: Path) -> str:
    """Return a typed DuckDB expression for one fixed-station CSV.GZ file."""
    return f"""
        read_csv(
            '{sql_path(path)}',
            header = true,
            compression = 'gzip',
            nullstr = '',
            timestampformat = '%Y-%m-%d %H:%M:%S',
            columns = {{
                'RECORD_ID': 'VARCHAR',
                'STATION_CODE': 'VARCHAR',
                'TIME_POINT': 'TIMESTAMP',
                'PM25': 'DOUBLE',
                'PM10': 'DOUBLE',
                'STATION_TIME_QC': 'UTINYINT',
                'PM25_QC': 'UTINYINT',
                'PM10_QC': 'UTINYINT',
                'ROW_QC': 'UTINYINT',
                'QC_FLAG': 'VARCHAR'
            }}
        )
    """


# ============================================================
# 4. Monthly aggregation
# ============================================================

def aggregate_month(
    con: duckdb.DuckDBPyConnection,
    month: str,
    path: Path,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """
    Aggregate coverage, variable validity and numeric QC levels in one scan.

    QC_FLAG reasons are counted in a second scan because each row may contain
    more than one semicolon-delimited reason.
    """
    source = csv_source(path)

    qc_count_expressions = []
    for qc_field in NUMERIC_QC_FIELDS:
        for level in (0, 1, 2):
            alias = f"{qc_field.lower()}_{level}_count"
            qc_count_expressions.append(
                f"COUNT(*) FILTER (WHERE {qc_field} = {level}) AS {alias}"
            )

    aggregate_sql = f"""
        SELECT
            COUNT(*) AS total_record_count,
            COUNT(DISTINCT DEVICE_ID) AS device_count,
            COUNT(DISTINCT CAST(TIME_POINT AS DATE))
                FILTER (WHERE TIME_POINT IS NOT NULL)
                AS observed_date_count,
            MIN(TIME_POINT) AS earliest_sampling_time,
            MAX(TIME_POINT) AS latest_sampling_time,
            {', '.join(qc_count_expressions)}
        FROM {source}
    """

    aggregate = con.execute(aggregate_sql).df().iloc[0]
    total_records = int(aggregate["total_record_count"])

    earliest = aggregate["earliest_sampling_time"]
    latest = aggregate["latest_sampling_time"]

    overview_row = {
        "month": month,
        "file_name": path.name,
        "total_record_count": total_records,
        "device_count": int(aggregate["device_count"]),
        "observed_date_count": int(aggregate["observed_date_count"]),
        "first_observation_date": (
            earliest.strftime("%Y-%m-%d")
            if pd.notna(earliest)
            else ""
        ),
        "last_observation_date": (
            latest.strftime("%Y-%m-%d")
            if pd.notna(latest)
            else ""
        ),
        "earliest_sampling_time": (
            earliest.strftime("%Y-%m-%d %H:%M:%S")
            if pd.notna(earliest)
            else ""
        ),
        "latest_sampling_time": (
            latest.strftime("%Y-%m-%d %H:%M:%S")
            if pd.notna(latest)
            else ""
        ),
    }

    validity_rows = []
    for variable, qc_field in VARIABLE_QC_FIELDS.items():
        valid_count = int(
            aggregate[f"{qc_field.lower()}_0_count"]
        )
        validity_rows.append(
            {
                "month": month,
                "variable": variable,
                "qc_field": qc_field,
                "total_existing_records": total_records,
                "valid_record_count": valid_count,
                "valid_record_percent": (
                    100.0 * valid_count / total_records
                    if total_records > 0
                    else np.nan
                ),
            }
        )

    full_qc_rows = []
    issue_rows = []

    for qc_field in NUMERIC_QC_FIELDS:
        for level in (0, 1, 2):
            count = int(
                aggregate[f"{qc_field.lower()}_{level}_count"]
            )
            percentage = (
                100.0 * count / total_records
                if total_records > 0
                else np.nan
            )

            full_qc_rows.append(
                {
                    "month": month,
                    "qc_field": qc_field,
                    "qc_level": level,
                    "record_count": count,
                    "record_percent": percentage,
                }
            )

            # The manuscript-facing QC issue table excludes QC=0 because
            # valid percentages are already displayed in Figure TV1.
            if level in (1, 2) and count > 0:
                issue_rows.append(
                    {
                        "month": month,
                        "category_type": "QC_LEVEL",
                        "qc_field": qc_field,
                        "category": (
                            "1_suspicious"
                            if level == 1
                            else "2_invalid_or_missing"
                        ),
                        "record_count": count,
                        "record_percent": percentage,
                    }
                )

    flag_reason_sql = f"""
        WITH flag_reasons AS (
            SELECT TRIM(reason) AS reason
            FROM {source},
            UNNEST(
                STRING_SPLIT(COALESCE(QC_FLAG, ''), ';')
            ) AS split_flags(reason)
        )
        SELECT
            reason,
            COUNT(*) AS reason_count
        FROM flag_reasons
        WHERE reason <> ''
        GROUP BY reason
        ORDER BY reason_count DESC, reason
    """

    reason_result = con.execute(flag_reason_sql).df()
    for row in reason_result.itertuples(index=False):
        issue_rows.append(
            {
                "month": month,
                "category_type": "QC_FLAG_REASON",
                "qc_field": "QC_FLAG",
                "category": str(row.reason),
                "record_count": int(row.reason_count),
                "record_percent": (
                    100.0 * int(row.reason_count) / total_records
                    if total_records > 0
                    else np.nan
                ),
            }
        )

    return overview_row, validity_rows, full_qc_rows, issue_rows


def aggregate_station_scope(
    con: duckdb.DuckDBPyConnection,
    scope: str,
    source: str,
    file_name: str,
) -> tuple[dict, list[dict], list[dict]]:
    """Check and summarize one fixed-station scope from released QC files."""
    qc_count_expressions = []
    for qc_field in STATION_NUMERIC_QC_FIELDS:
        for level in (0, 1, 2):
            alias = f"{qc_field.lower()}_{level}_count"
            qc_count_expressions.append(
                f"COUNT(*) FILTER (WHERE {qc_field} = {level}) AS {alias}"
            )

    aggregate_sql = f"""
        SELECT
            COUNT(*) AS total_record_count,
            COUNT(DISTINCT STATION_CODE) AS station_count,
            COUNT(DISTINCT CAST(TIME_POINT AS DATE))
                FILTER (WHERE TIME_POINT IS NOT NULL)
                AS observed_date_count,
            MIN(TIME_POINT) AS earliest_sampling_time,
            MAX(TIME_POINT) AS latest_sampling_time,
            COUNT(*) - COUNT(DISTINCT RECORD_ID) AS duplicate_record_id_count,
            COUNT(*) - COUNT(DISTINCT (STATION_CODE, TIME_POINT))
                FILTER (WHERE STATION_CODE IS NOT NULL AND TIME_POINT IS NOT NULL)
                AS duplicate_station_hour_count,
            COUNT(*) FILTER (
                WHERE ROW_QC <> GREATEST(
                    STATION_TIME_QC, PM25_QC, PM10_QC
                )
            ) AS row_qc_mismatch_count,
            COUNT(*) FILTER (
                WHERE STATION_TIME_QC NOT IN (0, 2)
                   OR PM25_QC NOT IN (0, 1, 2)
                   OR PM10_QC NOT IN (0, 1, 2)
                   OR ROW_QC NOT IN (0, 1, 2)
                   OR STATION_TIME_QC IS NULL
                   OR PM25_QC IS NULL
                   OR PM10_QC IS NULL
                   OR ROW_QC IS NULL
            ) AS unexpected_qc_value_count,
            COUNT(*) FILTER (
                WHERE PM25_QC = 2 AND PM25 IS NOT NULL
            ) AS pm25_qc2_nonblank_count,
            COUNT(*) FILTER (
                WHERE PM10_QC = 2 AND PM10 IS NOT NULL
            ) AS pm10_qc2_nonblank_count,
            {', '.join(qc_count_expressions)}
        FROM {source}
    """

    aggregate = con.execute(aggregate_sql).df().iloc[0]
    total_records = int(aggregate["total_record_count"])
    earliest = aggregate["earliest_sampling_time"]
    latest = aggregate["latest_sampling_time"]

    overview_row = {
        "scope": scope,
        "file_name": file_name,
        "total_record_count": total_records,
        "station_count": int(aggregate["station_count"]),
        "observed_date_count": int(aggregate["observed_date_count"]),
        "earliest_sampling_time": (
            earliest.strftime("%Y-%m-%d %H:%M:%S")
            if pd.notna(earliest)
            else ""
        ),
        "latest_sampling_time": (
            latest.strftime("%Y-%m-%d %H:%M:%S")
            if pd.notna(latest)
            else ""
        ),
        "PM25_QC0_count": int(aggregate["pm25_qc_0_count"]),
        "PM25_QC0_percent": (
            100.0 * int(aggregate["pm25_qc_0_count"]) / total_records
            if total_records > 0
            else np.nan
        ),
        "PM10_QC0_count": int(aggregate["pm10_qc_0_count"]),
        "PM10_QC0_percent": (
            100.0 * int(aggregate["pm10_qc_0_count"]) / total_records
            if total_records > 0
            else np.nan
        ),
        "STATION_TIME_QC0_count": int(
            aggregate["station_time_qc_0_count"]
        ),
        "STATION_TIME_QC0_percent": (
            100.0
            * int(aggregate["station_time_qc_0_count"])
            / total_records
            if total_records > 0
            else np.nan
        ),
        "duplicate_record_id_count": int(
            aggregate["duplicate_record_id_count"]
        ),
        "duplicate_station_hour_count": int(
            aggregate["duplicate_station_hour_count"]
        ),
        "row_qc_mismatch_count": int(
            aggregate["row_qc_mismatch_count"]
        ),
        "unexpected_qc_value_count": int(
            aggregate["unexpected_qc_value_count"]
        ),
        "pm25_qc2_nonblank_count": int(
            aggregate["pm25_qc2_nonblank_count"]
        ),
        "pm10_qc2_nonblank_count": int(
            aggregate["pm10_qc2_nonblank_count"]
        ),
    }

    integrity_fields = [
        "duplicate_record_id_count",
        "duplicate_station_hour_count",
        "row_qc_mismatch_count",
        "unexpected_qc_value_count",
        "pm25_qc2_nonblank_count",
        "pm10_qc2_nonblank_count",
    ]
    failures = {
        field: overview_row[field]
        for field in integrity_fields
        if overview_row[field] != 0
    }
    if failures:
        raise RuntimeError(
            f"Fixed-station integrity check failed for {scope}: {failures}"
        )

    full_qc_rows = []
    issue_rows = []
    for qc_field in STATION_NUMERIC_QC_FIELDS:
        for level in (0, 1, 2):
            count = int(aggregate[f"{qc_field.lower()}_{level}_count"])
            percentage = (
                100.0 * count / total_records
                if total_records > 0
                else np.nan
            )
            full_qc_rows.append(
                {
                    "scope": scope,
                    "qc_field": qc_field,
                    "qc_level": level,
                    "record_count": count,
                    "record_percent": percentage,
                }
            )
            if level in (1, 2) and count > 0:
                issue_rows.append(
                    {
                        "scope": scope,
                        "category_type": "QC_LEVEL",
                        "qc_field": qc_field,
                        "category": (
                            "1_suspicious"
                            if level == 1
                            else "2_invalid_or_missing"
                        ),
                        "record_count": count,
                        "record_percent": percentage,
                    }
                )

    reason_sql = f"""
        WITH flag_reasons AS (
            SELECT TRIM(reason) AS reason
            FROM {source},
            UNNEST(
                STRING_SPLIT(COALESCE(QC_FLAG, ''), ';')
            ) AS split_flags(reason)
        )
        SELECT reason, COUNT(*) AS reason_count
        FROM flag_reasons
        WHERE reason <> ''
        GROUP BY reason
        ORDER BY reason_count DESC, reason
    """
    reason_result = con.execute(reason_sql).df()
    for row in reason_result.itertuples(index=False):
        issue_rows.append(
            {
                "scope": scope,
                "category_type": "QC_FLAG_REASON",
                "qc_field": "QC_FLAG",
                "category": str(row.reason),
                "record_count": int(row.reason_count),
                "record_percent": (
                    100.0 * int(row.reason_count) / total_records
                    if total_records > 0
                    else np.nan
                ),
            }
        )

    return overview_row, full_qc_rows, issue_rows


# ============================================================
# 5. Output tables and supporting analysis data
# ============================================================

def check_output_path(path: Path) -> None:
    """Prevent unintentional replacement of a completed result file."""
    if path.exists() and not OVERWRITE_EXISTING:
        raise FileExistsError(
            f"Result already exists: {path}. "
            "Set OVERWRITE_EXISTING=True to replace existing results."
        )


def validate_output_destinations() -> None:
    """Check all planned outputs before starting the large CSV scans."""
    planned_outputs = [
        TABLE_DIR / "Table_TV1_monthly_coverage.csv",
        TABLE_DIR / "Table_TV2_qc_issues_and_reasons.csv",
        TABLE_DIR / "Table_TV3_station_monthly_qc_summary.csv",
        TABLE_DIR / "Table_TV4_station_qc_issues_and_reasons.csv",
        ANALYSIS_DATA_DIR / "valid_record_percentages.csv",
        ANALYSIS_DATA_DIR / "full_qc_level_statistics.csv",
        ANALYSIS_DATA_DIR / "station_full_qc_level_statistics.csv",
    ]
    for path in planned_outputs:
        check_output_path(path)


def save_csv(data: pd.DataFrame, path: Path) -> None:
    check_output_path(path)
    data.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )


# ============================================================
# 6. Main workflow
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse optional path and overwrite settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "Directory containing the six published CSV.GZ files; defaults "
            "to DATA_DIR configured at the top of this script."
        ),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        help=(
            "Directory for generated tables, figure files, and supporting "
            "analysis data; defaults to RESULT_DIR."
        ),
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="DuckDB temporary directory; defaults to TEMP_DIR.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing result files to be replaced.",
    )
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    """Apply reader-facing path overrides without changing analysis logic."""
    global DATA_DIR, RESULT_DIR, TABLE_DIR, ANALYSIS_DATA_DIR
    global TEMP_DIR, MONTH_FILES, STATION_MONTH_FILES, OVERWRITE_EXISTING

    configured_data_dir = args.data_dir if args.data_dir is not None else DATA_DIR
    DATA_DIR = configured_data_dir.expanduser().resolve()

    months = ("2023-03", "2023-08", "2023-11")
    MONTH_FILES = {
        month: DATA_DIR / f"Guangzhou_mobile_{month}_15s_QC.csv.gz"
        for month in months
    }
    STATION_MONTH_FILES = {
        month: DATA_DIR / f"Guangzhou_station_{month}_1h_QC.csv.gz"
        for month in months
    }

    if args.result_dir is not None:
        RESULT_DIR = args.result_dir.expanduser().resolve()
    elif args.data_dir is not None:
        RESULT_DIR = DATA_DIR / "results" / "data_completeness_qc"
    else:
        RESULT_DIR = RESULT_DIR.expanduser().resolve()

    TABLE_DIR = RESULT_DIR / "tables"
    ANALYSIS_DATA_DIR = RESULT_DIR / "analysis_data"

    configured_temp_dir = args.temp_dir if args.temp_dir is not None else TEMP_DIR
    TEMP_DIR = configured_temp_dir.expanduser().resolve()
    OVERWRITE_EXISTING = bool(args.overwrite)


def main() -> None:
    """Run Technical Validation 01 on the published data files."""
    configure_runtime(parse_args())
    start = perf_counter()

    validate_input_files()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Fail before scanning the large input files if completed results already
    # exist and overwriting has not been explicitly enabled.
    validate_output_destinations()

    con = duckdb.connect()
    con.execute(f"SET threads={DUCKDB_THREADS}")
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory='{sql_path(TEMP_DIR)}'")
    con.execute(
        f"SET max_temp_directory_size='{DUCKDB_MAX_TEMP_SIZE}'"
    )

    overview_rows = []
    validity_rows = []
    full_qc_rows = []
    issue_rows = []
    station_overview_rows = []
    station_full_qc_rows = []
    station_issue_rows = []

    try:
        print(
            "========== Technical Validation 01 started ==========",
            flush=True,
        )

        for month, path in MONTH_FILES.items():
            month_start = perf_counter()
            print(f"\nProcessing {month}: {path.name}", flush=True)

            (
                overview_row,
                month_validity_rows,
                month_full_qc_rows,
                month_issue_rows,
            ) = aggregate_month(con, month, path)

            overview_rows.append(overview_row)
            validity_rows.extend(month_validity_rows)
            full_qc_rows.extend(month_full_qc_rows)
            issue_rows.extend(month_issue_rows)

            print(
                f"Completed {month} in "
                f"{(perf_counter() - month_start) / 60:.1f} min",
                flush=True,
            )

        for month, path in STATION_MONTH_FILES.items():
            print(
                f"\nChecking fixed-station {month}: {path.name}",
                flush=True,
            )
            (
                station_overview_row,
                month_station_full_qc_rows,
                month_station_issue_rows,
            ) = aggregate_station_scope(
                con,
                month,
                station_csv_source(path),
                path.name,
            )
            station_overview_rows.append(station_overview_row)
            station_full_qc_rows.extend(month_station_full_qc_rows)
            station_issue_rows.extend(month_station_issue_rows)

        combined_station_source = (
            "("
            + " UNION ALL ".join(
                f"SELECT * FROM {station_csv_source(path)}"
                for path in STATION_MONTH_FILES.values()
            )
            + ") AS station_all"
        )
        (
            combined_station_overview,
            combined_station_full_qc,
            combined_station_issues,
        ) = aggregate_station_scope(
            con,
            "three_months_combined",
            combined_station_source,
            ";".join(path.name for path in STATION_MONTH_FILES.values()),
        )
        station_overview_rows.append(combined_station_overview)
        station_full_qc_rows.extend(combined_station_full_qc)
        station_issue_rows.extend(combined_station_issues)

    finally:
        con.close()

    overview = pd.DataFrame(
        overview_rows,
        columns=[
            "month",
            "file_name",
            "total_record_count",
            "device_count",
            "observed_date_count",
            "first_observation_date",
            "last_observation_date",
            "earliest_sampling_time",
            "latest_sampling_time",
        ],
    )
    validity = pd.DataFrame(
        validity_rows,
        columns=[
            "month",
            "variable",
            "qc_field",
            "total_existing_records",
            "valid_record_count",
            "valid_record_percent",
        ],
    )
    full_qc = pd.DataFrame(
        full_qc_rows,
        columns=[
            "month",
            "qc_field",
            "qc_level",
            "record_count",
            "record_percent",
        ],
    )
    issues = pd.DataFrame(
        issue_rows,
        columns=[
            "month",
            "category_type",
            "qc_field",
            "category",
            "record_count",
            "record_percent",
        ],
    )
    station_overview = pd.DataFrame(
        station_overview_rows,
        columns=[
            "scope",
            "file_name",
            "total_record_count",
            "station_count",
            "observed_date_count",
            "earliest_sampling_time",
            "latest_sampling_time",
            "PM25_QC0_count",
            "PM25_QC0_percent",
            "PM10_QC0_count",
            "PM10_QC0_percent",
            "STATION_TIME_QC0_count",
            "STATION_TIME_QC0_percent",
            "duplicate_record_id_count",
            "duplicate_station_hour_count",
            "row_qc_mismatch_count",
            "unexpected_qc_value_count",
            "pm25_qc2_nonblank_count",
            "pm10_qc2_nonblank_count",
        ],
    )
    station_full_qc = pd.DataFrame(
        station_full_qc_rows,
        columns=[
            "scope",
            "qc_field",
            "qc_level",
            "record_count",
            "record_percent",
        ],
    )
    station_issues = pd.DataFrame(
        station_issue_rows,
        columns=[
            "scope",
            "category_type",
            "qc_field",
            "category",
            "record_count",
            "record_percent",
        ],
    )

    save_csv(
        overview,
        TABLE_DIR / "Table_TV1_monthly_coverage.csv",
    )
    save_csv(
        issues,
        TABLE_DIR / "Table_TV2_qc_issues_and_reasons.csv",
    )
    save_csv(
        validity,
        ANALYSIS_DATA_DIR / "valid_record_percentages.csv",
    )
    save_csv(
        full_qc,
        ANALYSIS_DATA_DIR / "full_qc_level_statistics.csv",
    )
    save_csv(
        station_overview,
        TABLE_DIR / "Table_TV3_station_monthly_qc_summary.csv",
    )
    save_csv(
        station_issues,
        TABLE_DIR / "Table_TV4_station_qc_issues_and_reasons.csv",
    )
    save_csv(
        station_full_qc,
        ANALYSIS_DATA_DIR / "station_full_qc_level_statistics.csv",
    )

    print(
        "\n========== Technical Validation 01 completed ==========",
        flush=True,
    )
    print(f"Results: {RESULT_DIR}", flush=True)
    print(
        f"Total elapsed time: {(perf_counter() - start) / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
