"""Evaluate consistency among mobile-monitoring devices.

The script performs the complete 1 min interdevice consistency analysis,
the 5 min sensitivity analysis, and creates the associated publication figures
and reproducibility tables. Runtime paths are supplied independently of the
script filename.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from io import BytesIO
from pathlib import Path
from time import perf_counter

try:
    import duckdb
except ImportError:  # Allow --help to work before optional dependencies are installed.
    duckdb = None
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths and analysis settings
# ============================================================

RELEASE_DIR = Path("data")
OUTPUT_DIR = Path("results") / "interdevice_consistency"
TEMP_DIR = Path("duckdb_tmp_interdevice_consistency")
FONT_DIR = Path("fonts")

FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
ANALYSIS_DATA_DIR = OUTPUT_DIR / "analysis_data"

MONTH_FILES = {
    "2023-03": RELEASE_DIR / "Guangzhou_mobile_2023-03_15s_QC.csv.gz",
    "2023-08": RELEASE_DIR / "Guangzhou_mobile_2023-08_15s_QC.csv.gz",
    "2023-11": RELEASE_DIR / "Guangzhou_mobile_2023-11_15s_QC.csv.gz",
}

MONTH_LABELS = {
    "2023-03": "March",
    "2023-08": "August",
    "2023-11": "November",
    "All months": "All months",
}

MONTH_COLORS = {
    "2023-03": "#356FB6",
    "2023-08": "#D47A25",
    "2023-11": "#2E8B6C",
}

POLLUTANTS = {
    "PM25": {
        "value_column": "pm25_median",
        "label": r"PM$_{2.5}$",
        "plain_label": "PM2.5",
        "color": "#3973B7",
    },
    "PM10": {
        "value_column": "pm10_median",
        "label": r"PM$_{10}$",
        "plain_label": "PM10",
        "color": "#D77B2B",
    },
}

# Primary quasi-synchronous unit and temporal sensitivity analysis.
ANALYSIS_CRS = "EPSG:4547"
SOURCE_CRS = "EPSG:4326"
GRID_SIZE_M = 500
PRIMARY_TIME_BIN_MINUTES = 1
SENSITIVITY_TIME_BIN_MINUTES = [5]

# All device-pair results are exported. Only pairs meeting this threshold are
# included in summary statistics and manuscript figures.
MIN_COMMON_UNITS = 50

# A representative pair is selected from the upper quartile of overlap counts.
# Within that subset, choose the pair whose MAD is closest to the network-wide
# median MAD. This avoids choosing the visually best-performing pair.
REPRESENTATIVE_OVERLAP_QUANTILE = 0.75

OVERWRITE_EXISTING = False
KEEP_TEMPORARY_MATCHED_VALUES = False

DUCKDB_THREADS = 4
DUCKDB_MEMORY_LIMIT = "8GB"
DUCKDB_MAX_TEMP_SIZE = "500GB"

RANDOM_SEED = 20230811
MAX_JITTER_POINTS_PER_GROUP = 120
MAX_REPRESENTATIVE_POINTS_PER_MONTH = 6000

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

MEDIANS_PARQUET = ANALYSIS_DATA_DIR / "device_spatiotemporal_medians_1min.parquet"
PAIRWISE_METRICS_CSV = TABLE_DIR / "pairwise_consistency_metrics.csv"
SUMMARY_CSV = TABLE_DIR / "monthly_consistency_summary.csv"
SENSITIVITY_SUMMARY_CSV = TABLE_DIR / "time_bin_sensitivity_summary.csv"
SELECTED_PAIRS_CSV = TABLE_DIR / "representative_pairs_selected.csv"
REPRESENTATIVE_DATA_DIR = ANALYSIS_DATA_DIR / "representative_device_pairs"

FIGURE_7_PNG = FIGURE_DIR / "Figure_interdevice_consistency.png"
FIGURE_7_PDF = FIGURE_DIR / "Figure_interdevice_consistency.pdf"
# Retained only for the non-default summary helper; the release workflow does
# not call that helper or list these files as outputs.
FIGURE_6_PNG = FIGURE_DIR / "Figure_interdevice_consistency_summary.png"
FIGURE_6_PDF = FIGURE_DIR / "Figure_interdevice_consistency_summary.pdf"

OUTPUT_FILES = [
    MEDIANS_PARQUET,
    PAIRWISE_METRICS_CSV,
    SUMMARY_CSV,
    SENSITIVITY_SUMMARY_CSV,
    SELECTED_PAIRS_CSV,
    FIGURE_7_PNG,
    FIGURE_7_PDF,
]


# ============================================================
# 2. Input checks and DuckDB configuration
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate interdevice consistency in the Guangzhou mobile "
            "monitoring release and generate the associated outputs."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RELEASE_DIR,
        help="Directory containing the three published monthly CSV.GZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Root directory for generated figures, tables and analysis data.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=TEMP_DIR,
        help="DuckDB working directory.",
    )
    parser.add_argument(
        "--font-dir",
        type=Path,
        default=FONT_DIR,
        help="Optional directory containing Arial or other sans-serif fonts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs.",
    )
    parser.add_argument(
        "--keep-temporary-matched-values",
        action="store_true",
        help="Retain intermediate matched-value files and the DuckDB database.",
    )
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    """Apply runtime paths without relying on the script filename."""
    global RELEASE_DIR, OUTPUT_DIR, TEMP_DIR, FONT_DIR, MONTH_FILES
    global FIGURE_DIR, TABLE_DIR, ANALYSIS_DATA_DIR
    global MEDIANS_PARQUET, PAIRWISE_METRICS_CSV, SUMMARY_CSV
    global SENSITIVITY_SUMMARY_CSV, SELECTED_PAIRS_CSV
    global REPRESENTATIVE_DATA_DIR
    global FIGURE_6_PNG, FIGURE_6_PDF, FIGURE_7_PNG, FIGURE_7_PDF
    global OUTPUT_FILES, OVERWRITE_EXISTING, KEEP_TEMPORARY_MATCHED_VALUES

    RELEASE_DIR = args.data_dir.expanduser().resolve()
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    TEMP_DIR = args.temp_dir.expanduser().resolve()
    FONT_DIR = args.font_dir.expanduser().resolve()
    FIGURE_DIR = OUTPUT_DIR / "figures"
    TABLE_DIR = OUTPUT_DIR / "tables"
    ANALYSIS_DATA_DIR = OUTPUT_DIR / "analysis_data"
    REPRESENTATIVE_DATA_DIR = ANALYSIS_DATA_DIR / "representative_device_pairs"

    MONTH_FILES = {
        "2023-03": RELEASE_DIR / "Guangzhou_mobile_2023-03_15s_QC.csv.gz",
        "2023-08": RELEASE_DIR / "Guangzhou_mobile_2023-08_15s_QC.csv.gz",
        "2023-11": RELEASE_DIR / "Guangzhou_mobile_2023-11_15s_QC.csv.gz",
    }
    MEDIANS_PARQUET = ANALYSIS_DATA_DIR / "device_spatiotemporal_medians_1min.parquet"
    PAIRWISE_METRICS_CSV = TABLE_DIR / "pairwise_consistency_metrics.csv"
    SUMMARY_CSV = TABLE_DIR / "monthly_consistency_summary.csv"
    SENSITIVITY_SUMMARY_CSV = TABLE_DIR / "time_bin_sensitivity_summary.csv"
    SELECTED_PAIRS_CSV = TABLE_DIR / "representative_pairs_selected.csv"
    FIGURE_7_PNG = FIGURE_DIR / "Figure_interdevice_consistency.png"
    FIGURE_7_PDF = FIGURE_DIR / "Figure_interdevice_consistency.pdf"
    FIGURE_6_PNG = FIGURE_DIR / "Figure_interdevice_consistency_summary.png"
    FIGURE_6_PDF = FIGURE_DIR / "Figure_interdevice_consistency_summary.pdf"
    OUTPUT_FILES = [
        MEDIANS_PARQUET, PAIRWISE_METRICS_CSV, SUMMARY_CSV,
        SENSITIVITY_SUMMARY_CSV, SELECTED_PAIRS_CSV,
        FIGURE_7_PNG, FIGURE_7_PDF,
    ]
    OVERWRITE_EXISTING = args.overwrite
    KEEP_TEMPORARY_MATCHED_VALUES = args.keep_temporary_matched_values


def sql_path(path: Path) -> str:
    """Escape a filesystem path for use inside a DuckDB SQL string."""
    return str(path).replace("'", "''")


def sql_text(value: str) -> str:
    """Escape a text literal for use inside a DuckDB SQL string."""
    return value.replace("'", "''")


def read_csv_header(path: Path) -> list[str]:
    """Read only the header of a gzip-compressed release CSV file."""
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as file:
        return next(csv.reader(file))


def prepare_directories() -> None:
    for path in [
        FIGURE_DIR,
        TABLE_DIR,
        ANALYSIS_DATA_DIR,
        REPRESENTATIVE_DATA_DIR,
        TEMP_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def validate_inputs_and_outputs() -> None:
    """Validate schemas and protect existing completed results."""
    for month, path in MONTH_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing release file for {month}: {path}")
        observed = read_csv_header(path)
        if observed != EXPECTED_COLUMNS:
            raise RuntimeError(
                f"Unexpected columns in {path.name}.\n"
                f"Expected: {EXPECTED_COLUMNS}\nObserved: {observed}"
            )

    existing = [path for path in OUTPUT_FILES if path.exists()]
    if existing and not OVERWRITE_EXISTING:
        listed = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Interdevice consistency outputs already exist. Add --overwrite "
            f"to replace them:\n{listed}"
        )

    if OVERWRITE_EXISTING:
        for path in OUTPUT_FILES:
            if path.exists():
                path.unlink()


def csv_source(path: Path) -> str:
    """Return an explicit DuckDB reader for a release CSV.GZ file."""
    return f"""
        read_csv(
            '{sql_path(path)}',
            header = true,
            compression = 'gzip',
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


def configure_duckdb() -> duckdb.DuckDBPyConnection:
    """Create and configure the analytical DuckDB connection."""
    if duckdb is None:
        raise ModuleNotFoundError(
            "DuckDB is required for this analysis. Install it with "
            "'pip install duckdb'."
        )
    database_path = TEMP_DIR / "interdevice_consistency.duckdb"
    if database_path.exists() and OVERWRITE_EXISTING:
        database_path.unlink()
    con = duckdb.connect(str(database_path))
    con.execute(f"SET threads = {DUCKDB_THREADS}")
    con.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory = '{sql_path(TEMP_DIR)}'")
    con.execute(f"SET max_temp_directory_size = '{DUCKDB_MAX_TEMP_SIZE}'")
    try:
        con.execute("LOAD spatial")
    except duckdb.Error:
        con.execute("INSTALL spatial")
        con.execute("LOAD spatial")
    return con


# ============================================================
# 3. Device medians within fixed spatial and temporal units
# ============================================================

def monthly_medians_path(month: str, time_bin_minutes: int) -> Path:
    return TEMP_DIR / (
        f"device_unit_medians_{month}_{time_bin_minutes}min.parquet"
    )


def create_monthly_device_medians(
    con: duckdb.DuckDBPyConnection,
    month: str,
    release_path: Path,
    time_bin_minutes: int,
) -> Path:
    """Aggregate valid observations by month, device, grid and time bin."""
    output = monthly_medians_path(month, time_bin_minutes)
    if output.exists():
        output.unlink()

    source = csv_source(release_path)
    con.execute(
        f"""
        COPY (
            WITH projected AS (
                SELECT
                    DEVICE_ID,
                    TIME_POINT,
                    PM25,
                    PM10,
                    PM25_QC,
                    PM10_QC,
                    ST_Transform(
                        ST_Point(LONGITUDE, LATITUDE),
                        '{SOURCE_CRS}',
                        '{ANALYSIS_CRS}',
                        always_xy := true
                    ) AS point_geometry
                FROM {source}
                WHERE DEVICE_TIME_QC = 0
                  AND GPS_QC = 0
                  AND DEVICE_ID IS NOT NULL
                  AND TRIM(DEVICE_ID) <> ''
                  AND TIME_POINT IS NOT NULL
                  AND LONGITUDE IS NOT NULL
                  AND LATITUDE IS NOT NULL
                  AND (PM25_QC = 0 OR PM10_QC = 0)
            ), located AS (
                SELECT
                    DEVICE_ID,
                    TIME_POINT,
                    PM25,
                    PM10,
                    PM25_QC,
                    PM10_QC,
                    FLOOR(ST_X(point_geometry) / {GRID_SIZE_M})::BIGINT
                        AS grid_ix,
                    FLOOR(ST_Y(point_geometry) / {GRID_SIZE_M})::BIGINT
                        AS grid_iy,
                    time_bucket(
                        INTERVAL '{time_bin_minutes} minutes', TIME_POINT
                    ) AS time_bin
                FROM projected
            )
            SELECT
                '{sql_text(month)}'::VARCHAR AS month,
                grid_ix,
                grid_iy,
                time_bin,
                DEVICE_ID,
                median(CASE WHEN PM25_QC = 0 THEN PM25 END)
                    AS pm25_median,
                median(CASE WHEN PM10_QC = 0 THEN PM10 END)
                    AS pm10_median,
                COUNT(*) FILTER (WHERE PM25_QC = 0)::INTEGER
                    AS pm25_record_count,
                COUNT(*) FILTER (WHERE PM10_QC = 0)::INTEGER
                    AS pm10_record_count
            FROM located
            GROUP BY grid_ix, grid_iy, time_bin, DEVICE_ID
            HAVING COUNT(*) FILTER (WHERE PM25_QC = 0) > 0
                OR COUNT(*) FILTER (WHERE PM10_QC = 0) > 0
            ORDER BY grid_ix, grid_iy, time_bin, DEVICE_ID
        )
        TO '{sql_path(output)}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    return output


def combine_monthly_medians(
    con: duckdb.DuckDBPyConnection,
    paths: list[Path],
    output: Path,
) -> None:
    """Combine monthly medians into the published analysis-data Parquet."""
    path_list = ", ".join(f"'{sql_path(path)}'" for path in paths)
    con.execute(
        f"""
        COPY (
            SELECT *
            FROM read_parquet([{path_list}])
            ORDER BY month, grid_ix, grid_iy, time_bin, DEVICE_ID
        )
        TO '{sql_path(output)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
        """
    )


# ============================================================
# 4. Pairwise matching and consistency metrics
# ============================================================

def matched_values_path(pollutant: str, time_bin_minutes: int) -> Path:
    return TEMP_DIR / (
        f"matched_values_{pollutant.lower()}_{time_bin_minutes}min.parquet"
    )


def create_matched_values(
    con: duckdb.DuckDBPyConnection,
    pollutant: str,
    medians_path: Path,
    time_bin_minutes: int,
) -> Path:
    """Create one row per pollutant, shared unit and ordered device pair."""
    value_column = POLLUTANTS[pollutant]["value_column"]
    output = matched_values_path(pollutant, time_bin_minutes)
    if output.exists():
        output.unlink()

    con.execute(
        f"""
        COPY (
            SELECT
                a.month,
                a.grid_ix,
                a.grid_iy,
                a.time_bin,
                a.DEVICE_ID AS device_a,
                b.DEVICE_ID AS device_b,
                a.{value_column} AS value_a,
                b.{value_column} AS value_b,
                a.{value_column} - b.{value_column} AS difference,
                ABS(a.{value_column} - b.{value_column})
                    AS absolute_difference,
                (a.{value_column} + b.{value_column}) / 2.0
                    AS pair_mean
            FROM read_parquet('{sql_path(medians_path)}') AS a
            INNER JOIN read_parquet('{sql_path(medians_path)}') AS b
                ON a.month = b.month
               AND a.grid_ix = b.grid_ix
               AND a.grid_iy = b.grid_iy
               AND a.time_bin = b.time_bin
               AND a.DEVICE_ID < b.DEVICE_ID
            WHERE a.{value_column} IS NOT NULL
              AND b.{value_column} IS NOT NULL
        )
        TO '{sql_path(output)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
        """
    )
    return output


def calculate_pairwise_metrics(
    con: duckdb.DuckDBPyConnection,
    pollutant: str,
    matched_path: Path,
    time_bin_minutes: int,
) -> pd.DataFrame:
    """Calculate monthly and all-month metrics for every ordered pair."""
    result = con.execute(
        f"""
        SELECT
            CASE
                WHEN GROUPING(month) = 1 THEN 'All months'
                ELSE month
            END AS scope,
            {time_bin_minutes}::INTEGER AS time_bin_minutes,
            '{pollutant}'::VARCHAR AS pollutant,
            device_a,
            device_b,
            COUNT(*)::BIGINT AS common_unit_count,
            corr(value_a, value_b) AS pearson_r,
            AVG(difference) AS mean_difference,
            median(absolute_difference) AS median_absolute_difference,
            AVG(absolute_difference) AS mad,
            SQRT(AVG(difference * difference)) AS rmse,
            AVG(value_a) AS mean_value_a,
            AVG(value_b) AS mean_value_b
        FROM read_parquet('{sql_path(matched_path)}')
        GROUP BY GROUPING SETS (
            (month, device_a, device_b),
            (device_a, device_b)
        )
        ORDER BY scope, pollutant, device_a, device_b
        """
    ).df()
    result["meets_minimum_common_units"] = (
        result["common_unit_count"] >= MIN_COMMON_UNITS
    )
    result["absolute_mean_difference"] = result["mean_difference"].abs()
    return result


def build_consistency_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize qualifying pairs without treating devices as references."""
    scopes = [*MONTH_FILES.keys(), "All months"]
    rows: list[dict[str, object]] = []

    for scope in scopes:
        for pollutant in POLLUTANTS:
            available = metrics.loc[
                (metrics["scope"] == scope)
                & (metrics["pollutant"] == pollutant)
            ].copy()
            qualified = available.loc[
                available["meets_minimum_common_units"]
            ].copy()
            row: dict[str, object] = {
                "scope": scope,
                "time_bin_minutes": int(
                    available["time_bin_minutes"].iloc[0]
                ) if not available.empty else np.nan,
                "pollutant": pollutant,
                "available_pair_count": len(available),
                "qualified_pair_count": len(qualified),
                "minimum_common_units_required": MIN_COMMON_UNITS,
                "qualified_device_count": pd.concat(
                    [qualified["device_a"], qualified["device_b"]],
                    ignore_index=True,
                ).nunique(),
            }
            for column in [
                "common_unit_count",
                "pearson_r",
                "absolute_mean_difference",
                "median_absolute_difference",
                "mad",
                "rmse",
            ]:
                values = pd.to_numeric(qualified[column], errors="coerce").dropna()
                for suffix, quantile in [
                    ("q25", 0.25),
                    ("median", 0.50),
                    ("q75", 0.75),
                ]:
                    row[f"{column}_{suffix}"] = (
                        float(values.quantile(quantile))
                        if not values.empty
                        else np.nan
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def choose_representative_pairs(
    metrics: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Select one reproducible, high-overlap, typical-MAD pair per pollutant."""
    selected: dict[str, pd.Series] = {}
    for pollutant in POLLUTANTS:
        qualified = metrics.loc[
            (metrics["scope"] == "All months")
            & (metrics["pollutant"] == pollutant)
            & metrics["meets_minimum_common_units"]
        ].dropna(subset=["mad"]).copy()
        if qualified.empty:
            raise RuntimeError(
                f"No {pollutant} device pairs have at least "
                f"{MIN_COMMON_UNITS} common units."
            )

        overlap_cutoff = qualified["common_unit_count"].quantile(
            REPRESENTATIVE_OVERLAP_QUANTILE
        )
        high_overlap = qualified.loc[
            qualified["common_unit_count"] >= overlap_cutoff
        ].copy()
        target_mad = qualified["mad"].median()
        high_overlap["distance_from_target_mad"] = (
            high_overlap["mad"] - target_mad
        ).abs()
        selected[pollutant] = high_overlap.sort_values(
            ["distance_from_target_mad", "common_unit_count"],
            ascending=[True, False],
        ).iloc[0]
    return selected


def load_representative_values(
    con: duckdb.DuckDBPyConnection,
    pollutant: str,
    matched_path: Path,
    selected: pd.Series,
) -> pd.DataFrame:
    """Load shared unit medians for a selected ordered pair."""
    device_a = sql_text(str(selected["device_a"]))
    device_b = sql_text(str(selected["device_b"]))
    return con.execute(
        f"""
        SELECT
            month,
            grid_ix,
            grid_iy,
            time_bin,
            value_a,
            value_b,
            difference,
            absolute_difference,
            pair_mean
        FROM read_parquet('{sql_path(matched_path)}')
        WHERE device_a = '{device_a}'
          AND device_b = '{device_b}'
        ORDER BY month, time_bin, grid_ix, grid_iy
        """
    ).df()


def export_representative_results(
    selected_pairs: dict[str, pd.Series],
    representative_values: dict[str, pd.DataFrame],
) -> None:
    """Save selected-pair metrics and the exact units used in Figure 7."""
    selected_table = pd.DataFrame(
        [selected_pairs[pollutant] for pollutant in POLLUTANTS]
    )
    selected_table.to_csv(
        SELECTED_PAIRS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    for pollutant, values in representative_values.items():
        selected = selected_pairs[pollutant]
        output = values.copy()
        output.insert(0, "pollutant", pollutant)
        output.insert(1, "device_a", str(selected["device_a"]))
        output.insert(2, "device_b", str(selected["device_b"]))
        output_path = REPRESENTATIVE_DATA_DIR / (
            f"representative_pair_{pollutant}_"
            f"{selected['device_a']}_{selected['device_b']}.csv.gz"
        )
        output.to_csv(
            output_path,
            index=False,
            compression="gzip",
            encoding="utf-8",
            date_format="%Y-%m-%d %H:%M:%S",
        )


# ============================================================
# 5. Journal-style figures
# ============================================================

# Figure 6 retains the validated network-summary design.

def style_summary_axis(
    ax: plt.Axes,
    panel_label: str,
    title: str,
) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.9)
    ax.grid(color="#D9DDE2", linewidth=0.6, alpha=0.58, zorder=0)
    ax.set_title(
        title,
        loc="left",
        x=0.055,
        fontsize=10,
        fontweight="semibold",
        pad=10,
    )
    ax.text(
        0.0,
        1.035,
        panel_label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="center",
    )


def plot_overlap_support(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    base = metrics.loc[metrics["pollutant"] == "PM25"]
    thresholds = np.unique(
        np.r_[
            np.arange(1, 101, 2),
            np.arange(110, 501, 10),
            np.arange(550, 5001, 50),
        ]
    )
    pooled_color = "#30343B"
    for scope in [*MONTH_FILES.keys(), "All months"]:
        counts = base.loc[
            base["scope"] == scope, "common_unit_count"
        ].to_numpy()
        if counts.size == 0:
            continue
        retention = np.array(
            [(counts >= threshold).mean() * 100 for threshold in thresholds]
        )
        color = MONTH_COLORS.get(scope, pooled_color)
        ax.plot(
            thresholds,
            retention,
            color=color,
            linewidth=2.3 if scope == "All months" else 1.9,
            label=MONTH_LABELS[scope],
            zorder=3,
        )

    ax.axvline(
        MIN_COMMON_UNITS,
        color="#202124",
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    ax.text(
        MIN_COMMON_UNITS * 1.12,
        7,
        f"minimum $N={MIN_COMMON_UNITS}$",
        fontsize=7.8,
        color="#30343B",
        rotation=90,
        va="bottom",
    )
    positive = base.loc[base["common_unit_count"] > 0, "common_unit_count"]
    upper = max(100, int(positive.quantile(0.999)) if not positive.empty else 100)
    ax.set_xscale("log")
    ax.set_xlim(1, upper * 1.2)
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Minimum number of common spatiotemporal units")
    ax.set_ylabel("Retained device pairs (%)")
    ax.legend(
        frameon=False,
        fontsize=7.5,
        ncol=2,
        loc="lower left",
        handlelength=2.2,
    )
    style_summary_axis(
        ax,
        "a",
        "Overlap support for device-pair comparisons",
    )


def metric_quantiles(
    metrics: pd.DataFrame,
    metric: str,
) -> list[tuple[str, str, float, float, float, float, float, float]]:
    rows = []
    y = 8.0
    for pollutant in ["PM25", "PM10"]:
        for scope in [*MONTH_FILES.keys(), "All months"]:
            values = pd.to_numeric(
                metrics.loc[
                    (metrics["pollutant"] == pollutant)
                    & (metrics["scope"] == scope)
                    & metrics["meets_minimum_common_units"],
                    metric,
                ],
                errors="coerce",
            ).dropna()
            if values.empty:
                quantiles = [np.nan] * 5
            else:
                quantiles = values.quantile(
                    [0.05, 0.25, 0.50, 0.75, 0.95]
                ).to_list()
            rows.append((pollutant, scope, *quantiles, y))
            y -= 1.0
        y -= 0.8
    return rows


def plot_metric_intervals(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    metric: str,
    panel_label: str,
    title: str,
    xlabel: str,
) -> None:
    pollutant_colors = {"PM25": "#356EA6", "PM10": "#D97A28"}
    rows = metric_quantiles(metrics, metric)
    y_positions = [row[-1] for row in rows]
    y_labels = [
        "Mar", "Aug", "Nov", "Pooled",
        "Mar", "Aug", "Nov", "Pooled",
    ]

    for row in rows:
        pollutant, scope, q05, q25, median, q75, q95, y = row
        if not np.isfinite(median):
            continue
        if scope == "All months":
            ax.axhspan(y - 0.42, y + 0.42, color="#F1F0ED", zorder=0)
        color = pollutant_colors[pollutant]
        ax.plot([q05, q95], [y, y], color=color, alpha=0.38, lw=1.0, zorder=2)
        ax.plot(
            [q25, q75],
            [y, y],
            color=color,
            lw=4.4,
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter(
            median,
            y,
            s=27,
            marker="D" if scope == "All months" else "o",
            facecolor="white",
            edgecolor=color,
            linewidth=1.35,
            zorder=4,
        )

    ax.axhline(3.6, color="#C7CBD0", lw=0.75)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-0.2, 8.7)
    ax.set_xlabel(xlabel)
    ax.text(
        -0.18,
        0.79,
        r"PM$_{2.5}$",
        transform=ax.transAxes,
        rotation=90,
        color=pollutant_colors["PM25"],
        fontsize=8.5,
        fontweight="semibold",
        va="center",
    )
    ax.text(
        -0.18,
        0.25,
        r"PM$_{10}$",
        transform=ax.transAxes,
        rotation=90,
        color=pollutant_colors["PM10"],
        fontsize=8.5,
        fontweight="semibold",
        va="center",
    )
    style_summary_axis(ax, panel_label, title)


def plot_consistency_summary(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.5))
    fig.patch.set_facecolor("white")

    plot_overlap_support(axes[0, 0], metrics)
    plot_metric_intervals(
        axes[0, 1],
        metrics,
        "pearson_r",
        "b",
        "Association across qualifying device pairs",
        "Pearson correlation",
    )
    plot_metric_intervals(
        axes[1, 0],
        metrics,
        "absolute_mean_difference",
        "c",
        "Magnitude of pairwise mean difference",
        r"|MD| ($\mu$g m$^{-3}$)",
    )
    plot_metric_intervals(
        axes[1, 1],
        metrics,
        "mad",
        "d",
        "Absolute disagreement across common units",
        r"MAD ($\mu$g m$^{-3}$)",
    )

    interval_handles = [
        Line2D([0], [0], color="#67717B", lw=1.0, alpha=0.55,
               label="5th–95th percentile"),
        Line2D([0], [0], color="#67717B", lw=4.4,
               solid_capstyle="round", label="Interquartile range"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white",
               markeredgecolor="#30343B", label="Median"),
        Line2D([0], [0], marker="D", linestyle="None", markerfacecolor="white",
               markeredgecolor="#30343B", label="Pooled median"),
    ]
    fig.legend(
        handles=interval_handles,
        loc="upper center",
        bbox_to_anchor=(0.68, 0.995),
        ncol=4,
        frameon=False,
        fontsize=7.5,
        handlelength=2.2,
    )
    fig.text(
        0.5,
        0.012,
        f"Intervals summarize device pairs with ≥{MIN_COMMON_UNITS} common "
        f"{GRID_SIZE_M} m × {PRIMARY_TIME_BIN_MINUTES} min units; no device "
        "is treated as a reference.",
        ha="center",
        fontsize=7.5,
        color="#5B6066",
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.92,
        bottom=0.09,
        wspace=0.32,
        hspace=0.40,
    )
    fig.savefig(FIGURE_6_PDF, facecolor="white", bbox_inches="tight")
    fig.savefig(FIGURE_6_PNG, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# Figure 7 uses the final gray-density publication design.
FIGURE7_FIGSIZE = (7.4, 7.0)
FIGURE7_DPI = 600
FIGURE7_LABEL_SIZE = 11.5
FIGURE7_TICK_SIZE = 10.5
FIGURE7_NOTE_SIZE = 10
FIGURE7_FONT_FAMILY = "Arial"
FIGURE7_MONTHS = {
    "2023-03": ("March", "#0072B2"),
    "2023-08": ("August", "#D55E00"),
    "2023-11": ("November", "#943C91"),
}
FIGURE7_MONTH_MARKERS = {"2023-03": "o", "2023-08": "s", "2023-11": "^"}
FIGURE7_POLLUTANT_LABELS = {
    "PM25": r"$\mathrm{PM}_{2.5}$",
    "PM10": r"$\mathrm{PM}_{10}$",
}
FIGURE7_UNIT = r"$\mathrm{\mu\,g\,m^{-3}}$"
FIGURE7_CMAP = LinearSegmentedColormap.from_list(
    "device_pair_density_gray",
    ["#F3F3F3", "#D6D6D6", "#A5A5A5", "#545454"],
)
FIGURE7_BOX = {
    "boxstyle": "round,pad=0.22",
    "facecolor": "white",
    "edgecolor": "#D1D5D9",
    "linewidth": 0.5,
    "alpha": 0.94,
}


def set_figure7_style() -> None:
    """Load a publication font and apply the final Figure 7 style."""
    global FIGURE7_FONT_FAMILY
    if FONT_DIR.is_dir():
        for pattern in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
            for path in FONT_DIR.glob(pattern):
                font_manager.fontManager.addfont(str(path))

    for candidate in ("Arial", "Helvetica", "DejaVu Sans"):
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=candidate),
                fallback_to_default=False,
            )
        except ValueError:
            continue
        FIGURE7_FONT_FAMILY = candidate
        break
    else:
        raise RuntimeError("No usable sans-serif font was found.")

    print(f"Figure 7 font: {FIGURE7_FONT_FAMILY}")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FIGURE7_FONT_FAMILY],
        "font.weight": "normal",
        "font.style": "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "font.size": 11,
        "axes.labelsize": FIGURE7_LABEL_SIZE,
        "axes.titlesize": 11.5,
        "xtick.labelsize": FIGURE7_TICK_SIZE,
        "ytick.labelsize": FIGURE7_TICK_SIZE,
        "legend.fontsize": 11,
        "mathtext.fontset": "custom",
        "mathtext.rm": FIGURE7_FONT_FAMILY,
        "mathtext.it": FIGURE7_FONT_FAMILY,
        "mathtext.bf": f"{FIGURE7_FONT_FAMILY}:bold",
        "mathtext.cal": FIGURE7_FONT_FAMILY,
        "mathtext.sf": FIGURE7_FONT_FAMILY,
        "mathtext.tt": FIGURE7_FONT_FAMILY,
        "axes.linewidth": 0.85,
        "axes.unicode_minus": True,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def figure7_monthly_curves(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> None:
    """Draw eight-quantile monthly median curves with at least five units per bin."""
    for month, (_, color) in FIGURE7_MONTHS.items():
        subset = data.loc[
            data["month"] == month, [x_column, y_column]
        ].dropna().copy()
        if len(subset) < 16:
            continue
        edges = np.unique(
            subset[x_column].quantile(np.linspace(0, 1, 9)).to_numpy()
        )
        if len(edges) < 2:
            continue
        subset["bin"] = pd.cut(subset[x_column], edges, include_lowest=True)
        curve = subset.groupby("bin", observed=True).agg(
            x=(x_column, "median"),
            y=(y_column, "median"),
            n=(y_column, "size"),
        )
        curve = curve.loc[curve["n"] >= 5]
        line, = ax.plot(
            curve["x"], curve["y"], color=color, linewidth=2,
            marker=FIGURE7_MONTH_MARKERS[month], markersize=4.5,
            markerfacecolor="white", markeredgewidth=1.05, zorder=6,
        )
        line.set_path_effects([
            patheffects.Stroke(linewidth=3.3, foreground="white"),
            patheffects.Normal(),
        ])


def style_figure7_axis(ax: plt.Axes, panel_label: str, title: str) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.85, pad=3)
    ax.grid(color="#D9DDE2", linewidth=0.5, alpha=0.55)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.set_title(title, loc="left", pad=9, fontweight="normal")
    ax.text(
        -0.16, 1.045, panel_label, transform=ax.transAxes,
        fontsize=ax.xaxis.label.get_fontsize(), fontweight="bold",
        fontstyle="normal", va="bottom",
    )


def finalize_figure7_fonts(fig: plt.Figure) -> None:
    """Use regular text throughout Figure 7 except for panel labels."""
    for text in fig.findobj(match=Text):
        text.set_fontfamily(FIGURE7_FONT_FAMILY)
        text.set_fontweight("normal")
        text.set_fontstyle("normal")
        text.set_math_fontfamily("custom")
    for ax in fig.axes:
        for text in ax.texts:
            if text.get_text().strip() in {"a", "b", "c", "d"}:
                text.set_fontweight("bold")
                text.set_fontsize(ax.xaxis.label.get_fontsize())


def add_figure7_outlier_inset(
    ax: plt.Axes,
    data: pd.DataFrame,
    limits: tuple[float, float],
) -> None:
    outside = data.loc[~data["difference"].between(*limits)]
    if outside.empty:
        return
    inset = ax.inset_axes([0.65, 0.68, 0.33, 0.26], zorder=10)
    for month, (_, color) in FIGURE7_MONTHS.items():
        subset = outside.loc[outside["month"] == month]
        inset.scatter(
            subset["pair_mean"], subset["difference"], s=21,
            marker=FIGURE7_MONTH_MARKERS[month], color=color,
            edgecolor="white", linewidth=0.6,
        )
    for column, setter in (
        ("pair_mean", inset.set_xlim),
        ("difference", inset.set_ylim),
    ):
        lower, upper = outside[column].min(), outside[column].max()
        padding = max(1.0, upper - lower) * 0.15
        setter(lower - padding, upper + padding)
    inset.spines[["top", "right"]].set_visible(False)
    inset.tick_params(labelsize=8, length=2, pad=1)
    inset.xaxis.set_major_locator(MaxNLocator(2, min_n_ticks=2))
    inset.yaxis.set_major_locator(MaxNLocator(2, min_n_ticks=2))
    if outside["pair_mean"].nunique() == 1:
        inset.set_xticks([outside["pair_mean"].iloc[0]])
    if outside["difference"].nunique() == 1:
        inset.set_yticks([outside["difference"].iloc[0]])
    inset.grid(color="#E1E3E5", linewidth=0.4)
    inset.set_title(f"Outside (n = {len(outside)})", fontsize=8.5, pad=3)


def draw_figure7_row(
    axes: np.ndarray,
    data: pd.DataFrame,
    selected: pd.Series,
    pollutant: str,
    panel_labels: tuple[str, str],
) -> None:
    left, right = axes
    device_a = str(selected["device_a"])
    device_b = str(selected["device_b"])
    pretty = FIGURE7_POLLUTANT_LABELS[pollutant]

    maximum = max(data["value_a"].max(), data["value_b"].max())
    upper = max(10.0, float(np.ceil(maximum / 10) * 10))
    left.hexbin(
        data["value_a"], data["value_b"], gridsize=38,
        extent=(0, upper, 0, upper), mincnt=1, cmap=FIGURE7_CMAP,
        norm=LogNorm(), linewidths=0, zorder=1,
    )
    figure7_monthly_curves(left, data, "value_a", "value_b")
    left.plot([0, upper], [0, upper], color="#22272B", linewidth=1.1, zorder=3)
    left.set(
        xlim=(0, upper), ylim=(0, upper),
        xlabel=f"Device {device_a} ({FIGURE7_UNIT})",
        ylabel=f"Device {device_b} ({FIGURE7_UNIT})",
    )
    left.set_aspect("equal", adjustable="box")
    left.set_anchor("E")
    style_figure7_axis(
        left, panel_labels[0], f"{pretty}: devices {device_a} and {device_b}"
    )
    correlation = data["value_a"].corr(data["value_b"])
    mad = data["difference"].abs().mean()
    left.text(
        0.97, 0.045,
        f"N = {len(data):,}\nr = {correlation:.3f}\n"
        f"MAD = {mad:.2f} {FIGURE7_UNIT}",
        transform=left.transAxes, ha="right", va="bottom",
        fontsize=FIGURE7_NOTE_SIZE, linespacing=1.2,
        bbox=FIGURE7_BOX, zorder=7,
    )

    mean_difference = data["difference"].mean()
    sd_difference = data["difference"].std(ddof=1)
    lower_loa = mean_difference - 1.96 * sd_difference
    upper_loa = mean_difference + 1.96 * sd_difference
    q005, q995 = data["difference"].quantile([0.005, 0.995])
    lower = min(q005, lower_loa)
    upper_difference = max(q995, upper_loa)
    span = max(1.0, upper_difference - lower)
    limits = (
        np.floor((lower - 0.08 * span) / 5) * 5,
        np.ceil((upper_difference + 0.08 * span) / 5) * 5,
    )
    x_upper = max(
        10.0,
        float(np.ceil(data["pair_mean"].quantile(0.995) / 10) * 10),
    )
    main_data = data.loc[data["difference"].between(*limits)]
    right.hexbin(
        main_data["pair_mean"], main_data["difference"], gridsize=39,
        extent=(0, x_upper, *limits), mincnt=1, cmap=FIGURE7_CMAP,
        norm=LogNorm(), linewidths=0, zorder=1,
    )
    right.axhline(0, color="#22272B", linewidth=0.75, zorder=2)
    right.axhline(mean_difference, color="#22272B", linewidth=1.4, zorder=3)
    for bound in (lower_loa, upper_loa):
        right.axhline(
            bound, color="#697077", linewidth=0.95,
            linestyle=(0, (5, 3)), zorder=2,
        )
    figure7_monthly_curves(right, data, "pair_mean", "difference")
    right.set(
        xlim=(0, x_upper), ylim=limits,
        xlabel=f"Pair mean ({FIGURE7_UNIT})",
        ylabel=f"{device_a} − {device_b} ({FIGURE7_UNIT})",
    )
    style_figure7_axis(
        right, panel_labels[1], f"{pretty}: pairwise difference"
    )
    right.text(
        0.025, 0.97,
        f"MD = {mean_difference:.2f}\n"
        f"Limits: [{lower_loa:.2f}, {upper_loa:.2f}]",
        transform=right.transAxes, ha="left", va="top",
        fontsize=9, linespacing=1.2, bbox=FIGURE7_BOX, zorder=7,
    )
    add_figure7_outlier_inset(right, data, limits)


def plot_representative_pairs(
    representative_values: dict[str, pd.DataFrame],
    selected_pairs: dict[str, pd.Series],
) -> None:
    """Draw Figure 7 using the final gray-density publication design."""
    set_figure7_style()
    fig, axes = plt.subplots(2, 2, figsize=FIGURE7_FIGSIZE)
    fig.subplots_adjust(
        left=0.10, right=0.985, bottom=0.08, top=0.84,
        wspace=0.40, hspace=0.52,
    )
    draw_figure7_row(
        axes[0], representative_values["PM25"], selected_pairs["PM25"],
        "PM25", ("a", "b"),
    )
    draw_figure7_row(
        axes[1], representative_values["PM10"], selected_pairs["PM10"],
        "PM10", ("c", "d"),
    )

    month_handles = [
        Line2D(
            [0], [0], color=color, linewidth=2,
            marker=FIGURE7_MONTH_MARKERS[month], markersize=4.5,
            markerfacecolor="white", markeredgewidth=1.05, label=label,
        )
        for month, (label, color) in FIGURE7_MONTHS.items()
    ]
    fig.legend(
        handles=month_handles, loc="upper center",
        bbox_to_anchor=(0.52, 0.993), ncol=3, frameon=False,
        handlelength=2, columnspacing=1.8,
    )
    guide_handles = [
        Line2D(
            [0], [0], color="#22272B", linewidth=1.1,
            label="1:1 or zero-difference reference",
        ),
        Line2D(
            [0], [0], color="#697077", linewidth=0.95,
            linestyle=(0, (5, 3)), label="95% limits of agreement",
        ),
    ]
    fig.legend(
        handles=guide_handles, loc="upper center",
        bbox_to_anchor=(0.52, 0.952), ncol=2, frameon=False,
        fontsize=10, handlelength=2.2,
    )
    finalize_figure7_fonts(fig)

    try:
        for extension, path in (("png", FIGURE_7_PNG), ("pdf", FIGURE_7_PDF)):
            with BytesIO() as buffer:
                fig.savefig(
                    buffer, format=extension, dpi=FIGURE7_DPI,
                    bbox_inches="tight", pad_inches=0.04,
                )
                content = buffer.getvalue()
            if extension == "pdf" and not content.rstrip().endswith(b"%%EOF"):
                raise RuntimeError("Figure 7 PDF rendering was incomplete.")
            path.write_bytes(content)
            print(f"Saved: {path}")
    finally:
        plt.close(fig)


# ============================================================
# 6. Documentation and orchestration
# ============================================================

def cleanup_temporary_files(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()
    database_path = TEMP_DIR / "interdevice_consistency.duckdb"
    if database_path.exists():
        database_path.unlink()
    for suffix in [".wal", ".tmp"]:
        candidate = Path(f"{database_path}{suffix}")
        if candidate.exists():
            candidate.unlink()
    try:
        if TEMP_DIR.exists() and not any(TEMP_DIR.iterdir()):
            TEMP_DIR.rmdir()
    except OSError:
        pass


def main() -> None:
    args = parse_args()
    configure_runtime(args)
    started = perf_counter()
    prepare_directories()
    validate_inputs_and_outputs()
    con = configure_duckdb()

    temporary_paths: list[Path] = []
    primary_matched_paths: dict[str, Path] = {}
    try:
        print(
            f"Primary analysis: {GRID_SIZE_M} m grid x "
            f"{PRIMARY_TIME_BIN_MINUTES} min time bins"
        )
        primary_monthly_paths: list[Path] = []
        for month, release_path in MONTH_FILES.items():
            print(f"Aggregating {month} into {GRID_SIZE_M} m x "
                  f"{PRIMARY_TIME_BIN_MINUTES} min device medians...")
            monthly_path = create_monthly_device_medians(
                con,
                month,
                release_path,
                PRIMARY_TIME_BIN_MINUTES,
            )
            primary_monthly_paths.append(monthly_path)
            temporary_paths.append(monthly_path)

        print("Combining monthly device medians...")
        combine_monthly_medians(
            con,
            primary_monthly_paths,
            MEDIANS_PARQUET,
        )

        metric_frames: list[pd.DataFrame] = []
        for pollutant in POLLUTANTS:
            print(f"Matching shared units for {pollutant}...")
            matched_path = create_matched_values(
                con,
                pollutant,
                MEDIANS_PARQUET,
                PRIMARY_TIME_BIN_MINUTES,
            )
            primary_matched_paths[pollutant] = matched_path
            temporary_paths.append(matched_path)
            print(f"Calculating pairwise metrics for {pollutant}...")
            metric_frames.append(
                calculate_pairwise_metrics(
                    con,
                    pollutant,
                    matched_path,
                    PRIMARY_TIME_BIN_MINUTES,
                )
            )

        metrics = pd.concat(metric_frames, ignore_index=True)
        scope_order = {scope: i for i, scope in enumerate(
            [*MONTH_FILES.keys(), "All months"]
        )}
        pollutant_order = {name: i for i, name in enumerate(POLLUTANTS)}
        metrics["_scope_order"] = metrics["scope"].map(scope_order)
        metrics["_pollutant_order"] = metrics["pollutant"].map(
            pollutant_order
        )
        metrics = metrics.sort_values(
            ["_scope_order", "_pollutant_order", "device_a", "device_b"]
        ).drop(columns=["_scope_order", "_pollutant_order"])
        metrics.to_csv(PAIRWISE_METRICS_CSV, index=False, encoding="utf-8-sig")

        summary = build_consistency_summary(metrics)
        summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

        sensitivity_summaries = [summary]
        for sensitivity_minutes in SENSITIVITY_TIME_BIN_MINUTES:
            print(
                f"Sensitivity analysis: {GRID_SIZE_M} m grid x "
                f"{sensitivity_minutes} min time bins"
            )
            sensitivity_monthly_paths: list[Path] = []
            for month, release_path in MONTH_FILES.items():
                sensitivity_path = create_monthly_device_medians(
                    con,
                    month,
                    release_path,
                    sensitivity_minutes,
                )
                sensitivity_monthly_paths.append(sensitivity_path)
                temporary_paths.append(sensitivity_path)

            sensitivity_medians = TEMP_DIR / (
                f"device_spatiotemporal_medians_{sensitivity_minutes}min.parquet"
            )
            if sensitivity_medians.exists():
                sensitivity_medians.unlink()
            combine_monthly_medians(
                con,
                sensitivity_monthly_paths,
                sensitivity_medians,
            )
            temporary_paths.append(sensitivity_medians)

            sensitivity_metric_frames: list[pd.DataFrame] = []
            for pollutant in POLLUTANTS:
                sensitivity_matched = create_matched_values(
                    con,
                    pollutant,
                    sensitivity_medians,
                    sensitivity_minutes,
                )
                temporary_paths.append(sensitivity_matched)
                sensitivity_metric_frames.append(
                    calculate_pairwise_metrics(
                        con,
                        pollutant,
                        sensitivity_matched,
                        sensitivity_minutes,
                    )
                )
            sensitivity_metrics = pd.concat(
                sensitivity_metric_frames,
                ignore_index=True,
            )
            sensitivity_summaries.append(
                build_consistency_summary(sensitivity_metrics)
            )

        sensitivity_summary = pd.concat(
            sensitivity_summaries,
            ignore_index=True,
        ).sort_values(["time_bin_minutes", "scope", "pollutant"])
        sensitivity_summary.to_csv(
            SENSITIVITY_SUMMARY_CSV,
            index=False,
            encoding="utf-8-sig",
        )

        selected_pairs = choose_representative_pairs(metrics)
        representative_values = {
            pollutant: load_representative_values(
                con,
                pollutant,
                primary_matched_paths[pollutant],
                selected_pairs[pollutant],
            )
            for pollutant in POLLUTANTS
        }
        export_representative_results(
            selected_pairs,
            representative_values,
        )

        print("Drawing interdevice-consistency figure...")
        plot_representative_pairs(
            representative_values,
            selected_pairs,
        )
    finally:
        con.close()
        if not KEEP_TEMPORARY_MATCHED_VALUES:
            cleanup_temporary_files(temporary_paths)

    elapsed_minutes = (perf_counter() - started) / 60.0
    print(f"Completed in {elapsed_minutes:.1f} minutes.")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
