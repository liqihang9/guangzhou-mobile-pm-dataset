#!/usr/bin/env python3
"""Evaluate diurnal and weekday-weekend observation coverage.

The script reads the three published Guangzhou 15-second mobile files,
summarizes hourly records, valid-PM coverage, active devices, day-night shares,
and weekday-weekend coverage, and generates publication-ready outputs.
Published input files are read only and are never modified.

Dependencies: Python >=3.10, DuckDB, pandas, NumPy, and Matplotlib.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import gzip
from pathlib import Path
from time import perf_counter

import duckdb
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths and analysis settings
# ============================================================

# Edit these paths when running the script directly in an IDE. Command-line
# arguments are optional and, when supplied, override the values below.
DATA_DIR = Path(".")
OUTPUT_DIR = Path("results") / "diurnal_weekday_weekend_coverage"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
ANALYSIS_DATA_DIR = OUTPUT_DIR / "analysis_data"

TEMP_DIR = Path("duckdb_tmp_diurnal_weekday_weekend_coverage")

MONTH_FILES = {
    month: DATA_DIR / f"Guangzhou_mobile_{month}_15s_QC.csv.gz"
    for month in ("2023-03", "2023-08", "2023-11")
}

MONTH_LABELS = {
    "2023-03": "March 2023",
    "2023-08": "August 2023",
    "2023-11": "November 2023",
}

MONTH_COLORS = {
    "2023-03": "#356FB6",
    "2023-08": "#D47A25",
    "2023-11": "#2E8B6C",
}

FONT_DIR = Path("fonts")
HOUR_TICKS = [0, 4, 8, 12, 16, 20, 23]
NIGHT_COLOR = "#EEF1F4"
GRID_COLOR = "#D9DEE3"
VALID_PM_COLOR = "#263238"
LEGEND_COLOR = "#455A64"
DASH_STYLE = (0, (3.0, 2.0))

# China Standard Time definitions used in this analysis.
DAY_START_HOUR = 6
DAY_END_HOUR = 17

# A valid PM record must pass both particle-size-channel checks.
VALID_PM_DEFINITION = "PM25_QC=0 and PM10_QC=0"

# False prevents accidental replacement of completed validation products.
OVERWRITE_EXISTING = False

DUCKDB_THREADS = 4
DUCKDB_MEMORY_LIMIT = "8GB"
DUCKDB_MAX_TEMP_SIZE = "500GB"

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

OUTPUT_FILES = [
    FIGURE_DIR / "Figure_diurnal_weekday_weekend_coverage.png",
    FIGURE_DIR / "Figure_diurnal_weekday_weekend_coverage.pdf",
    TABLE_DIR / "Table_TV4_day_night_summary.csv",
    ANALYSIS_DATA_DIR / "hourly_coverage_metrics.csv",
    ANALYSIS_DATA_DIR / "date_hour_coverage_metrics.csv",
    ANALYSIS_DATA_DIR / "weekday_weekend_hourly_metrics.csv",
]


# ============================================================
# 2. Input checks and DuckDB extraction
# ============================================================

def sql_path(path: Path) -> str:
    """Escape a filesystem path for use in a DuckDB SQL string."""
    return str(path).replace("'", "''")


def read_csv_header(path: Path) -> list[str]:
    """Read only the header of a gzip-compressed CSV file."""
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as file:
        return next(csv.reader(file))


def validate_inputs_and_outputs() -> None:
    """Check release schemas and protect completed output files."""
    for month, path in MONTH_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing release file for {month}: {path}")
        columns = read_csv_header(path)
        if columns != EXPECTED_COLUMNS:
            raise RuntimeError(
                f"Unexpected columns in {path.name}.\n"
                f"Expected: {EXPECTED_COLUMNS}\nObserved: {columns}"
            )

    existing = [path for path in OUTPUT_FILES if path.exists()]
    if existing and not OVERWRITE_EXISTING:
        listed = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Validation outputs already exist. Set OVERWRITE_EXISTING=True "
            f"to replace them:\n{listed}"
        )


def csv_source(path: Path) -> str:
    """Return an explicit DuckDB reader for one release CSV.GZ file."""
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


def create_device_date_hour_table(
    con: duckdb.DuckDBPyConnection,
    month: str,
    path: Path,
) -> str:
    """Scan one release file once and cache device-date-hour counts."""
    table_name = f"device_date_hour_{month.replace('-', '_')}"
    source = csv_source(path)
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(
        f"""
        CREATE TEMP TABLE {table_name} AS
        SELECT
            CAST(TIME_POINT AS DATE) AS observation_date,
            EXTRACT(HOUR FROM TIME_POINT)::INTEGER AS hour,
            CASE
                WHEN EXTRACT(ISODOW FROM TIME_POINT) BETWEEN 1 AND 5
                THEN 'weekday'
                ELSE 'weekend'
            END AS day_type,
            CASE
                WHEN EXTRACT(HOUR FROM TIME_POINT)
                     BETWEEN {DAY_START_HOUR} AND {DAY_END_HOUR}
                THEN 'day'
                ELSE 'night'
            END AS day_night,
            DEVICE_ID,
            COUNT(*)::BIGINT AS record_count,
            COUNT(*) FILTER (
                WHERE PM25_QC = 0 AND PM10_QC = 0
            )::BIGINT AS valid_pm_record_count
        FROM {source}
        WHERE DEVICE_TIME_QC = 0
          AND DEVICE_ID IS NOT NULL
          AND TRIM(DEVICE_ID) <> ''
          AND TIME_POINT IS NOT NULL
        GROUP BY
            observation_date,
            hour,
            day_type,
            day_night,
            DEVICE_ID
        """
    )
    return table_name


def query_month_metrics(
    con: duckdb.DuckDBPyConnection,
    month: str,
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return hourly, date-hour and day/night metrics for one month."""
    table_name = create_device_date_hour_table(con, month, path)
    try:
        hourly = con.execute(
            f"""
            SELECT
                hour,
                SUM(record_count)::BIGINT AS record_count,
                SUM(valid_pm_record_count)::BIGINT
                    AS valid_pm_record_count,
                100.0 * SUM(valid_pm_record_count)
                    / NULLIF(SUM(record_count), 0) AS valid_pm_percent,
                COUNT(DISTINCT DEVICE_ID)::BIGINT AS active_device_count
            FROM {table_name}
            GROUP BY hour
            ORDER BY hour
            """
        ).df()

        date_hour = con.execute(
            f"""
            SELECT
                observation_date,
                hour,
                day_type,
                SUM(record_count)::BIGINT AS record_count,
                SUM(valid_pm_record_count)::BIGINT
                    AS valid_pm_record_count,
                COUNT(DISTINCT DEVICE_ID)::BIGINT AS active_device_count
            FROM {table_name}
            GROUP BY observation_date, hour, day_type
            ORDER BY observation_date, hour
            """
        ).df()

        day_night = con.execute(
            f"""
            SELECT
                day_night,
                SUM(record_count)::BIGINT AS record_count,
                SUM(valid_pm_record_count)::BIGINT
                    AS valid_pm_record_count,
                100.0 * SUM(valid_pm_record_count)
                    / NULLIF(SUM(record_count), 0) AS valid_pm_percent,
                COUNT(DISTINCT DEVICE_ID)::BIGINT AS active_device_count
            FROM {table_name}
            GROUP BY day_night
            ORDER BY CASE WHEN day_night = 'day' THEN 1 ELSE 2 END
            """
        ).df()
    finally:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")

    hourly.insert(0, "month", month)
    date_hour.insert(0, "month", month)
    day_night.insert(0, "month", month)
    return hourly, date_hour, day_night


# ============================================================
# 3. Complete calendars and summary products
# ============================================================

def complete_hourly_metrics(hourly: pd.DataFrame) -> pd.DataFrame:
    """Ensure that every month contains all 24 hourly bins."""
    parts: list[pd.DataFrame] = []
    for month in MONTH_FILES:
        subset = hourly.loc[hourly["month"] == month].copy()
        subset = subset.set_index("hour").reindex(range(24))
        count_columns = [
            "record_count",
            "valid_pm_record_count",
            "active_device_count",
        ]
        subset[count_columns] = subset[count_columns].fillna(0)
        subset["valid_pm_percent"] = np.where(
            subset["record_count"] > 0,
            100.0
            * subset["valid_pm_record_count"]
            / subset["record_count"],
            np.nan,
        )
        subset["month"] = month
        subset.index.name = "hour"
        parts.append(subset.reset_index())
    return pd.concat(parts, ignore_index=True)


def complete_date_hour_calendar(date_hour: pd.DataFrame) -> pd.DataFrame:
    """Add zero-count date-hour cells before weekday/weekend averaging."""
    parts: list[pd.DataFrame] = []
    for month in MONTH_FILES:
        start = pd.Timestamp(f"{month}-01")
        number_of_days = calendar.monthrange(start.year, start.month)[1]
        dates = pd.date_range(start, periods=number_of_days, freq="D")
        full_index = pd.MultiIndex.from_product(
            [dates, range(24)],
            names=["observation_date", "hour"],
        )

        subset = date_hour.loc[date_hour["month"] == month].copy()
        subset["observation_date"] = pd.to_datetime(
            subset["observation_date"]
        )
        subset = (
            subset.drop(columns=["month", "day_type"])
            .set_index(["observation_date", "hour"])
            .reindex(full_index)
            .reset_index()
        )
        count_columns = [
            "record_count",
            "valid_pm_record_count",
            "active_device_count",
        ]
        subset[count_columns] = subset[count_columns].fillna(0)
        subset["day_type"] = np.where(
            subset["observation_date"].dt.dayofweek < 5,
            "weekday",
            "weekend",
        )
        subset.insert(0, "month", month)
        parts.append(subset)

    return pd.concat(parts, ignore_index=True)


def build_weekday_weekend_hourly_metrics(
    completed_date_hour: pd.DataFrame,
) -> pd.DataFrame:
    """Compare weekday and weekend coverage after equalizing day counts."""
    result = (
        completed_date_hour.groupby(
            ["month", "day_type", "hour"],
            as_index=False,
        )
        .agg(
            calendar_days=("observation_date", "nunique"),
            total_record_count=("record_count", "sum"),
            total_valid_pm_record_count=(
                "valid_pm_record_count",
                "sum",
            ),
            mean_records_per_calendar_day=("record_count", "mean"),
            mean_valid_pm_records_per_calendar_day=(
                "valid_pm_record_count",
                "mean",
            ),
            mean_active_devices_per_calendar_day=(
                "active_device_count",
                "mean",
            ),
        )
        .sort_values(["month", "day_type", "hour"])
    )
    return result


def finalize_day_night_summary(day_night: pd.DataFrame) -> pd.DataFrame:
    """Add monthly record shares and explicit time definitions."""
    result = day_night.copy()
    month_totals = result.groupby("month")["record_count"].transform("sum")
    result["record_percent_of_month"] = (
        100.0 * result["record_count"] / month_totals
    )
    result["time_definition_cst"] = result["day_night"].map(
        {
            "day": "06:00-17:59",
            "night": "18:00-05:59",
        }
    )
    result["valid_pm_definition"] = VALID_PM_DEFINITION
    result["period_order"] = result["day_night"].map(
        {"day": 1, "night": 2}
    )
    result = result.sort_values(["month", "period_order"]).drop(
        columns="period_order"
    )
    return result[
        [
            "month",
            "day_night",
            "time_definition_cst",
            "record_count",
            "record_percent_of_month",
            "valid_pm_record_count",
            "valid_pm_percent",
            "active_device_count",
            "valid_pm_definition",
        ]
    ]


# ============================================================
# 4. Publication-style combined figure
# ============================================================

def set_figure_style() -> None:
    """Apply the final typography used for the combined figure."""
    if FONT_DIR.is_dir():
        for path in sorted(FONT_DIR.iterdir()):
            if path.suffix.lower() in {".ttf", ".otf"}:
                font_manager.fontManager.addfont(str(path))

    font_family = None
    for candidate in ("Arial", "Helvetica", "DejaVu Sans"):
        try:
            font_manager.findfont(
                font_manager.FontProperties(
                    family=candidate,
                    weight="normal",
                    style="normal",
                ),
                fallback_to_default=False,
            )
        except ValueError:
            continue
        font_family = candidate
        break

    if font_family is None:
        raise RuntimeError("No usable sans-serif font was found.")

    print(f"Figure font: {font_family}")
    if font_family != "Arial":
        print(
            "Arial was not found; using "
            f"{font_family}. Optional Arial font files may be placed in "
            f"{FONT_DIR}."
        )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_family],
            "font.size": 12,
            "font.weight": "normal",
            "font.style": "normal",
            "axes.labelsize": 12,
            "axes.labelweight": "normal",
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "text.usetex": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": font_family,
            "mathtext.it": f"{font_family}:italic",
            "mathtext.bf": f"{font_family}:bold",
            "mathtext.sf": font_family,
            "mathtext.default": "regular",
            "axes.linewidth": 0.8,
            "axes.unicode_minus": True,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.constrained_layout.use": False,
            "figure.autolayout": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_line_axis(
    axis: plt.Axes,
    panel_label: str,
    show_bottom: bool,
) -> None:
    """Style one line panel and add its panel label."""
    axis.axvspan(
        -0.5,
        5.5,
        color=NIGHT_COLOR,
        linewidth=0,
        zorder=0,
    )
    axis.axvspan(
        17.5,
        23.5,
        color=NIGHT_COLOR,
        linewidth=0,
        zorder=0,
    )
    axis.grid(
        axis="y",
        color=GRID_COLOR,
        linewidth=0.5,
        alpha=0.8,
    )
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.set_xlim(-0.5, 23.5)
    axis.set_xticks(HOUR_TICKS)
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=10.5,
        length=3.5,
        width=0.8,
        pad=3,
    )

    if not show_bottom:
        axis.spines["bottom"].set_visible(False)
        axis.tick_params(axis="x", bottom=False, labelbottom=False)

    axis.annotate(
        panel_label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 8),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        fontstyle="normal",
        annotation_clip=False,
    )


def plot_combined_coverage(
    hourly: pd.DataFrame,
    weekday_weekend: pd.DataFrame,
) -> None:
    """Create the final nine-panel hourly coverage figure."""
    months = list(MONTH_FILES)

    hourly = hourly.loc[hourly["month"].isin(months)].copy()
    weekday_weekend = weekday_weekend.loc[
        weekday_weekend["month"].isin(months)
    ].copy()

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(8.0, 7.2),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": [1.0, 0.88, 1.0]},
    )
    fig.subplots_adjust(
        left=0.13,
        right=0.985,
        bottom=0.095,
        top=0.93,
        wspace=0.18,
        hspace=0.27,
    )

    maximum_hourly_records = float(hourly["record_count"].max()) / 1e6
    minimum_active_devices = float(hourly["active_device_count"].min())
    maximum_active_devices = float(hourly["active_device_count"].max())
    maximum_daily_mean = (
        float(weekday_weekend["mean_records_per_calendar_day"].max())
        / 1e3
    )

    active_minimum = (
        max(
            0,
            5 * np.floor(minimum_active_devices / 5) - 5,
        )
        if minimum_active_devices > 0
        else 0
    )
    active_maximum = 5 * np.ceil(maximum_active_devices / 5) + 5

    for column, month in enumerate(months):
        color = MONTH_COLORS[month]
        hourly_month = (
            hourly.loc[hourly["month"] == month]
            .sort_values("hour")
            .copy()
        )
        comparison_month = weekday_weekend.loc[
            weekday_weekend["month"] == month
        ].copy()
        weekday = comparison_month.loc[
            comparison_month["day_type"] == "weekday"
        ].sort_values("hour")
        weekend = comparison_month.loc[
            comparison_month["day_type"] == "weekend"
        ].sort_values("hour")

        hours = hourly_month["hour"].to_numpy(dtype=int)

        record_axis = axes[0, column]
        record_axis.plot(
            hours,
            hourly_month["record_count"] / 1e6,
            color=color,
            linewidth=1.6,
            solid_capstyle="round",
        )
        record_axis.plot(
            hours,
            hourly_month["valid_pm_record_count"] / 1e6,
            color=VALID_PM_COLOR,
            linewidth=1.1,
            linestyle=DASH_STYLE,
        )
        record_axis.set_ylim(0, maximum_hourly_records * 1.08)
        record_axis.set_title(
            MONTH_LABELS[month],
            loc="center",
            pad=10,
            fontsize=12,
            fontweight="normal",
        )

        device_axis = axes[1, column]
        device_axis.plot(
            hours,
            hourly_month["active_device_count"],
            color=color,
            linewidth=1.5,
            solid_capstyle="round",
        )
        device_axis.scatter(
            hours,
            hourly_month["active_device_count"],
            color=color,
            s=6,
            linewidth=0,
            zorder=3,
        )
        device_axis.set_ylim(active_minimum, active_maximum)

        comparison_axis = axes[2, column]
        comparison_axis.plot(
            weekday["hour"],
            weekday["mean_records_per_calendar_day"] / 1e3,
            color=color,
            linewidth=1.6,
            solid_capstyle="round",
        )
        comparison_axis.plot(
            weekend["hour"],
            weekend["mean_records_per_calendar_day"] / 1e3,
            color=color,
            linewidth=1.4,
            linestyle=DASH_STYLE,
        )
        comparison_axis.set_ylim(0, maximum_daily_mean * 1.10)
        comparison_axis.set_xlabel("Hour of day (CST)", labelpad=6)

        for row in range(3):
            style_line_axis(
                axes[row, column],
                panel_label=chr(ord("a") + row * 3 + column),
                show_bottom=(row == 2),
            )

    axes[0, 0].set_ylabel(
        "Monthly records by hour\n" + r"($\times10^{6}$)",
        labelpad=6,
    )
    axes[1, 0].set_ylabel("Active devices", labelpad=6)
    axes[2, 0].set_ylabel(
        "Mean records per day\n" + r"($\times10^{3}$)",
        labelpad=6,
    )

    axes[1, 0].yaxis.set_major_locator(
        MaxNLocator(nbins=4, integer=True)
    )

    record_handles = [
        Line2D([0], [0], color=LEGEND_COLOR, linewidth=1.6),
        Line2D(
            [0],
            [0],
            color=VALID_PM_COLOR,
            linewidth=1.1,
            linestyle=DASH_STYLE,
        ),
    ]
    comparison_handles = [
        Line2D([0], [0], color=LEGEND_COLOR, linewidth=1.6),
        Line2D(
            [0],
            [0],
            color=LEGEND_COLOR,
            linewidth=1.4,
            linestyle=DASH_STYLE,
        ),
    ]

    legend_options = {
        "loc": "lower left",
        "frameon": False,
        "handlelength": 2.0,
        "handletextpad": 0.65,
        "borderpad": 0.25,
        "borderaxespad": 0.35,
        "labelspacing": 0.3,
        "prop": {
            "size": 10.5,
            "weight": "normal",
            "style": "normal",
        },
    }

    axes[0, 0].legend(
        record_handles,
        ["All time-valid records", "Valid PM records"],
        **legend_options,
    )
    axes[2, 0].legend(
        comparison_handles,
        ["Weekday", "Weekend"],
        **legend_options,
    )

    stem = FIGURE_DIR / "Figure_diurnal_weekday_weekend_coverage"
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(fig)


# ============================================================
# 5. Runtime configuration and main workflow
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse optional path and overwrite settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "Directory containing the three published mobile CSV.GZ files; "
            "defaults to DATA_DIR configured at the top of this script."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for figures, tables, and supporting analysis data; "
            "defaults to OUTPUT_DIR."
        ),
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="DuckDB temporary directory; defaults to TEMP_DIR.",
    )
    parser.add_argument(
        "--font-dir",
        type=Path,
        help="Optional directory containing local TTF or OTF font files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing result files to be replaced.",
    )
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    """Apply path overrides without tying execution to the script filename."""
    global DATA_DIR, OUTPUT_DIR, FIGURE_DIR, TABLE_DIR, ANALYSIS_DATA_DIR
    global TEMP_DIR, FONT_DIR, MONTH_FILES, OUTPUT_FILES, OVERWRITE_EXISTING

    configured_data_dir = args.data_dir if args.data_dir is not None else DATA_DIR
    DATA_DIR = configured_data_dir.expanduser().resolve()
    MONTH_FILES = {
        month: DATA_DIR / f"Guangzhou_mobile_{month}_15s_QC.csv.gz"
        for month in ("2023-03", "2023-08", "2023-11")
    }

    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir.expanduser().resolve()
    elif args.data_dir is not None:
        OUTPUT_DIR = DATA_DIR / "results" / "diurnal_weekday_weekend_coverage"
    else:
        OUTPUT_DIR = OUTPUT_DIR.expanduser().resolve()

    FIGURE_DIR = OUTPUT_DIR / "figures"
    TABLE_DIR = OUTPUT_DIR / "tables"
    ANALYSIS_DATA_DIR = OUTPUT_DIR / "analysis_data"

    configured_temp_dir = args.temp_dir if args.temp_dir is not None else TEMP_DIR
    TEMP_DIR = configured_temp_dir.expanduser().resolve()
    configured_font_dir = args.font_dir if args.font_dir is not None else FONT_DIR
    FONT_DIR = configured_font_dir.expanduser().resolve()
    OVERWRITE_EXISTING = bool(args.overwrite)

    OUTPUT_FILES = [
        FIGURE_DIR / "Figure_diurnal_weekday_weekend_coverage.png",
        FIGURE_DIR / "Figure_diurnal_weekday_weekend_coverage.pdf",
        TABLE_DIR / "Table_TV4_day_night_summary.csv",
        ANALYSIS_DATA_DIR / "hourly_coverage_metrics.csv",
        ANALYSIS_DATA_DIR / "date_hour_coverage_metrics.csv",
        ANALYSIS_DATA_DIR / "weekday_weekend_hourly_metrics.csv",
    ]


def configure_duckdb() -> duckdb.DuckDBPyConnection:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads = {DUCKDB_THREADS}")
    con.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory = '{sql_path(TEMP_DIR)}'")
    con.execute(f"SET max_temp_directory_size = '{DUCKDB_MAX_TEMP_SIZE}'")
    return con


def main() -> None:
    """Run the complete diurnal and weekday-weekend coverage analysis."""
    configure_runtime(parse_args())
    started = perf_counter()
    validate_inputs_and_outputs()

    for directory in [FIGURE_DIR, TABLE_DIR, ANALYSIS_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    con = configure_duckdb()
    try:
        hourly_parts = []
        date_hour_parts = []
        day_night_parts = []
        for month, path in MONTH_FILES.items():
            print(f"Processing {month}: {path.name}")
            hourly, date_hour, day_night = query_month_metrics(
                con,
                month,
                path,
            )
            hourly_parts.append(hourly)
            date_hour_parts.append(date_hour)
            day_night_parts.append(day_night)
    finally:
        con.close()

    hourly = complete_hourly_metrics(
        pd.concat(hourly_parts, ignore_index=True)
    )
    date_hour = complete_date_hour_calendar(
        pd.concat(date_hour_parts, ignore_index=True)
    )
    weekday_weekend = build_weekday_weekend_hourly_metrics(date_hour)
    day_night = finalize_day_night_summary(
        pd.concat(day_night_parts, ignore_index=True)
    )

    hourly["valid_pm_percent"] = hourly["valid_pm_percent"].round(4)
    weekday_weekend[
        [
            "mean_records_per_calendar_day",
            "mean_valid_pm_records_per_calendar_day",
            "mean_active_devices_per_calendar_day",
        ]
    ] = weekday_weekend[
        [
            "mean_records_per_calendar_day",
            "mean_valid_pm_records_per_calendar_day",
            "mean_active_devices_per_calendar_day",
        ]
    ].round(4)
    day_night[
        ["record_percent_of_month", "valid_pm_percent"]
    ] = day_night[
        ["record_percent_of_month", "valid_pm_percent"]
    ].round(4)

    hourly.to_csv(
        ANALYSIS_DATA_DIR / "hourly_coverage_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    date_hour.to_csv(
        ANALYSIS_DATA_DIR / "date_hour_coverage_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    weekday_weekend.to_csv(
        ANALYSIS_DATA_DIR / "weekday_weekend_hourly_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    day_night.to_csv(
        TABLE_DIR / "Table_TV4_day_night_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    set_figure_style()
    plot_combined_coverage(hourly, weekday_weekend)

    elapsed = perf_counter() - started
    print(f"Completed in {elapsed / 60:.2f} minutes")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
