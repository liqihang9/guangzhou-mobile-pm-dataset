#!/usr/bin/env python3
"""Evaluate device operation time and temporal coverage.

The script reads the three published Guangzhou 15-second mobile files,
calculates device-day operating-time and gap metrics, summarizes daily and
date-hour coverage, and generates publication-ready tables and figures.
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm, PowerNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths and analysis settings
# ============================================================

# Edit these paths when running the script directly in an IDE. Command-line
# arguments are optional and, when supplied, override the values below.
DATA_DIR = Path(".")
OUTPUT_DIR = Path("results") / "device_operation_time_coverage"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
ANALYSIS_DATA_DIR = OUTPUT_DIR / "analysis_data"

TEMP_DIR = Path("duckdb_tmp_device_operation_time_coverage")

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
    "2023-03": "#3B6FB6",
    "2023-08": "#D9822B",
    "2023-11": "#3A8D6D",
}

# Optional local font files may be placed here or supplied with --font-dir.
FONT_DIR = Path("fonts")

# Only non-negative, within-day intervals no longer than this threshold are
# counted as device operating time. Longer intervals are counted as gaps.
GAP_THRESHOLD_SECONDS = 60

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
    FIGURE_DIR / "Figure_combined_activity_and_time_coverage.png",
    FIGURE_DIR / "Figure_combined_activity_and_time_coverage.pdf",
    TABLE_DIR / "Table_TV3_monthly_long_gap_summary.csv",
    ANALYSIS_DATA_DIR / "device_daily_metrics.csv",
    ANALYSIS_DATA_DIR / "daily_network_metrics.csv",
    ANALYSIS_DATA_DIR / "date_hour_record_counts.csv",
]


# ============================================================
# 2. Input checks and DuckDB queries
# ============================================================

def sql_path(path: Path) -> str:
    """Escape a filesystem path for use in a DuckDB SQL string."""
    return str(path).replace("'", "''")


def read_csv_header(path: Path) -> list[str]:
    """Read only the header of a gzip-compressed CSV file."""
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as file:
        return next(csv.reader(file))


def validate_inputs_and_outputs() -> None:
    """Check all input schemas and protect existing output files."""
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


def query_device_daily(
    con: duckdb.DuckDBPyConnection,
    month: str,
    path: Path,
) -> pd.DataFrame:
    """Calculate record, operating-time and long-gap metrics per device-day."""
    source = csv_source(path)
    result = con.execute(
        f"""
        WITH valid_records AS (
            SELECT
                DEVICE_ID,
                TIME_POINT,
                DEVICE_TIME_QC,
                CAST(TIME_POINT AS DATE) AS observation_date
            FROM {source}
            WHERE DEVICE_TIME_QC = 0
              AND DEVICE_ID IS NOT NULL
              AND TRIM(DEVICE_ID) <> ''
              AND TIME_POINT IS NOT NULL
        ),
        ordered AS (
            SELECT
                *,
                LAG(TIME_POINT) OVER (
                    PARTITION BY DEVICE_ID
                    ORDER BY TIME_POINT
                ) AS previous_time
            FROM valid_records
        ),
        intervals AS (
            SELECT
                *,
                CASE
                    WHEN previous_time IS NOT NULL
                     AND CAST(previous_time AS DATE) = observation_date
                    THEN DATE_DIFF('second', previous_time, TIME_POINT)
                    ELSE NULL
                END AS interval_seconds
            FROM ordered
        )
        SELECT
            observation_date,
            DEVICE_ID,
            COUNT(*)::BIGINT AS record_count,
            COUNT(*) FILTER (WHERE DEVICE_TIME_QC = 0)::BIGINT
                AS valid_record_count,
            MIN(TIME_POINT) AS first_sample_time,
            MAX(TIME_POINT) AS last_sample_time,
            SUM(
                CASE
                    WHEN interval_seconds BETWEEN 0
                         AND {GAP_THRESHOLD_SECONDS}
                    THEN interval_seconds
                    ELSE 0
                END
            ) / 3600.0 AS operating_hours,
            COUNT(*) FILTER (
                WHERE interval_seconds > {GAP_THRESHOLD_SECONDS}
            )::BIGINT AS long_gap_count,
            SUM(
                CASE
                    WHEN interval_seconds > {GAP_THRESHOLD_SECONDS}
                    THEN interval_seconds
                    ELSE 0
                END
            ) / 3600.0 AS long_gap_duration_hours,
            MAX(
                CASE
                    WHEN interval_seconds > {GAP_THRESHOLD_SECONDS}
                    THEN interval_seconds
                    ELSE NULL
                END
            ) / 3600.0 AS maximum_gap_hours
        FROM intervals
        GROUP BY observation_date, DEVICE_ID
        ORDER BY observation_date, DEVICE_ID
        """
    ).df()
    result.insert(0, "month", month)
    return result


def query_date_hour_counts(
    con: duckdb.DuckDBPyConnection,
    month: str,
    path: Path,
) -> pd.DataFrame:
    """Count structurally valid records by calendar date and hour."""
    source = csv_source(path)
    result = con.execute(
        f"""
        SELECT
            CAST(TIME_POINT AS DATE) AS observation_date,
            EXTRACT(HOUR FROM TIME_POINT)::INTEGER AS hour,
            COUNT(*)::BIGINT AS record_count
        FROM {source}
        WHERE DEVICE_TIME_QC = 0
          AND DEVICE_ID IS NOT NULL
          AND TRIM(DEVICE_ID) <> ''
          AND TIME_POINT IS NOT NULL
        GROUP BY observation_date, hour
        ORDER BY observation_date, hour
        """
    ).df()
    result.insert(0, "month", month)
    return result


# ============================================================
# 3. Summary tables
# ============================================================

def build_daily_network_metrics(device_daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate device-day metrics to one row per calendar date."""
    daily = (
        device_daily.groupby(["month", "observation_date"], as_index=False)
        .agg(
            record_count=("record_count", "sum"),
            valid_record_count=("valid_record_count", "sum"),
            active_device_count=("DEVICE_ID", "nunique"),
            total_operating_hours=("operating_hours", "sum"),
            long_gap_count=("long_gap_count", "sum"),
        )
        .sort_values(["month", "observation_date"])
    )
    return daily


def build_monthly_gap_summary(device_daily: pd.DataFrame) -> pd.DataFrame:
    """Create the non-graphical monthly summary focused on long gaps."""
    rows: list[dict[str, object]] = []
    for month in MONTH_FILES:
        group = device_daily.loc[device_daily["month"] == month]
        days_with_gap = int((group["long_gap_count"] > 0).sum())
        device_days = int(len(group))
        rows.append(
            {
                "month": month,
                "active_devices": int(group["DEVICE_ID"].nunique()),
                "device_days": device_days,
                "device_days_with_long_gap": days_with_gap,
                "device_days_with_long_gap_percent": (
                    100.0 * days_with_gap / device_days
                    if device_days else np.nan
                ),
                "total_long_gap_count": int(group["long_gap_count"].sum()),
                "total_within_day_gap_duration_hours": float(
                    group["long_gap_duration_hours"].sum()
                ),
                "maximum_within_day_gap_hours": float(
                    group["maximum_gap_hours"].max()
                ) if group["maximum_gap_hours"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def complete_daily_calendar(daily: pd.DataFrame, month: str) -> pd.DataFrame:
    """Insert zero rows for calendar dates without any valid records."""
    start = pd.Timestamp(f"{month}-01")
    days = calendar.monthrange(start.year, start.month)[1]
    dates = pd.date_range(start, periods=days, freq="D")
    subset = daily.loc[daily["month"] == month].copy()
    subset["observation_date"] = pd.to_datetime(subset["observation_date"])
    subset = subset.set_index("observation_date").reindex(dates)
    numeric = [
        "record_count",
        "valid_record_count",
        "active_device_count",
        "total_operating_hours",
        "long_gap_count",
    ]
    subset[numeric] = subset[numeric].fillna(0)
    subset["month"] = month
    subset.index.name = "observation_date"
    return subset.reset_index()


# ============================================================
# 4. Publication-style figures
# ============================================================

def set_figure_style() -> None:
    """Apply a restrained journal-style Matplotlib theme."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def clean_axis(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7)
        ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_operating_duration_distribution(device_daily: pd.DataFrame) -> None:
    """Draw device-day operating-duration distributions by month."""
    months = list(MONTH_FILES)
    datasets = [
        device_daily.loc[
            device_daily["month"] == month, "operating_hours"
        ].to_numpy(dtype=float)
        for month in months
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.7))
    violins = ax.violinplot(
        datasets,
        positions=np.arange(1, len(months) + 1),
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, month in zip(violins["bodies"], months):
        body.set_facecolor(MONTH_COLORS[month])
        body.set_edgecolor(MONTH_COLORS[month])
        body.set_alpha(0.38)
        body.set_linewidth(0.8)

    box = ax.boxplot(
        datasets,
        positions=np.arange(1, len(months) + 1),
        widths=0.17,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.15},
        whiskerprops={"color": "#303030", "linewidth": 0.8},
        capprops={"color": "#303030", "linewidth": 0.8},
        boxprops={"color": "#303030", "linewidth": 0.8},
    )
    for patch, month in zip(box["boxes"], months):
        patch.set_facecolor(MONTH_COLORS[month])
        patch.set_alpha(0.78)

    ax.set_xticks(
        np.arange(1, len(months) + 1),
        [MONTH_LABELS[month].replace(" 2023", "") for month in months],
    )
    ax.set_ylabel("Daily operating duration per device (h)")
    ax.set_ylim(bottom=0)
    clean_axis(ax)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.17, top=0.96)
    save_figure(
        fig,
        FIGURE_DIR / "Figure_daily_operating_duration_distribution",
    )


def plot_daily_network_activity(daily: pd.DataFrame) -> None:
    """Draw daily records and active devices without dual axes."""
    months = list(MONTH_FILES)
    completed = {
        month: complete_daily_calendar(daily, month) for month in months
    }
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.4, 5.2),
        sharey="row",
        constrained_layout=True,
    )

    for column, month in enumerate(months):
        data = completed[month]
        days = data["observation_date"].dt.day.to_numpy()
        color = MONTH_COLORS[month]

        axes[0, column].plot(
            days,
            data["record_count"],
            color=color,
            linewidth=1.25,
            marker="o",
            markersize=2.4,
            markeredgewidth=0,
        )
        axes[1, column].plot(
            days,
            data["active_device_count"],
            color=color,
            linewidth=1.25,
            marker="o",
            markersize=2.4,
            markeredgewidth=0,
        )
        axes[0, column].set_title(MONTH_LABELS[month])

        for row in range(2):
            ax = axes[row, column]
            ax.set_xlim(1, int(days.max()))
            ax.set_xticks([1, 5, 10, 15, 20, 25, int(days.max())])
            clean_axis(ax)
            ax.text(
                0.02,
                0.95,
                chr(ord("a") + row * 3 + column),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontweight="bold",
            )

    axes[0, 0].set_ylabel("Daily record count")
    axes[1, 0].set_ylabel("Active device count")
    for ax in axes[1, :]:
        ax.set_xlabel("Day of month")

    save_figure(
        fig,
        FIGURE_DIR / "Figure_daily_records_and_active_devices",
    )


def heatmap_matrix(
    date_hour: pd.DataFrame,
    month: str,
) -> tuple[np.ndarray, int]:
    """Build a 24 x calendar-day count matrix, retaining zero cells."""
    start = pd.Timestamp(f"{month}-01")
    days = calendar.monthrange(start.year, start.month)[1]
    matrix = np.zeros((24, days), dtype=float)
    subset = date_hour.loc[date_hour["month"] == month].copy()
    subset["observation_date"] = pd.to_datetime(subset["observation_date"])
    for row in subset.itertuples(index=False):
        day = int(row.observation_date.day)
        matrix[int(row.hour), day - 1] = float(row.record_count)
    return matrix, days


def plot_date_hour_heatmap(date_hour: pd.DataFrame) -> None:
    """Draw date-by-hour record-count heatmaps with shared scaling."""
    months = list(MONTH_FILES)
    matrices = [heatmap_matrix(date_hour, month) for month in months]
    maximum = max(float(matrix.max()) for matrix, _ in matrices)
    norm = LogNorm(vmin=1, vmax=max(1.01, maximum))
    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad("#F1F1F1")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.5, 3.65),
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for column, (ax, month, result) in enumerate(
        zip(axes, months, matrices)
    ):
        matrix, days = result
        masked = np.ma.masked_less_equal(matrix, 0)
        image = ax.imshow(
            masked,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            extent=[0.5, days + 0.5, -0.5, 23.5],
        )
        ax.set_title(MONTH_LABELS[month])
        ax.set_xlabel("Day of month")
        ax.set_xticks([1, 5, 10, 15, 20, 25, days])
        ax.set_yticks([0, 4, 8, 12, 16, 20, 23])
        ax.text(
            0.02,
            0.96,
            chr(ord("a") + column),
            transform=ax.transAxes,
            va="top",
            ha="left",
            color="#202020",
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 1.2,
            },
        )
    axes[0].set_ylabel("Hour of day (China Standard Time)")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, pad=0.025, fraction=0.025)
        colorbar.set_label("Record count per date-hour")
        colorbar.outline.set_linewidth(0.7)

    save_figure(
        fig,
        FIGURE_DIR / "Figure_date_hour_record_heatmap",
    )


def set_combined_figure_style() -> None:
    """Apply the final typography used for the combined nine-panel figure."""
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

    print(f"Combined-figure font: {font_family}")
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_combined_activity_and_time_coverage(
    daily: pd.DataFrame,
    date_hour: pd.DataFrame,
) -> None:
    """Draw the final nine-panel activity and temporal-coverage figure."""
    set_combined_figure_style()

    months = list(MONTH_FILES)
    completed = {
        month: complete_daily_calendar(daily, month) for month in months
    }
    matrices = {
        month: heatmap_matrix(date_hour, month) for month in months
    }

    all_counts = np.concatenate(
        [matrix.ravel() for matrix, _ in matrices.values()]
    )
    positive_counts = all_counts[all_counts > 0]
    color_maximum = (
        float(np.quantile(positive_counts, 0.995))
        if positive_counts.size
        else 1.0
    )

    coverage_cmap = LinearSegmentedColormap.from_list(
        "coverage_blue",
        [
            "#F7FBFF",
            "#DCEAF4",
            "#9ECAE1",
            "#4292C6",
            "#1764A5",
            "#08306B",
        ],
    )
    coverage_norm = PowerNorm(
        gamma=0.72,
        vmin=0,
        vmax=max(1.0, color_maximum),
        clip=True,
    )

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(8.0, 7.2),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": [0.85, 0.75, 1.55]},
    )
    fig.subplots_adjust(
        left=0.125,
        right=0.865,
        bottom=0.09,
        top=0.93,
        wspace=0.16,
        hspace=0.25,
    )

    maximum_daily_records = max(
        float(data["record_count"].max())
        for data in completed.values()
    )
    minimum_active_devices = min(
        float(data["active_device_count"].min())
        for data in completed.values()
    )
    maximum_active_devices = max(
        float(data["active_device_count"].max())
        for data in completed.values()
    )
    active_axis_minimum = max(
        0.0,
        5.0 * np.floor(minimum_active_devices / 5.0) - 5.0,
    )
    active_axis_maximum = (
        5.0 * np.ceil(maximum_active_devices / 5.0) + 5.0
    )

    image = None
    for column, month in enumerate(months):
        data = completed[month]
        matrix, days_in_month = matrices[month]
        days = data["observation_date"].dt.day.to_numpy()
        color = MONTH_COLORS[month]
        record_axis = axes[0, column]
        active_axis = axes[1, column]
        heatmap_axis = axes[2, column]

        records_million = (
            data["record_count"].to_numpy(dtype=float) / 1_000_000.0
        )
        record_axis.plot(
            days,
            records_million,
            color=color,
            linewidth=1.55,
            solid_capstyle="round",
        )
        record_axis.fill_between(
            days,
            records_million,
            0,
            color=color,
            alpha=0.055,
            linewidth=0,
        )
        record_axis.set_title(
            MONTH_LABELS[month],
            pad=10,
            fontsize=12,
            fontweight="normal",
        )
        record_axis.set_ylim(
            0,
            maximum_daily_records / 1_000_000.0 * 1.05,
        )

        active_axis.plot(
            days,
            data["active_device_count"],
            color=color,
            linewidth=1.45,
            solid_capstyle="round",
        )
        active_axis.scatter(
            days,
            data["active_device_count"],
            s=6,
            color=color,
            linewidth=0,
            zorder=3,
        )
        active_axis.set_ylim(active_axis_minimum, active_axis_maximum)

        image = heatmap_axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=coverage_cmap,
            norm=coverage_norm,
            extent=[0.5, days_in_month + 0.5, -0.5, 23.5],
        )
        heatmap_axis.set_xticks(
            [1, 5, 10, 15, 20, 25, days_in_month]
        )
        heatmap_axis.set_yticks([0, 6, 12, 18, 23])
        heatmap_axis.set_xlabel("Day of month", labelpad=6)

        for row in range(3):
            axis = axes[row, column]
            axis.set_xlim(0.5, days_in_month + 0.5)
            axis.tick_params(
                axis="both",
                which="major",
                labelsize=10.5,
                length=3.5,
                width=0.8,
                pad=3,
            )
            axis.annotate(
                chr(ord("a") + row * 3 + column),
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

        for axis in (record_axis, active_axis):
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["bottom"].set_visible(False)
            axis.tick_params(axis="x", bottom=False, labelbottom=False)
            axis.grid(
                axis="y",
                color="#D9DEE3",
                linewidth=0.5,
                alpha=0.85,
            )
            axis.set_axisbelow(True)

    axes[0, 0].set_ylabel(
        "Daily records\n" + r"($\times10^{6}$)",
        labelpad=6,
    )
    axes[1, 0].set_ylabel("Active devices", labelpad=6)
    axes[2, 0].set_ylabel("Hour of day (CST)", labelpad=6)

    axes[0, 0].yaxis.set_major_locator(
        MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
    )
    axes[0, 0].yaxis.set_major_formatter(
        FuncFormatter(lambda value, position: f"{value:.2f}")
    )
    axes[1, 0].yaxis.set_major_locator(
        MaxNLocator(nbins=4, integer=True)
    )

    if image is not None:
        position = axes[2, 2].get_position()
        colorbar_axis = fig.add_axes(
            [0.887, position.y0, 0.016, position.height]
        )
        colorbar = fig.colorbar(
            image,
            cax=colorbar_axis,
        )
        colorbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, position: f"{value / 1000:g}")
        )
        colorbar.set_label(
            r"Records per date-hour ($\times10^{3}$)",
            fontsize=12,
            fontweight="normal",
            labelpad=8,
        )
        colorbar.ax.tick_params(
            labelsize=10.5,
            length=3.5,
            width=0.8,
            pad=3,
        )
        colorbar.outline.set_linewidth(0.7)

    stem = FIGURE_DIR / "Figure_combined_activity_and_time_coverage"
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
        OUTPUT_DIR = DATA_DIR / "results" / "device_operation_time_coverage"
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
        FIGURE_DIR / "Figure_combined_activity_and_time_coverage.png",
        FIGURE_DIR / "Figure_combined_activity_and_time_coverage.pdf",
        TABLE_DIR / "Table_TV3_monthly_long_gap_summary.csv",
        ANALYSIS_DATA_DIR / "device_daily_metrics.csv",
        ANALYSIS_DATA_DIR / "daily_network_metrics.csv",
        ANALYSIS_DATA_DIR / "date_hour_record_counts.csv",
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
    """Run the complete device operation-time coverage analysis."""
    configure_runtime(parse_args())
    started = perf_counter()
    validate_inputs_and_outputs()

    for directory in [FIGURE_DIR, TABLE_DIR, ANALYSIS_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    con = configure_duckdb()
    try:
        device_daily_parts = []
        date_hour_parts = []
        for month, path in MONTH_FILES.items():
            print(f"Processing {month}: {path.name}")
            device_daily_parts.append(query_device_daily(con, month, path))
            date_hour_parts.append(query_date_hour_counts(con, month, path))
    finally:
        con.close()

    device_daily = pd.concat(device_daily_parts, ignore_index=True)
    date_hour = pd.concat(date_hour_parts, ignore_index=True)
    daily_network = build_daily_network_metrics(device_daily)
    monthly_gap = build_monthly_gap_summary(device_daily)

    # Stable numeric precision keeps the analysis products compact and clear.
    float_columns = [
        "operating_hours",
        "long_gap_duration_hours",
        "maximum_gap_hours",
    ]
    device_daily[float_columns] = device_daily[float_columns].round(6)
    daily_network["total_operating_hours"] = daily_network[
        "total_operating_hours"
    ].round(6)
    monthly_gap[
        [
            "device_days_with_long_gap_percent",
            "total_within_day_gap_duration_hours",
            "maximum_within_day_gap_hours",
        ]
    ] = monthly_gap[
        [
            "device_days_with_long_gap_percent",
            "total_within_day_gap_duration_hours",
            "maximum_within_day_gap_hours",
        ]
    ].round(3)

    device_daily.to_csv(
        ANALYSIS_DATA_DIR / "device_daily_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily_network.to_csv(
        ANALYSIS_DATA_DIR / "daily_network_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    date_hour.to_csv(
        ANALYSIS_DATA_DIR / "date_hour_record_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    monthly_gap.to_csv(
        TABLE_DIR / "Table_TV3_monthly_long_gap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_combined_activity_and_time_coverage(daily_network, date_hour)

    elapsed = perf_counter() - started
    print(f"Completed in {elapsed / 60:.2f} minutes")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
