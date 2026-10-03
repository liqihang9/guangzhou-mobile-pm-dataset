"""Evaluate and visualize the spatial coverage of the Guangzhou release.

The script performs the complete spatial-coverage validation, writes the
reproducibility tables and GIS layers, and creates the manuscript figures.
All paths are supplied through command-line options; the script can therefore
be renamed or moved without changing its behaviour.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import shutil
import zipfile
from io import BytesIO
from pathlib import Path
from time import perf_counter

try:
    import duckdb
except ImportError:  # Allow --help to work before optional dependencies are installed.
    duckdb = None
import geopandas as gpd
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import font_manager
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, TextArea, VPacker
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator, NullLocator
import numpy as np
import pandas as pd
from shapely.geometry import box as geometry_box


# ============================================================
# 1. Paths and analysis settings
# ============================================================

RELEASE_DIR = Path("data")
SPATIAL_REFERENCE_DIR = Path("spatial_reference")
OUTPUT_DIR = Path("results") / "spatial_coverage"
TEMP_DIR = Path("duckdb_tmp_spatial_coverage")
FONT_DIR = Path("fonts")

FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
ANALYSIS_DATA_DIR = OUTPUT_DIR / "analysis_data"
REFERENCE_EXTRACT_DIR = ANALYSIS_DATA_DIR / "reference_layers"
ADMIN_BOUNDARY_ZIP = SPATIAL_REFERENCE_DIR / "Guangzhou_district_boundaries.zip"
ROAD_NETWORK_ZIP = SPATIAL_REFERENCE_DIR / "Guangzhou_road_network.zip"

MONTH_FILES = {
    "2023-03": RELEASE_DIR / "Guangzhou_mobile_2023-03_15s_QC.csv.gz",
    "2023-08": RELEASE_DIR / "Guangzhou_mobile_2023-08_15s_QC.csv.gz",
    "2023-11": RELEASE_DIR / "Guangzhou_mobile_2023-11_15s_QC.csv.gz",
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
    "All months": "#252525",
}

DISTRICT_ORDER = [
    "Yuexiu",
    "Liwan",
    "Haizhu",
    "Tianhe",
    "Baiyun",
    "Huangpu",
    "Panyu",
    "Huadu",
    "Nansha",
    "Conghua",
    "Zengcheng",
]

# Manual offsets keep labels for the compact central districts from
# overlapping one another. Values are display points, not map metres.
DISTRICT_LABEL_OFFSETS_POINTS = {
    "Liwan": (-37, 17),
    "Yuexiu": (-25, -11),
    "Haizhu": (-12, -30),
    "Tianhe": (34, -7),
    "Baiyun": (0, 16),
    "Huangpu": (27, 11),
    "Panyu": (0, -14),
}

# Administrative boundary source: WGS 1984 World Mercator.
ADMIN_SOURCE_CRS = "EPSG:3395"
# Road source: WGS 84 longitude/latitude.
ROAD_SOURCE_CRS = "EPSG:4326"
# Guangzhou analysis CRS: CGCS2000 / 3-degree Gauss-Kruger CM 114E.
ANALYSIS_CRS = "EPSG:4547"

GRID_SIZES_M = (500, 1000)
MAIN_GRID_SIZE_M = 500

# To reduce tens of millions of GPS records before nearest-road matching,
# records in the same 25 m cell are represented by their mean coordinate.
ROAD_MATCH_SAMPLE_CELL_M = 25
MAIN_ROAD_MATCH_DISTANCE_M = 50
ROAD_MATCH_SENSITIVITY_M = (30, 50, 100)
ROAD_MATCH_CHUNK_SIZE = 250_000

DRIVABLE_ROAD_CLASSES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
}

ROAD_GROUP_ORDER = [
    "Expressways",
    "Primary roads",
    "Secondary roads",
    "Local roads",
    "Service roads",
]

SCOPES = [*MONTH_LABELS, "All months"]
MARKERS = ["o", "s", "^", "D"]

DPI = 600
TEXT_SIZE = 13.5
TICK_SIZE = 11.5
LEGEND_SIZE = 11.5
MAP_LABEL_SIZE = 11.5
CELL_TEXT_SIZE = 12
SCALE_TEXT_SIZE = 11.5
NORTH_POSITION = (0.11, 0.94)
NORTH_SCALE = 0.75
NORTH_MAP_GAP_PT = 10.0
MAP_EXTRA_LEFT_PADDING = 0.10

OVERWRITE_EXISTING = False
REDRAW_FIGURES_ONLY = False
KEEP_PROJECTED_POINT_PARQUET = False

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

FIGURE_1_PNG = FIGURE_DIR / "Figure_spatial_density_and_persistence.png"
FIGURE_1_PDF = FIGURE_DIR / "Figure_spatial_density_and_persistence.pdf"
FIGURE_2_PNG = FIGURE_DIR / "Figure_district_and_road_coverage.png"
FIGURE_2_PDF = FIGURE_DIR / "Figure_district_and_road_coverage.pdf"
APPENDIX_FIGURE_1_PNG = FIGURE_DIR / "Figure_1km_sampling_density.png"
APPENDIX_FIGURE_1_PDF = FIGURE_DIR / "Figure_1km_sampling_density.pdf"
APPENDIX_FIGURE_2_PNG = FIGURE_DIR / "Figure_road_match_sensitivity.png"
APPENDIX_FIGURE_2_PDF = FIGURE_DIR / "Figure_road_match_sensitivity.pdf"

MONTHLY_SUMMARY_CSV = TABLE_DIR / "Table_TV5_monthly_spatial_coverage_summary.csv"
DISTRICT_SUMMARY_CSV = TABLE_DIR / "Table_TV6_district_grid_coverage.csv"
ROAD_CLASS_SUMMARY_CSV = TABLE_DIR / "Table_TV7_road_coverage_by_class.csv"
ROAD_SENSITIVITY_CSV = TABLE_DIR / "Table_TV8_road_match_sensitivity.csv"
ROAD_CLASS_SENSITIVITY_CSV = (
    TABLE_DIR / "Table_TV9_road_class_match_sensitivity.csv"
)

GRID_500_GPKG = ANALYSIS_DATA_DIR / "grid_metrics_500m.gpkg"
GRID_1000_GPKG = ANALYSIS_DATA_DIR / "grid_metrics_1000m.gpkg"
ROAD_COVERAGE_GPKG = ANALYSIS_DATA_DIR / "road_coverage.gpkg"

OUTPUT_FILES = [
    FIGURE_1_PNG,
    FIGURE_1_PDF,
    FIGURE_2_PNG,
    FIGURE_2_PDF,
    APPENDIX_FIGURE_1_PNG,
    APPENDIX_FIGURE_1_PDF,
    APPENDIX_FIGURE_2_PNG,
    APPENDIX_FIGURE_2_PDF,
    MONTHLY_SUMMARY_CSV,
    DISTRICT_SUMMARY_CSV,
    ROAD_CLASS_SUMMARY_CSV,
    ROAD_SENSITIVITY_CSV,
    ROAD_CLASS_SENSITIVITY_CSV,
    GRID_500_GPKG,
    GRID_1000_GPKG,
    ROAD_COVERAGE_GPKG,
]

FIGURE_OUTPUT_FILES = [
    FIGURE_1_PNG,
    FIGURE_1_PDF,
    FIGURE_2_PNG,
    FIGURE_2_PDF,
    APPENDIX_FIGURE_1_PNG,
    APPENDIX_FIGURE_1_PDF,
    APPENDIX_FIGURE_2_PNG,
    APPENDIX_FIGURE_2_PDF,
]


# ============================================================
# 2. General utilities and input checks
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Guangzhou mobile-observation spatial coverage and "
            "create the associated tables, GIS layers and figures."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RELEASE_DIR,
        help="Directory containing the three published monthly CSV.GZ files.",
    )
    parser.add_argument(
        "--spatial-reference-dir",
        type=Path,
        default=SPATIAL_REFERENCE_DIR,
        help=(
            "Directory containing 'Guangzhou_district_boundaries.zip' and "
            "'Guangzhou_road_network.zip'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Root directory for generated figures, tables and GIS layers.",
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
        help="Optional directory containing Arial or other sans-serif font files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs.",
    )
    parser.add_argument(
        "--redraw-figures-only",
        action="store_true",
        help="Reuse completed tables and GIS layers and regenerate only the figures.",
    )
    parser.add_argument(
        "--keep-projected-points",
        action="store_true",
        help="Retain intermediate projected point Parquet files.",
    )
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    """Apply command-line paths without relying on the script filename."""
    global RELEASE_DIR, SPATIAL_REFERENCE_DIR, OUTPUT_DIR, TEMP_DIR, FONT_DIR
    global FIGURE_DIR, TABLE_DIR, ANALYSIS_DATA_DIR, REFERENCE_EXTRACT_DIR
    global ADMIN_BOUNDARY_ZIP, ROAD_NETWORK_ZIP, MONTH_FILES
    global FIGURE_1_PNG, FIGURE_1_PDF, FIGURE_2_PNG, FIGURE_2_PDF
    global APPENDIX_FIGURE_1_PNG, APPENDIX_FIGURE_1_PDF
    global APPENDIX_FIGURE_2_PNG, APPENDIX_FIGURE_2_PDF
    global MONTHLY_SUMMARY_CSV, DISTRICT_SUMMARY_CSV, ROAD_CLASS_SUMMARY_CSV
    global ROAD_SENSITIVITY_CSV, ROAD_CLASS_SENSITIVITY_CSV
    global GRID_500_GPKG, GRID_1000_GPKG, ROAD_COVERAGE_GPKG
    global OUTPUT_FILES, FIGURE_OUTPUT_FILES
    global OVERWRITE_EXISTING, REDRAW_FIGURES_ONLY, KEEP_PROJECTED_POINT_PARQUET

    RELEASE_DIR = args.data_dir.expanduser().resolve()
    SPATIAL_REFERENCE_DIR = args.spatial_reference_dir.expanduser().resolve()
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    TEMP_DIR = args.temp_dir.expanduser().resolve()
    FONT_DIR = args.font_dir.expanduser().resolve()

    FIGURE_DIR = OUTPUT_DIR / "figures"
    TABLE_DIR = OUTPUT_DIR / "tables"
    ANALYSIS_DATA_DIR = OUTPUT_DIR / "analysis_data"
    REFERENCE_EXTRACT_DIR = ANALYSIS_DATA_DIR / "reference_layers"
    ADMIN_BOUNDARY_ZIP = SPATIAL_REFERENCE_DIR / "Guangzhou_district_boundaries.zip"
    ROAD_NETWORK_ZIP = SPATIAL_REFERENCE_DIR / "Guangzhou_road_network.zip"
    MONTH_FILES = {
        "2023-03": RELEASE_DIR / "Guangzhou_mobile_2023-03_15s_QC.csv.gz",
        "2023-08": RELEASE_DIR / "Guangzhou_mobile_2023-08_15s_QC.csv.gz",
        "2023-11": RELEASE_DIR / "Guangzhou_mobile_2023-11_15s_QC.csv.gz",
    }

    FIGURE_1_PNG = FIGURE_DIR / "Figure_spatial_density_and_persistence.png"
    FIGURE_1_PDF = FIGURE_DIR / "Figure_spatial_density_and_persistence.pdf"
    FIGURE_2_PNG = FIGURE_DIR / "Figure_district_and_road_coverage.png"
    FIGURE_2_PDF = FIGURE_DIR / "Figure_district_and_road_coverage.pdf"
    APPENDIX_FIGURE_1_PNG = FIGURE_DIR / "Figure_1km_sampling_density.png"
    APPENDIX_FIGURE_1_PDF = FIGURE_DIR / "Figure_1km_sampling_density.pdf"
    APPENDIX_FIGURE_2_PNG = FIGURE_DIR / "Figure_road_match_sensitivity.png"
    APPENDIX_FIGURE_2_PDF = FIGURE_DIR / "Figure_road_match_sensitivity.pdf"

    MONTHLY_SUMMARY_CSV = TABLE_DIR / "Table_TV5_monthly_spatial_coverage_summary.csv"
    DISTRICT_SUMMARY_CSV = TABLE_DIR / "Table_TV6_district_grid_coverage.csv"
    ROAD_CLASS_SUMMARY_CSV = TABLE_DIR / "Table_TV7_road_coverage_by_class.csv"
    ROAD_SENSITIVITY_CSV = TABLE_DIR / "Table_TV8_road_match_sensitivity.csv"
    ROAD_CLASS_SENSITIVITY_CSV = TABLE_DIR / "Table_TV9_road_class_match_sensitivity.csv"
    GRID_500_GPKG = ANALYSIS_DATA_DIR / "grid_metrics_500m.gpkg"
    GRID_1000_GPKG = ANALYSIS_DATA_DIR / "grid_metrics_1000m.gpkg"
    ROAD_COVERAGE_GPKG = ANALYSIS_DATA_DIR / "road_coverage.gpkg"

    FIGURE_OUTPUT_FILES = [
        FIGURE_1_PNG, FIGURE_1_PDF, FIGURE_2_PNG, FIGURE_2_PDF,
        APPENDIX_FIGURE_1_PNG, APPENDIX_FIGURE_1_PDF,
        APPENDIX_FIGURE_2_PNG, APPENDIX_FIGURE_2_PDF,
    ]
    OUTPUT_FILES = [
        *FIGURE_OUTPUT_FILES, MONTHLY_SUMMARY_CSV, DISTRICT_SUMMARY_CSV,
        ROAD_CLASS_SUMMARY_CSV, ROAD_SENSITIVITY_CSV,
        ROAD_CLASS_SENSITIVITY_CSV, GRID_500_GPKG, GRID_1000_GPKG,
        ROAD_COVERAGE_GPKG,
    ]
    OVERWRITE_EXISTING = args.overwrite
    REDRAW_FIGURES_ONLY = args.redraw_figures_only
    KEEP_PROJECTED_POINT_PARQUET = args.keep_projected_points


def sql_path(path: Path) -> str:
    """Escape a filesystem path for use in a DuckDB SQL string."""
    return str(path).replace("'", "''")


def read_csv_header(path: Path) -> list[str]:
    """Read only the header from one gzip-compressed release CSV."""
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as file:
        return next(csv.reader(file))


def validate_inputs_and_outputs() -> None:
    """Validate release files, spatial references and output protection."""
    for path in (ADMIN_BOUNDARY_ZIP, ROAD_NETWORK_ZIP):
        if not path.exists():
            raise FileNotFoundError(f"Missing spatial reference ZIP: {path}")

    if REDRAW_FIGURES_ONLY:
        required_results = [
            GRID_500_GPKG,
            GRID_1000_GPKG,
            ROAD_COVERAGE_GPKG,
            DISTRICT_SUMMARY_CSV,
            ROAD_CLASS_SUMMARY_CSV,
            ROAD_SENSITIVITY_CSV,
            ROAD_CLASS_SENSITIVITY_CSV,
        ]
        missing_results = [path for path in required_results if not path.exists()]
        if missing_results:
            listed = "\n".join(f"  - {path}" for path in missing_results)
            raise FileNotFoundError(
                "--redraw-figures-only requires completed analysis files:\n"
                f"{listed}"
            )
        existing_figures = [path for path in FIGURE_OUTPUT_FILES if path.exists()]
        if existing_figures and not OVERWRITE_EXISTING:
            listed = "\n".join(f"  - {path}" for path in existing_figures)
            raise FileExistsError(
                "Figure outputs already exist. Add --overwrite to replace them:\n"
                f"{listed}"
            )
        return

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
            "Spatial-validation outputs already exist. Add --overwrite to "
            f"replace them:\n{listed}"
        )


def prepare_output_directories() -> None:
    """Create output and temporary directories."""
    for path in (
        FIGURE_DIR,
        TABLE_DIR,
        ANALYSIS_DATA_DIR,
        REFERENCE_EXTRACT_DIR,
        TEMP_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

    if REDRAW_FIGURES_ONLY and OVERWRITE_EXISTING:
        for path in FIGURE_OUTPUT_FILES:
            if path.exists() and path.is_file():
                path.unlink()
    elif OVERWRITE_EXISTING:
        for path in OUTPUT_FILES:
            if path.exists() and path.is_file():
                path.unlink()


def extract_shapefile(zip_path: Path, destination: Path) -> Path:
    """Safely extract Shapefile components and return the .shp path."""
    destination.mkdir(parents=True, exist_ok=True)
    allowed_suffixes = {
        ".shp",
        ".shx",
        ".dbf",
        ".prj",
        ".cpg",
        ".sbn",
        ".sbx",
        ".xml",
    }
    extracted_shp: Path | None = None

    with zipfile.ZipFile(zip_path) as archive:
        for item in archive.infolist():
            name = Path(item.filename).name
            if not name or ".sr.lock" in name:
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in allowed_suffixes:
                continue
            target = destination / name
            with archive.open(item) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            if suffix == ".shp":
                extracted_shp = target

    if extracted_shp is None:
        raise RuntimeError(f"No .shp file found in {zip_path}")
    return extracted_shp


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


def configure_duckdb() -> duckdb.DuckDBPyConnection:
    """Create and configure the analytical DuckDB connection."""
    if duckdb is None:
        raise ModuleNotFoundError(
            "DuckDB is required for the full analysis. Install it with 'pip install duckdb'."
        )
    database_path = TEMP_DIR / "spatial_coverage.duckdb"
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
# 3. Spatial reference layers
# ============================================================

def road_group(fclass: str) -> str:
    """Collapse detailed OSM road classes into five manuscript groups."""
    if fclass in {"motorway", "motorway_link", "trunk", "trunk_link"}:
        return "Expressways"
    if fclass in {"primary", "primary_link"}:
        return "Primary roads"
    if fclass in {"secondary", "secondary_link"}:
        return "Secondary roads"
    if fclass in {
        "tertiary",
        "tertiary_link",
        "unclassified",
        "residential",
    }:
        return "Local roads"
    return "Service roads"


def load_and_prepare_reference_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Path]:
    """Extract, validate, reproject and clip district and road layers."""
    admin_shp = extract_shapefile(
        ADMIN_BOUNDARY_ZIP,
        REFERENCE_EXTRACT_DIR / "administrative_boundary",
    )
    road_shp = extract_shapefile(
        ROAD_NETWORK_ZIP,
        REFERENCE_EXTRACT_DIR / "road_network",
    )

    districts = gpd.read_file(admin_shp, encoding="utf-8")
    required_district_fields = {"Name", "AdminCode"}
    missing = required_district_fields - set(districts.columns)
    if missing:
        raise RuntimeError(f"Administrative boundary is missing fields: {missing}")
    if len(districts) != 11:
        raise RuntimeError(
            f"Expected 11 Guangzhou districts, observed {len(districts)}"
        )
    districts = districts.set_crs(ADMIN_SOURCE_CRS, allow_override=True)
    districts = districts.to_crs(ANALYSIS_CRS)
    districts = districts[["Name", "AdminCode", "geometry"]].copy()
    if set(districts["Name"]) != set(DISTRICT_ORDER):
        unknown = sorted(set(districts["Name"]) - set(DISTRICT_ORDER))
        raise RuntimeError(f"Unexpected district names: {unknown}")
    districts["district_en"] = districts["Name"]

    roads = gpd.read_file(road_shp)
    required_road_fields = {"osm_id", "fclass"}
    missing = required_road_fields - set(roads.columns)
    if missing:
        raise RuntimeError(f"Road network is missing fields: {missing}")
    roads = roads.set_crs(ROAD_SOURCE_CRS, allow_override=True)
    roads = roads.loc[roads["fclass"].isin(DRIVABLE_ROAD_CLASSES)].copy()
    roads = roads.to_crs(ANALYSIS_CRS)
    city_geometry = districts.geometry.union_all()
    roads = gpd.clip(roads, city_geometry, keep_geom_type=True)
    roads = roads.explode(index_parts=False, ignore_index=True)
    roads = roads.loc[~roads.geometry.is_empty & roads.geometry.notna()].copy()
    roads["road_group"] = roads["fclass"].map(road_group)
    roads["ROAD_SEGMENT_ID"] = np.arange(len(roads), dtype=np.int64)
    roads["road_length_m"] = roads.geometry.length
    roads = roads.loc[roads["road_length_m"] > 0].copy()
    roads = roads[
        [
            "ROAD_SEGMENT_ID",
            "osm_id",
            "fclass",
            "road_group",
            "road_length_m",
            "geometry",
        ]
    ]
    return districts, roads, admin_shp


def create_master_grid(
    districts: gpd.GeoDataFrame,
    grid_size_m: int,
) -> gpd.GeoDataFrame:
    """Create one fixed grid and assign edge cells by largest district overlap."""
    xmin, ymin, xmax, ymax = districts.total_bounds
    x0 = math.floor(xmin / grid_size_m) * grid_size_m
    y0 = math.floor(ymin / grid_size_m) * grid_size_m
    x1 = math.ceil(xmax / grid_size_m) * grid_size_m
    y1 = math.ceil(ymax / grid_size_m) * grid_size_m

    records = []
    for x in np.arange(x0, x1, grid_size_m, dtype=float):
        for y in np.arange(y0, y1, grid_size_m, dtype=float):
            ix = int(math.floor(x / grid_size_m))
            iy = int(math.floor(y / grid_size_m))
            records.append(
                {
                    "grid_ix": ix,
                    "grid_iy": iy,
                    "GRID_ID": f"G{grid_size_m}_{ix}_{iy}",
                    "geometry": geometry_box(x, y, x + grid_size_m, y + grid_size_m),
                }
            )

    full_grid = gpd.GeoDataFrame(records, crs=ANALYSIS_CRS)
    intersections = gpd.overlay(
        full_grid,
        districts[["Name", "AdminCode", "district_en", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    intersections["overlap_area_m2"] = intersections.geometry.area
    assignments = (
        intersections.sort_values("overlap_area_m2", ascending=False)
        .drop_duplicates("GRID_ID")
        [
            [
                "GRID_ID",
                "Name",
                "AdminCode",
                "district_en",
                "overlap_area_m2",
            ]
        ]
    )
    grid = full_grid.merge(assignments, on="GRID_ID", how="inner")
    grid["grid_size_m"] = grid_size_m
    grid["city_overlap_fraction"] = (
        grid["overlap_area_m2"] / float(grid_size_m**2)
    )
    return gpd.GeoDataFrame(grid, geometry="geometry", crs=ANALYSIS_CRS)


def register_city_boundary(
    con: duckdb.DuckDBPyConnection,
    admin_shp: Path,
) -> None:
    """Register the unioned Guangzhou boundary in the analysis CRS."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE city_boundary AS
        SELECT ST_Union_Agg(
            ST_Transform(
                geom,
                '{ADMIN_SOURCE_CRS}',
                '{ANALYSIS_CRS}',
                always_xy := true
            )
        ) AS geometry
        FROM ST_Read('{sql_path(admin_shp)}')
        """
    )


# ============================================================
# 4. Monthly point projection and grid aggregation
# ============================================================

def projected_month_path(month: str) -> Path:
    return TEMP_DIR / f"projected_valid_gps_{month}.parquet"


def materialize_projected_points(
    con: duckdb.DuckDBPyConnection,
    month: str,
    release_path: Path,
) -> tuple[int, int, Path]:
    """Project GPS_QC=0 points and retain only points inside Guangzhou."""
    source = csv_source(release_path)
    valid_gps_count = con.execute(
        f"SELECT COUNT(*) FROM {source} WHERE GPS_QC = 0"
    ).fetchone()[0]

    output = projected_month_path(month)
    if output.exists():
        output.unlink()

    con.execute(
        f"""
        COPY (
            WITH projected AS (
                SELECT
                    DEVICE_ID,
                    CAST(TIME_POINT AS DATE) AS observation_date,
                    PM25_QC,
                    PM10_QC,
                    ST_Transform(
                        ST_Point(LONGITUDE, LATITUDE),
                        '{ROAD_SOURCE_CRS}',
                        '{ANALYSIS_CRS}',
                        always_xy := true
                    ) AS point_geometry
                FROM {source}
                WHERE GPS_QC = 0
                  AND LONGITUDE IS NOT NULL
                  AND LATITUDE IS NOT NULL
            )
            SELECT
                DEVICE_ID,
                observation_date,
                PM25_QC,
                PM10_QC,
                ST_X(point_geometry) AS x,
                ST_Y(point_geometry) AS y
            FROM projected
            CROSS JOIN city_boundary
            WHERE ST_Within(point_geometry, city_boundary.geometry)
        )
        TO '{sql_path(output)}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    inside_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{sql_path(output)}')"
    ).fetchone()[0]
    return int(valid_gps_count), int(inside_count), output


def aggregate_grid_metrics(
    con: duckdb.DuckDBPyConnection,
    month: str,
    projected_path: Path,
    grid_size_m: int,
) -> pd.DataFrame:
    """Aggregate records, valid PM, dates and devices by fixed spatial grid."""
    result = con.execute(
        f"""
        SELECT
            FLOOR(x / {grid_size_m})::BIGINT AS grid_ix,
            FLOOR(y / {grid_size_m})::BIGINT AS grid_iy,
            COUNT(*)::BIGINT AS record_count,
            COUNT(*) FILTER (
                WHERE PM25_QC = 0 AND PM10_QC = 0
            )::BIGINT AS valid_pm_record_count,
            COUNT(DISTINCT observation_date)::INTEGER AS observation_days,
            COUNT(DISTINCT DEVICE_ID)::INTEGER AS device_count
        FROM read_parquet('{sql_path(projected_path)}')
        GROUP BY grid_ix, grid_iy
        ORDER BY grid_ix, grid_iy
        """
    ).df()
    result.insert(0, "month", month)
    result["GRID_ID"] = (
        "G"
        + str(grid_size_m)
        + "_"
        + result["grid_ix"].astype(str)
        + "_"
        + result["grid_iy"].astype(str)
    )
    return result


def road_match_sample_points(
    con: duckdb.DuckDBPyConnection,
    projected_path: Path,
) -> pd.DataFrame:
    """Represent occupied 25 m cells by their mean observed coordinate."""
    return con.execute(
        f"""
        SELECT
            FLOOR(x / {ROAD_MATCH_SAMPLE_CELL_M})::BIGINT AS sample_ix,
            FLOOR(y / {ROAD_MATCH_SAMPLE_CELL_M})::BIGINT AS sample_iy,
            AVG(x) AS x,
            AVG(y) AS y,
            COUNT(*)::BIGINT AS represented_record_count
        FROM read_parquet('{sql_path(projected_path)}')
        GROUP BY sample_ix, sample_iy
        ORDER BY sample_ix, sample_iy
        """
    ).df()


def match_points_to_nearest_roads(
    points: pd.DataFrame,
    roads: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Return minimum nearest-point distance for every matched road segment."""
    minimum_distance: dict[int, float] = {}
    road_lookup = roads[["ROAD_SEGMENT_ID", "geometry"]]

    for start in range(0, len(points), ROAD_MATCH_CHUNK_SIZE):
        chunk = points.iloc[start : start + ROAD_MATCH_CHUNK_SIZE].copy()
        chunk["sample_id"] = np.arange(start, start + len(chunk), dtype=np.int64)
        point_gdf = gpd.GeoDataFrame(
            chunk[["sample_id"]],
            geometry=gpd.points_from_xy(chunk["x"], chunk["y"]),
            crs=ANALYSIS_CRS,
        )
        joined = gpd.sjoin_nearest(
            point_gdf,
            road_lookup,
            how="left",
            max_distance=max(ROAD_MATCH_SENSITIVITY_M),
            distance_col="match_distance_m",
        )
        joined = joined.dropna(subset=["ROAD_SEGMENT_ID", "match_distance_m"])
        if joined.empty:
            continue
        joined = (
            joined.sort_values("match_distance_m")
            .drop_duplicates("sample_id")
        )
        per_road = joined.groupby("ROAD_SEGMENT_ID")["match_distance_m"].min()
        for segment_id, distance in per_road.items():
            segment_id = int(segment_id)
            distance = float(distance)
            previous = minimum_distance.get(segment_id)
            if previous is None or distance < previous:
                minimum_distance[segment_id] = distance

    return pd.DataFrame(
        {
            "ROAD_SEGMENT_ID": list(minimum_distance.keys()),
            "minimum_match_distance_m": list(minimum_distance.values()),
        }
    )


def attach_metrics_to_grid(
    master_grid: gpd.GeoDataFrame,
    metrics: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Join one month of observed metrics to the master grid."""
    joined = master_grid.merge(metrics, on=["GRID_ID", "grid_ix", "grid_iy"], how="left")
    numeric = [
        "record_count",
        "valid_pm_record_count",
        "observation_days",
        "device_count",
    ]
    joined[numeric] = joined[numeric].fillna(0)
    for column in numeric:
        joined[column] = joined[column].astype(np.int64)
    joined["observed"] = joined["record_count"] > 0
    return gpd.GeoDataFrame(joined, geometry="geometry", crs=ANALYSIS_CRS)


# ============================================================
# 5. District and road summaries
# ============================================================

def district_grid_summary(
    grid_layers: dict[str, gpd.GeoDataFrame],
) -> pd.DataFrame:
    """Calculate observed-grid counts and proportions for 11 districts."""
    rows = []
    for month, grid in grid_layers.items():
        grouped = grid.groupby(["Name", "district_en"], observed=True)
        summary = grouped.agg(
            total_grid_count=("GRID_ID", "nunique"),
            observed_grid_count=("observed", "sum"),
            record_count=("record_count", "sum"),
            valid_pm_record_count=("valid_pm_record_count", "sum"),
        ).reset_index()
        summary.insert(0, "month", month)
        summary["grid_coverage_percent"] = (
            100.0
            * summary["observed_grid_count"]
            / summary["total_grid_count"]
        )
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def road_coverage_summaries(
    roads: gpd.GeoDataFrame,
    road_matches: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    """Summarize monthly, cumulative and threshold-sensitive road coverage."""
    total_by_group = roads.groupby("road_group")["road_length_m"].sum()
    total_all = float(roads["road_length_m"].sum())
    class_rows = []
    sensitivity_rows = []
    class_sensitivity_rows = []

    main_sets: dict[str, set[int]] = {}
    for month, matches in road_matches.items():
        main_sets[month] = set(
            matches.loc[
                matches["minimum_match_distance_m"]
                <= MAIN_ROAD_MATCH_DISTANCE_M,
                "ROAD_SEGMENT_ID",
            ].astype(int)
        )

    scopes: dict[str, set[int]] = dict(main_sets)
    scopes["All months"] = set().union(*main_sets.values())

    for scope, matched_ids in scopes.items():
        matched = roads.loc[roads["ROAD_SEGMENT_ID"].isin(matched_ids)]
        covered_by_group = matched.groupby("road_group")["road_length_m"].sum()
        for group in ROAD_GROUP_ORDER:
            total_length = float(total_by_group.get(group, 0.0))
            covered_length = float(covered_by_group.get(group, 0.0))
            class_rows.append(
                {
                    "scope": scope,
                    "road_group": group,
                    "total_road_length_km": total_length / 1000.0,
                    "covered_road_length_km": covered_length / 1000.0,
                    "road_length_coverage_percent": (
                        100.0 * covered_length / total_length
                        if total_length > 0
                        else np.nan
                    ),
                    "match_distance_m": MAIN_ROAD_MATCH_DISTANCE_M,
                }
            )

    for threshold in ROAD_MATCH_SENSITIVITY_M:
        threshold_sets = {
            month: set(
                matches.loc[
                    matches["minimum_match_distance_m"] <= threshold,
                    "ROAD_SEGMENT_ID",
                ].astype(int)
            )
            for month, matches in road_matches.items()
        }
        threshold_sets["All months"] = set().union(*threshold_sets.values())
        for scope, matched_ids in threshold_sets.items():
            threshold_matched = roads.loc[
                roads["ROAD_SEGMENT_ID"].isin(matched_ids)
            ]
            covered_length = float(threshold_matched["road_length_m"].sum())
            sensitivity_rows.append(
                {
                    "scope": scope,
                    "match_distance_m": threshold,
                    "total_road_length_km": total_all / 1000.0,
                    "covered_road_length_km": covered_length / 1000.0,
                    "road_length_coverage_percent": (
                        100.0 * covered_length / total_all
                    ),
                }
            )
            threshold_by_group = threshold_matched.groupby("road_group")[
                "road_length_m"
            ].sum()
            for group in ROAD_GROUP_ORDER:
                group_total = float(total_by_group.get(group, 0.0))
                group_covered = float(threshold_by_group.get(group, 0.0))
                class_sensitivity_rows.append(
                    {
                        "scope": scope,
                        "road_group": group,
                        "match_distance_m": threshold,
                        "total_road_length_km": group_total / 1000.0,
                        "covered_road_length_km": group_covered / 1000.0,
                        "road_length_coverage_percent": (
                            100.0 * group_covered / group_total
                            if group_total > 0
                            else np.nan
                        ),
                    }
                )

    road_output = roads.copy()
    for month, matched_ids in main_sets.items():
        column = "observed_" + month.replace("-", "_")
        road_output[column] = road_output["ROAD_SEGMENT_ID"].isin(matched_ids)
    observed_columns = [
        column for column in road_output.columns if column.startswith("observed_")
    ]
    road_output["observed_month_count"] = road_output[observed_columns].sum(axis=1)

    return (
        pd.DataFrame(class_rows),
        pd.DataFrame(sensitivity_rows),
        pd.DataFrame(class_sensitivity_rows),
        road_output,
    )


def monthly_spatial_summary(
    input_counts: dict[str, tuple[int, int]],
    grid_500: dict[str, gpd.GeoDataFrame],
    grid_1000: dict[str, gpd.GeoDataFrame],
    road_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """Build the compact manuscript-level monthly spatial summary."""
    rows = []
    for month in MONTH_FILES:
        valid_gps_count, inside_count = input_counts[month]
        observed_500 = grid_500[month].loc[grid_500[month]["observed"]]
        observed_1000 = grid_1000[month].loc[grid_1000[month]["observed"]]
        road_row = road_sensitivity.loc[
            (road_sensitivity["scope"] == month)
            & (
                road_sensitivity["match_distance_m"]
                == MAIN_ROAD_MATCH_DISTANCE_M
            )
        ].iloc[0]
        rows.append(
            {
                "month": month,
                "gps_qc_0_record_count": valid_gps_count,
                "inside_guangzhou_record_count": inside_count,
                "inside_guangzhou_percent_of_gps_qc_0": (
                    100.0 * inside_count / valid_gps_count
                    if valid_gps_count > 0
                    else np.nan
                ),
                "observed_500m_grid_count": len(observed_500),
                "observed_1000m_grid_count": len(observed_1000),
                "median_observation_days_per_observed_500m_grid": float(
                    observed_500["observation_days"].median()
                ),
                "median_devices_per_observed_500m_grid": float(
                    observed_500["device_count"].median()
                ),
                "valid_pm_records_inside_guangzhou": int(
                    observed_500["valid_pm_record_count"].sum()
                ),
                "covered_road_length_km_50m": float(
                    road_row["covered_road_length_km"]
                ),
                "road_length_coverage_percent_50m": float(
                    road_row["road_length_coverage_percent"]
                ),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# 6. Figure styling and drawing
# ============================================================

def set_publication_style() -> None:
    if FONT_DIR.is_dir():
        for path in sorted(FONT_DIR.iterdir()):
            if path.suffix.lower() in {".ttf", ".otf"}:
                font_manager.fontManager.addfont(str(path))
    family = None
    for candidate in ("Arial", "Helvetica", "DejaVu Sans"):
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=candidate),
                fallback_to_default=False,
            )
        except ValueError:
            continue
        family = candidate
        break
    if family is None:
        raise RuntimeError("No usable sans-serif font was found.")
    print(f"Font: {family}")
    if family != "Arial":
        print(
            "Arial was not found. To use it, place the Arial font files in "
            f"{FONT_DIR} and rerun the script."
        )
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [family],
        "font.size": TEXT_SIZE, "font.weight": "normal", "font.style": "normal",
        "axes.titlesize": TEXT_SIZE, "axes.titleweight": "normal",
        "axes.labelsize": TEXT_SIZE, "axes.labelweight": "normal",
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE, "legend.title_fontsize": LEGEND_SIZE,
        "axes.linewidth": 0.8, "axes.unicode_minus": True,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "text.usetex": False, "mathtext.fontset": "custom",
        "mathtext.rm": family, "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold", "mathtext.sf": family,
        "mathtext.cal": family, "mathtext.tt": family,
        "mathtext.default": "regular",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.constrained_layout.use": False, "figure.autolayout": False,
    })


def add_panel_label(ax: plt.Axes, label: str) -> None:
    # Preserve the manuscript panel order and place labels above the axes.
    ax.annotate(
        label, xy=(0, 1), xycoords="axes fraction", xytext=(0, 7),
        textcoords="offset points", ha="left", va="bottom",
        fontsize=TEXT_SIZE, fontweight="bold", fontstyle="normal",
        annotation_clip=False, zorder=40,
    )


def set_map_extent(ax: plt.Axes, bounds, padding_fraction: float = 0.025) -> None:
    xmin, ymin, xmax, ymax = bounds
    xpad = (xmax - xmin) * padding_fraction
    ypad = (ymax - ymin) * padding_fraction
    ax.set_xlim(xmin - xpad - MAP_EXTRA_LEFT_PADDING * (xmax - xmin), xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_aspect("equal")
    ax.set_axis_off()


def add_north_arrow_and_scale(
    ax: plt.Axes, scale_km: int = 20,
    north_position: tuple[float, float] = NORTH_POSITION,
    scale_side: str = "left",
    map_geometry=None,
    extent_axes=None,
) -> None:
    """Add a collision-aware north arrow and a two-segment scale bar."""
    # Draw one outlined silhouette so the two filled halves share a clean tip.
    width, height, notch = np.array([22.0, 30.0, 5.0]) * NORTH_SCALE
    needle = DrawingArea(width + 2, height + 2, 0, 0)
    tip = (1 + width / 2, 1 + height)
    left, right = (1, 1), (1 + width, 1)
    centre = (1 + width / 2, 1 + notch)
    silhouette = [tip, left, centre, right]
    needle.add_artist(MplPolygon(silhouette, closed=True, facecolor="white",
                                 edgecolor="none", linewidth=0))
    needle.add_artist(MplPolygon([tip, left, centre], closed=True, facecolor="black",
                                 edgecolor="none", linewidth=0))
    needle.add_artist(MplPolygon(silhouette, closed=True, facecolor="none",
                                 edgecolor="black", linewidth=0.65, joinstyle="round"))
    symbol = VPacker(
        children=[TextArea("N", textprops={
            "fontsize": TICK_SIZE, "fontweight": "normal", "color": "#252525",
        }), needle], align="center", pad=0, sep=3,
    )
    compass = AnnotationBbox(
        symbol, north_position, xycoords="axes fraction", box_alignment=(0.5, 1),
        frameon=False, pad=0, annotation_clip=False, zorder=31,
    )
    ax.add_artist(compass)
    ax.apply_aspect()
    renderer = ax.figure.canvas.get_renderer()
    pixels_per_point = ax.figure.dpi / 72.0
    clearance = NORTH_MAP_GAP_PT * pixels_per_point
    edge_clearance = 3 * pixels_per_point

    def try_position(position):
        compass.xy = position
        compass.xybox = position
        compass.update_positions(renderer)
        symbol_box = compass.get_window_extent(renderer)
        bounds = symbol_box.padded(edge_clearance)
        area = ax.get_window_extent(renderer)
        inside = (bounds.x0 >= area.x0 and bounds.x1 <= area.x1
                  and bounds.y0 >= area.y0 and bounds.y1 <= area.y1)
        if not inside:
            return False
        corners = ax.transData.inverted().transform(symbol_box.padded(clearance).get_points())
        return map_geometry is None or not map_geometry.intersects(
            geometry_box(*corners.ravel())
        )

    # Measure the rendered symbol, then search the upper-left map whitespace.
    compass.update_positions(renderer)
    symbol_bounds = compass.get_window_extent(renderer)
    area = ax.get_window_extent(renderer)
    inset = edge_clearance + 2 * pixels_per_point
    x_min = (symbol_bounds.width / 2 + inset) / area.width
    x_max = min(0.42, 1 - x_min)
    y_min = max(0.68, (symbol_bounds.height + inset) / area.height)
    y_max = 1 - inset / area.height
    found = try_position(north_position)
    if not found and x_min <= x_max and y_min <= y_max:
        # Use a maximum 3-point search interval so layout changes remain robust.
        nx = max(2, int(np.ceil((x_max - x_min) * area.width / (3 * pixels_per_point))) + 1)
        ny = max(2, int(np.ceil((y_max - y_min) * area.height / (3 * pixels_per_point))) + 1)
        candidates = [(x, y) for x in np.linspace(x_min, x_max, nx)
                      for y in np.linspace(y_min, y_max, ny)]
        candidates.sort(key=lambda p: ((p[0] - north_position[0]) * area.width) ** 2
                        + ((p[1] - north_position[1]) * area.height) ** 2)
        found = any(try_position(position) for position in candidates)

    if not found:
        # If needed, add a left margin to all peer maps without moving geometry.
        peers = [ax] if extent_axes is None else list(np.asarray(extent_axes, dtype=object).ravel())
        if ax not in peers:
            peers.append(ax)
        xmin = min(peer.get_xlim()[0] for peer in peers)
        xmax = max(peer.get_xlim()[1] for peer in peers)
        if map_geometry is not None and not map_geometry.is_empty:
            xmin = min(xmin, map_geometry.bounds[0])
            xmax = max(xmax, map_geometry.bounds[2])
        strip_width = symbol_bounds.width + inset + clearance + 4 * pixels_per_point
        extra = 0.0
        for peer in peers:
            slot = peer.get_position(original=True).transformed(ax.figure.transFigure)
            # Expand only unusually narrow custom canvases.
            if slot.width <= strip_width + 12 * pixels_per_point:
                growth = (strip_width + 24 * pixels_per_point) / slot.width
                ax.figure.set_figwidth(ax.figure.get_figwidth() * growth)
                slot = peer.get_position(original=True).transformed(ax.figure.transFigure)
            yrange = abs(peer.get_ylim()[1] - peer.get_ylim()[0])
            extra = max(extra, strip_width * yrange / slot.height,
                        strip_width * (xmax - xmin) / (slot.width - strip_width))
        for peer in peers:
            peer.set_xlim(xmin - 1.03 * extra, xmax)
            peer.apply_aspect()
        area = ax.get_window_extent(renderer)
        position = ((symbol_bounds.width / 2 + inset) / area.width,
                    1 - inset / area.height)
        compass.xy = compass.xybox = position
        compass.update_positions(renderer)
        print("Added map margin to preserve north-arrow size and proportions.")
    # Reserve the compass area for district-label placement.
    ax._north_compass = compass
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    width, height = xmax - xmin, ymax - ymin
    length = scale_km * 1000.0
    x0 = xmin + 0.055 * width if scale_side == "left" else xmax - 0.055 * width - length
    y0 = ymin + 0.05 * height
    # The scale-bar length uses projected metres.
    half = length / 2.0
    bar_height = 0.008 * height
    for start, facecolor in ((x0, "#24292B"), (x0 + half, "white")):
        ax.add_patch(Rectangle(
            (start, y0), half, bar_height,
            facecolor=facecolor, edgecolor="#24292B", linewidth=0.55, zorder=30,
        ))
    label_y = y0 - 1.35 * bar_height
    for position, label, alignment in (
        (x0, "0", "right"),
        (x0 + half, f"{scale_km / 2:g}", "center"),
        (x0 + length, f"{scale_km:g} km", "left"),
    ):
        ax.text(
            position, label_y, label, ha=alignment, va="top",
            fontsize=SCALE_TEXT_SIZE, fontweight="normal", fontstyle="normal",
            color="#252525", zorder=31,
        )


def style_colorbar(cbar, label: str, logarithmic: bool = False) -> None:
    cbar.set_label(label, fontsize=TEXT_SIZE, fontweight="normal", labelpad=7)
    cbar.ax.tick_params(labelsize=TICK_SIZE, length=3, width=0.8, pad=3)
    cbar.outline.set_linewidth(0.7)
    if logarithmic:
        cbar.locator = LogLocator(base=10, subs=(1,), numticks=10)
        cbar.formatter = LogFormatterMathtext(base=10)
        cbar.update_ticks()
    # Remove dense minor ticks without changing the normalization range.
    cbar.ax.yaxis.set_minor_locator(NullLocator())


def save_figure(fig: plt.Figure, stem_name: str) -> None:
    """Save publication figures using the stable manuscript filenames."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / stem_name
    try:
        for suffix in (".png", ".pdf"):
            path = stem.with_suffix(suffix)
            with BytesIO() as buffer:
                fig.savefig(
                    buffer, format=suffix[1:], dpi=DPI,
                    bbox_inches="tight", pad_inches=0.05,
                )
                content = buffer.getvalue()
            if suffix == ".pdf" and not content.rstrip().endswith(b"%%EOF"):
                raise RuntimeError(f"Incomplete PDF render; output was not written: {path}")
            path.write_bytes(content)
            print(f"Saved: {path}")
    finally:
        plt.close(fig)


def annotate_heatmap(ax, heat: pd.DataFrame, cutoff: float) -> None:
    for row in range(heat.shape[0]):
        for column in range(heat.shape[1]):
            value = float(heat.iloc[row, column])
            ax.text(
                column, row, f"{value:.1f}" if np.isfinite(value) else "–",
                ha="center", va="center", fontsize=CELL_TEXT_SIZE,
                fontweight="normal", color="white" if value >= cutoff else "#202020",
            )


# ============================================================
# 6.1 Grid-density and persistence figures
# ============================================================

def draw_grid_map(ax, districts, roads, layer, column, cmap, norm) -> None:
    districts.plot(ax=ax, facecolor="#FAFAFA", edgecolor="none", zorder=0)
    roads.plot(ax=ax, color="#D9D9D9", linewidth=0.08, alpha=0.52,
               rasterized=True, zorder=1)
    observed = layer.loc[layer["observed"].astype(bool)]
    if not observed.empty:
        observed.plot(ax=ax, column=column, cmap=cmap, norm=norm,
                      edgecolor="none", rasterized=True, zorder=2)
    districts.boundary.plot(ax=ax, color="#2F2F2F", linewidth=0.45, zorder=4)
    set_map_extent(ax, districts.total_bounds)


def plot_density_and_persistence(districts, roads, grid_layers) -> None:
    max_records = max(int(layer["record_count"].max()) for layer in grid_layers.values())
    max_days = max(int(layer["observation_days"].max()) for layer in grid_layers.values())
    record_norm = LogNorm(vmin=1, vmax=max_records)
    day_norm = Normalize(vmin=1, vmax=max_days)
    density_cmap, persistence_cmap = mpl.colormaps["viridis"], mpl.colormaps["YlGnBu"]
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 7.35))
    fig.subplots_adjust(left=0.015, right=0.865, bottom=0.035, top=0.935,
                        wspace=0.10, hspace=0.13)
    for column, month in enumerate(MONTH_LABELS):
        for row in range(2):
            ax = axes[row, column]
            draw_grid_map(
                ax, districts, roads, grid_layers[month],
                "record_count" if row == 0 else "observation_days",
                density_cmap if row == 0 else persistence_cmap,
                record_norm if row == 0 else day_norm,
            )
            # Preserve the manuscript order: a/c/e above b/d/f.
            add_panel_label(ax, chr(ord("a") + column * 2 + row))
            if row == 0:
                ax.set_title(MONTH_LABELS[month], y=1.0, pad=10, fontweight="normal")
    add_north_arrow_and_scale(
        axes[0, 0], map_geometry=districts.geometry.union_all(), extent_axes=axes,
    )
    for row, norm, cmap, label in (
        (0, record_norm, density_cmap, "Records per 500 m grid cell"),
        (1, day_norm, persistence_cmap, "Observation days\nper 500 m grid cell"),
    ):
        position = axes[row, -1].get_position()
        cax = fig.add_axes([0.885, position.y0, 0.018, position.height])
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        style_colorbar(cbar, label, logarithmic=(row == 0))
    save_figure(fig, "Figure_spatial_density_and_persistence")


def plot_appendix_1km_density(districts, roads, grid_layers) -> None:
    maximum = max(int(layer["record_count"].max()) for layer in grid_layers.values())
    norm = LogNorm(vmin=1, vmax=maximum)
    cmap = mpl.colormaps["viridis"]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.95))
    fig.subplots_adjust(left=0.015, right=0.865, bottom=0.045, top=0.89, wspace=0.10)
    for column, (ax, month) in enumerate(zip(axes, MONTH_LABELS)):
        draw_grid_map(ax, districts, roads, grid_layers[month], "record_count", cmap, norm)
        ax.set_title(MONTH_LABELS[month], y=1.0, pad=10, fontweight="normal")
        add_panel_label(ax, chr(ord("a") + column))
    add_north_arrow_and_scale(
        axes[0], map_geometry=districts.geometry.union_all(), extent_axes=axes,
    )
    position = axes[-1].get_position()
    cax = fig.add_axes([0.885, position.y0, 0.018, position.height])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    style_colorbar(cbar, "Records per 1 km grid cell", logarithmic=True)
    save_figure(fig, "Figure_1km_sampling_density")


# ============================================================
# 6.2 District and road coverage figure
# ============================================================

def arrange_district_labels(fig, ax, labels) -> None:
    """Resolve label collisions in display space without moving map geometry."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    points_per_pixel = 72.0 / fig.dpi
    padding_pixels = 2.5 * fig.dpi / 72.0

    def text_box(label):
        label.update_positions(renderer)
        return Text.get_window_extent(label, renderer).padded(padding_pixels)

    for _ in range(100):
        changed = False
        for i, first in enumerate(labels):
            for second in labels[i + 1:]:
                a, b = text_box(first), text_box(second)
                if not a.overlaps(b):
                    continue
                overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
                shift = (overlap / 2 + 0.6) * points_per_pixel
                sign = 1 if (a.y0 + a.y1) >= (b.y0 + b.y1) else -1
                x, y = first.get_position()
                first.set_position((x, y + sign * shift))
                x, y = second.get_position()
                second.set_position((x, y - sign * shift))
                changed = True
        # Keep labels inside the map panel and outside the compass area.
        for label in labels:
            bounds = text_box(label)
            compass = getattr(ax, "_north_compass", None)
            if compass is not None:
                compass.update_positions(renderer)
                reserved = compass.get_window_extent(renderer).padded(padding_pixels)
                if bounds.overlaps(reserved):
                    x, y = label.get_position()
                    label.set_position((x, y - (bounds.y1 - reserved.y0 + 1) * points_per_pixel))
                    bounds = text_box(label)
                    changed = True
            area = ax.get_window_extent(renderer)
            dx = max(0, area.x0 - bounds.x0) - max(0, bounds.x1 - area.x1)
            dy = max(0, area.y0 - bounds.y0) - max(0, bounds.y1 - area.y1)
            if dx or dy:
                x, y = label.get_position()
                label.set_position((x + dx * points_per_pixel, y + dy * points_per_pixel))
                changed = True
        if not changed:
            break


def plot_district_and_road_coverage(districts, road_output, district_summary, road_class_summary):
    fig = plt.figure(figsize=(8.4, 7.3))
    ax_map = fig.add_axes([0.015, 0.13, 0.435, 0.815])
    ax_heat = fig.add_axes([0.60, 0.57, 0.29, 0.37])
    ax_dot = fig.add_axes([0.60, 0.10, 0.29, 0.292])
    cax = fig.add_axes([0.905, 0.57, 0.018, 0.37])

    road_colors = {0: "#E2E2E2", 1: "#A6CEE3", 2: "#4F91C5", 3: "#165A8D"}
    road_widths = {0: 0.06, 1: 0.14, 2: 0.22, 3: 0.30}
    for count in range(4):
        subset = road_output.loc[road_output["observed_month_count"] == count]
        if not subset.empty:
            subset.plot(ax=ax_map, color=road_colors[count], linewidth=road_widths[count],
                        alpha=0.78 if count else 0.34, rasterized=True, zorder=1 + count)
    districts.boundary.plot(ax=ax_map, color="#252525", linewidth=0.55, zorder=8)
    set_map_extent(ax_map, districts.total_bounds, padding_fraction=0.05)
    labels = []
    for _, row in districts.iterrows():
        point = row.geometry.representative_point()
        labels.append(ax_map.annotate(
            row["district_en"], xy=(point.x, point.y),
            xytext=DISTRICT_LABEL_OFFSETS_POINTS.get(row["Name"], (0, 0)),
            textcoords="offset points", ha="center", va="center",
            fontsize=MAP_LABEL_SIZE, fontweight="normal", color="#222222", zorder=20,
            path_effects=[path_effects.withStroke(linewidth=2.8, foreground="white")],
            arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.5,
                        "shrinkA": 3, "shrinkB": 2},
        ))
    add_north_arrow_and_scale(ax_map, scale_side="left", map_geometry=districts.geometry.union_all())
    add_panel_label(ax_map, "a")
    # Place the legend below the map so it does not cover roads or labels.
    fig.legend(
        handles=[Line2D([0], [0], color=road_colors[count], lw=2,
                        label=label) for count, label in enumerate(
                            ("Not observed", "1 month", "2 months", "3 months"))],
        title="Months observed", loc="lower left", bbox_to_anchor=(0.02, 0.008),
        ncol=2, frameon=False, fontsize=LEGEND_SIZE, title_fontsize=LEGEND_SIZE,
        handlelength=1.4, handletextpad=0.5, columnspacing=1.0,
    )

    heat = district_summary.pivot(index="Name", columns="month", values="grid_coverage_percent")
    heat = heat.reindex(index=DISTRICT_ORDER, columns=list(MONTH_LABELS))
    image = ax_heat.imshow(heat.to_numpy(), cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax_heat.set_xticks(range(3), ["Mar", "Aug", "Nov"])
    ax_heat.set_yticks(range(len(DISTRICT_ORDER)), DISTRICT_ORDER)
    ax_heat.tick_params(length=0, pad=5)
    annotate_heatmap(ax_heat, heat, cutoff=60)
    style_colorbar(fig.colorbar(image, cax=cax), "Observed 500 m grid cells (%)")
    add_panel_label(ax_heat, "b")

    y_base = np.arange(len(ROAD_GROUP_ORDER))
    for scope, offset, marker in zip(SCOPES, (-0.27, -0.09, 0.09, 0.27), MARKERS):
        subset = road_class_summary.loc[road_class_summary["scope"] == scope]
        subset = subset.set_index("road_group").reindex(ROAD_GROUP_ORDER)
        ax_dot.scatter(subset["road_length_coverage_percent"], y_base + offset,
                       s=28, color=MONTH_COLORS[scope], marker=marker,
                       edgecolor="white", linewidth=0.4,
                       label=MONTH_LABELS.get(scope, "All months"), zorder=4)
    ax_dot.set_yticks(y_base, ROAD_GROUP_ORDER)
    ax_dot.invert_yaxis()
    ax_dot.set_xlabel("Road-length coverage (%)", labelpad=6)
    ax_dot.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_dot.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    ax_dot.set_axisbelow(True)
    ax_dot.spines[["top", "right", "left"]].set_visible(False)
    ax_dot.tick_params(axis="y", length=0, pad=5)
    handles, texts = ax_dot.get_legend_handles_labels()
    fig.legend(handles, texts, loc="center", bbox_to_anchor=(0.745, 0.483),
               ncol=2, frameon=False, fontsize=LEGEND_SIZE, handlelength=1.0,
               handletextpad=0.35, columnspacing=0.9, labelspacing=0.5)
    add_panel_label(ax_dot, "c")
    arrange_district_labels(fig, ax_map, labels)
    save_figure(fig, "Figure_district_and_road_coverage")


# ============================================================
# 6.3 Road-matching sensitivity figure
# ============================================================

def plot_appendix_road_sensitivity(road_sensitivity, road_class_sensitivity) -> None:
    fig = plt.figure(figsize=(8.4, 3.8))
    ax_line = fig.add_axes([0.10, 0.20, 0.33, 0.68])
    ax_heat = fig.add_axes([0.65, 0.20, 0.245, 0.68])
    cax = fig.add_axes([0.915, 0.20, 0.017, 0.68])
    for scope, marker in zip(SCOPES, MARKERS):
        subset = road_sensitivity.loc[road_sensitivity["scope"] == scope]
        subset = subset.sort_values("match_distance_m")
        ax_line.plot(subset["match_distance_m"], subset["road_length_coverage_percent"],
                     color=MONTH_COLORS[scope], marker=marker, markersize=5.5,
                     linewidth=1.5, label=MONTH_LABELS.get(scope, "All months"))
    ax_line.set_xticks(ROAD_MATCH_SENSITIVITY_M)
    ax_line.set_xlabel("Nearest-road matching\ndistance (m)", labelpad=6)
    ax_line.set_ylabel("Road-length coverage (%)", labelpad=6)
    ax_line.grid(color="#DDDDDD", linewidth=0.6)
    ax_line.set_axisbelow(True)
    ax_line.spines[["top", "right"]].set_visible(False)
    # A single-column legend accommodates the larger publication text.
    ax_line.legend(frameon=False, ncol=1, loc="center", handlelength=1.8,
                   fontsize=LEGEND_SIZE, labelspacing=0.6)
    add_panel_label(ax_line, "a")
    heat = road_class_sensitivity.loc[road_class_sensitivity["scope"] == "All months"]
    heat = heat.pivot(index="road_group", columns="match_distance_m",
                      values="road_length_coverage_percent")
    heat = heat.reindex(index=ROAD_GROUP_ORDER, columns=list(ROAD_MATCH_SENSITIVITY_M))
    values = heat.to_numpy(dtype=float)
    heat_min = max(0.0, float(np.nanmin(values)) - 5.0)
    heat_max = min(100.0, float(np.nanmax(values)) + 5.0)
    image = ax_heat.imshow(values, cmap="YlGnBu", vmin=heat_min, vmax=heat_max, aspect="auto")
    ax_heat.set_xticks(range(3), [f"{n} m" for n in ROAD_MATCH_SENSITIVITY_M])
    ax_heat.set_yticks(range(len(ROAD_GROUP_ORDER)), ROAD_GROUP_ORDER)
    ax_heat.tick_params(length=0, pad=5)
    annotate_heatmap(ax_heat, heat, cutoff=(heat_min + heat_max) / 2)
    style_colorbar(fig.colorbar(image, cax=cax), "Cumulative road coverage (%)")
    add_panel_label(ax_heat, "b")
    save_figure(fig, "Figure_road_match_sensitivity")


# ============================================================
# 7. Output writing and orchestration
# ============================================================

def write_geopackage(
    layers: dict[str, gpd.GeoDataFrame],
    path: Path,
) -> None:
    """Write named GeoPackage layers, replacing one protected output file."""
    if path.exists():
        path.unlink()
    for index, (layer_name, frame) in enumerate(layers.items()):
        frame.to_file(
            path,
            layer=layer_name,
            driver="GPKG",
            mode="w" if index == 0 else "a",
        )


def read_monthly_geopackage(path: Path) -> dict[str, gpd.GeoDataFrame]:
    """Read the three monthly layers written by write_geopackage."""
    return {
        month: gpd.read_file(path, layer=month.replace("-", "_"))
        for month in MONTH_FILES
    }


def redraw_all_figures_from_saved_results(
    districts: gpd.GeoDataFrame,
) -> None:
    """Redraw all main and appendix figures without repeating analysis."""
    grid_500 = read_monthly_geopackage(GRID_500_GPKG)
    grid_1000 = read_monthly_geopackage(GRID_1000_GPKG)
    road_output = gpd.read_file(ROAD_COVERAGE_GPKG, layer="road_coverage")
    district_summary = pd.read_csv(DISTRICT_SUMMARY_CSV, encoding="utf-8-sig")
    road_class_summary = pd.read_csv(
        ROAD_CLASS_SUMMARY_CSV,
        encoding="utf-8-sig",
    )
    road_sensitivity = pd.read_csv(ROAD_SENSITIVITY_CSV, encoding="utf-8-sig")
    road_class_sensitivity = pd.read_csv(
        ROAD_CLASS_SENSITIVITY_CSV,
        encoding="utf-8-sig",
    )

    # The saved layer contains the complete clipped road network, including
    # roads that were not observed, so it is also the correct map background.
    plot_density_and_persistence(districts, road_output, grid_500)
    plot_district_and_road_coverage(
        districts,
        road_output,
        district_summary,
        road_class_summary,
    )
    plot_appendix_1km_density(districts, road_output, grid_1000)
    plot_appendix_road_sensitivity(
        road_sensitivity,
        road_class_sensitivity,
    )


def main() -> None:
    args = parse_args()
    configure_runtime(args)
    started = perf_counter()
    print("Checking input files...")
    validate_inputs_and_outputs()
    prepare_output_directories()
    set_publication_style()

    print("Preparing administrative boundary and drivable road network...")
    districts, roads, admin_shp = load_and_prepare_reference_layers()

    if REDRAW_FIGURES_ONLY:
        print("Redrawing figures from completed spatial-analysis results...")
        redraw_all_figures_from_saved_results(districts)
        elapsed = perf_counter() - started
        print(f"Figures redrawn in {elapsed / 60:.1f} minutes.")
        print(f"Figures: {FIGURE_DIR}")
        return

    master_grids = {
        size: create_master_grid(districts, size) for size in GRID_SIZES_M
    }

    con = configure_duckdb()
    register_city_boundary(con, admin_shp)

    input_counts: dict[str, tuple[int, int]] = {}
    raw_grid_metrics: dict[int, dict[str, pd.DataFrame]] = {
        size: {} for size in GRID_SIZES_M
    }
    road_matches: dict[str, pd.DataFrame] = {}

    try:
        for month, release_path in MONTH_FILES.items():
            print(f"[{month}] Projecting GPS_QC=0 records and applying city boundary...")
            valid_count, inside_count, projected_path = materialize_projected_points(
                con,
                month,
                release_path,
            )
            input_counts[month] = (valid_count, inside_count)

            for size in GRID_SIZES_M:
                print(f"[{month}] Aggregating {size} m grid metrics...")
                raw_grid_metrics[size][month] = aggregate_grid_metrics(
                    con,
                    month,
                    projected_path,
                    size,
                )

            print(f"[{month}] Reducing GPS records to road-matching cells...")
            sample_points = road_match_sample_points(con, projected_path)
            print(
                f"[{month}] Matching {len(sample_points):,} occupied cells "
                "to the nearest drivable road..."
            )
            road_matches[month] = match_points_to_nearest_roads(
                sample_points,
                roads,
            )
            del sample_points

            if not KEEP_PROJECTED_POINT_PARQUET and projected_path.exists():
                projected_path.unlink()
    finally:
        con.close()

    grid_layers_by_size: dict[int, dict[str, gpd.GeoDataFrame]] = {}
    for size in GRID_SIZES_M:
        grid_layers_by_size[size] = {
            month: attach_metrics_to_grid(
                master_grids[size],
                raw_grid_metrics[size][month],
            )
            for month in MONTH_FILES
        }

    grid_500 = grid_layers_by_size[500]
    grid_1000 = grid_layers_by_size[1000]
    district_summary = district_grid_summary(grid_500)
    (
        road_class_summary,
        road_sensitivity,
        road_class_sensitivity,
        road_output,
    ) = road_coverage_summaries(roads, road_matches)
    monthly_summary = monthly_spatial_summary(
        input_counts,
        grid_500,
        grid_1000,
        road_sensitivity,
    )

    print("Writing tables and spatial analysis layers...")
    monthly_summary.to_csv(MONTHLY_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    district_summary.to_csv(DISTRICT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    road_class_summary.to_csv(ROAD_CLASS_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    road_sensitivity.to_csv(ROAD_SENSITIVITY_CSV, index=False, encoding="utf-8-sig")
    road_class_sensitivity.to_csv(
        ROAD_CLASS_SENSITIVITY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    write_geopackage(
        {month.replace("-", "_"): frame for month, frame in grid_500.items()},
        GRID_500_GPKG,
    )
    write_geopackage(
        {month.replace("-", "_"): frame for month, frame in grid_1000.items()},
        GRID_1000_GPKG,
    )
    write_geopackage({"road_coverage": road_output}, ROAD_COVERAGE_GPKG)

    print("Drawing the two main manuscript figures...")
    plot_density_and_persistence(districts, roads, grid_500)
    plot_district_and_road_coverage(
        districts,
        road_output,
        district_summary,
        road_class_summary,
    )
    print("Drawing the two appendix figures...")
    plot_appendix_1km_density(districts, roads, grid_1000)
    plot_appendix_road_sensitivity(
        road_sensitivity,
        road_class_sensitivity,
    )
    elapsed = perf_counter() - started
    print(f"Completed in {elapsed / 60:.1f} minutes.")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Tables: {TABLE_DIR}")
    print(f"Spatial analysis data: {ANALYSIS_DATA_DIR}")


if __name__ == "__main__":
    main()
