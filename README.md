# Guangzhou mobile PM dataset: technical validation code

This repository contains the Python code used to reproduce the technical-validation analyses for the Guangzhou mobile particulate matter dataset. The scripts generate summary tables, supporting analysis files, GIS layers, and publication-ready figures.

## Scripts

| Analysis | Script |
|---|---|
| Data completeness and quality control | `1_data_completeness_qc.py` |
| Device operation time and temporal coverage | `2.1_device_operation_time_coverage.py` |
| Diurnal and weekday-weekend coverage | `2.2_diurnal_weekday_weekend_coverage.py` |
| Spatial coverage | `3_spatial_coverage.py` |
| Interdevice consistency | `4_interdevice_consistency.py` |
| Mobile-fixed-site consistency | `5_mobile_station_consistency.py` |

## Requirements

- Python 3.10 or later
- DuckDB
- pandas
- NumPy
- Matplotlib
- SciPy
- pyproj
- GeoPandas
- Shapely

Install the required packages with:

```bash
python3 -m pip install duckdb pandas numpy matplotlib scipy pyproj geopandas shapely
```

## Input files

Place the released data files in one directory:

```text
Guangzhou_mobile_2023-03_15s_QC.csv.gz
Guangzhou_mobile_2023-08_15s_QC.csv.gz
Guangzhou_mobile_2023-11_15s_QC.csv.gz
Guangzhou_station_2023-03_1h_QC.csv.gz
Guangzhou_station_2023-08_1h_QC.csv.gz
Guangzhou_station_2023-11_1h_QC.csv.gz
station_coordinates.csv
```

The three hourly fixed-site files contain observations from 20 monitoring stations. Station coordinates are provided separately in `station_coordinates.csv` and are linked by `STATION_CODE`.

The spatial-coverage analysis additionally requires:

```text
Guangzhou_district_boundaries.zip
Guangzhou_road_network.zip
```

These archives do not need to be extracted manually.

## Running the analyses

Replace `/path/to/project` and `/path/to/published_data` with local paths.

### 1. Data completeness and quality control

```bash
python3 /path/to/project/1_data_completeness_qc.py \
  --data-dir /path/to/published_data \
  --result-dir /path/to/project/results/data_completeness_qc \
  --temp-dir /path/to/project/duckdb_tmp/data_completeness_qc
```

### 2.1 Device operation time and temporal coverage

```bash
python3 /path/to/project/2.1_device_operation_time_coverage.py \
  --data-dir /path/to/published_data \
  --output-dir /path/to/project/results/device_operation_time_coverage \
  --temp-dir /path/to/project/duckdb_tmp/device_operation_time_coverage
```

### 2.2 Diurnal and weekday-weekend coverage

```bash
python3 /path/to/project/2.2_diurnal_weekday_weekend_coverage.py \
  --data-dir /path/to/published_data \
  --output-dir /path/to/project/results/diurnal_weekday_weekend_coverage \
  --temp-dir /path/to/project/duckdb_tmp/diurnal_weekday_weekend_coverage
```

### 3. Spatial coverage

```bash
python3 /path/to/project/3_spatial_coverage.py \
  --data-dir /path/to/published_data \
  --spatial-reference-dir /path/to/project \
  --output-dir /path/to/project/results/spatial_coverage \
  --temp-dir /path/to/project/duckdb_tmp/spatial_coverage
```

Use `--redraw-figures-only --overwrite` to regenerate figures from completed tables and GIS layers.

### 4. Interdevice consistency

```bash
python3 /path/to/project/4_interdevice_consistency.py \
  --data-dir /path/to/published_data \
  --output-dir /path/to/project/results/interdevice_consistency \
  --temp-dir /path/to/project/duckdb_tmp/interdevice_consistency
```

Use `--keep-temporary-matched-values` to retain intermediate matched-value files and the DuckDB database.

### 5. Mobile-fixed-site consistency

```bash
python3 /path/to/project/5_mobile_station_consistency.py \
  --release-dir /path/to/published_data \
  --output /path/to/project/results/mobile_station_consistency
```

The primary analysis uses a 500 m matching radius and evaluates 1,000 m as a sensitivity analysis. Figures can be regenerated without rescanning the mobile files:

```bash
python3 /path/to/project/5_mobile_station_consistency.py \
  --output /path/to/project/results/mobile_station_consistency \
  --plot-only
```

## Outputs

Each script writes to its own result directory, which may contain:

```text
figures/        PNG and PDF figures
tables/         Summary and sensitivity-analysis tables
analysis_data/  Supporting analysis data
cache/          Audit cache for the mobile-fixed-site analysis
```

All input files are read only. Existing outputs are protected by default. Scripts 1--4 accept `--overwrite` for intentional replacement; script 5 requires an empty output directory for a full rerun.

All timestamps are interpreted as China Standard Time (UTC+8), consistent with the dataset.
