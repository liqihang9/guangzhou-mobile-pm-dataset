#!/usr/bin/env python3
"""Hourly mobile / fixed-station proximity comparison. Run with Python >=3.10.

Dependencies: pandas>=2, numpy, scipy, matplotlib, pyproj. No xlrd/duckdb.
Defaults use the QC releases, not the raw station_reference CSVs.
No source writes, calibration, interpolation, or concentration-based pair removal.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import sqlite3
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

RELEASE = Path('data')
OUTPUT = Path('results') / 'mobile_station_consistency'
FONT_DIR = Path('fonts')
MONTHS = ('2023-03', '2023-08', '2023-11')
LABELS = {'2023-03': 'March', '2023-08': 'August', '2023-11': 'November', 'Pooled': 'Pooled'}
MONTH_COLORS = {'2023-03': '#0072B2', '2023-08': '#D55E00', '2023-11': '#943C91'}
MONTH_MARKERS = {'2023-03': 'o', '2023-08': 's', '2023-11': '^'}
PM = ('PM25', 'PM10')
PM_LABEL = {'PM25': r'PM$_{2.5}$', 'PM10': r'PM$_{10}$'}
PM_COLOR = {'PM25': '#257888', 'PM10': '#C88343'}
TEXT_SIZE = 12.5
TICK_SIZE = 11.5
NOTE_SIZE = 11.0
VERSION = 'station-comparison-1.2.0'
NS_HOUR = 3_600_000_000_000
NS_MIN = 60_000_000_000
MOBILE_COLUMNS = ['DEVICE_ID', 'TIME_POINT', 'LONGITUDE', 'LATITUDE', 'PM25', 'PM10',
                  'DEVICE_TIME_QC', 'GPS_QC', 'PM25_QC', 'PM10_QC']


@dataclass(frozen=True)
class Settings:
    radius_m: float = 500.0
    comparison_radius_m: float = 1000.0
    min_minutes: int = 6
    min_quarters: int = 3
    min_span_minutes: float = 30.0
    min_station_hours: int = 30
    min_station_days: int = 5
    chunksize: int = 300_000
    expected_stations: int = 20
    dpi: int = 600
    # Initial study support rules, NOT regulatory hourly-completeness standards.
    source_crs: str = 'EPSG:4326'
    analysis_crs: str = 'EPSG:4547'


def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig', na_rep='',
              date_format='%Y-%m-%d %H:%M:%S',
              compression={'method': 'gzip', 'mtime': 0} if path.suffix == '.gz' else None)


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def timestamp(s):
    s = s.astype('string').str.strip()
    good = s.str.fullmatch(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?', na=False)
    return pd.to_datetime(s.where(good), format='mixed', errors='coerce').astype('datetime64[ns]')


def station_code(s):
    return s.astype('string').str.strip().str.replace(r'\.0+$', '', regex=True)


def get_transformer(cfg):
    from pyproj import CRS, Transformer
    source, target = CRS(cfg.source_crs), CRS(cfg.analysis_crs)
    if source.to_epsg() != 4326 or target.to_epsg() != 4547:
        raise ValueError('This protocol requires source EPSG:4326 and target EPSG:4547.')
    return Transformer.from_crs(source, target, always_xy=True)


def load_stations(release, months, cfg, transformer):
    co = pd.read_csv(release/'station_coordinates.csv', dtype={'STATION_CODE': str})
    required_coordinates = {'STATION_CODE', 'LONGITUDE', 'LATITUDE'}
    if not required_coordinates.issubset(co.columns):
        raise ValueError('station_coordinates.csv lacks STATION_CODE, LONGITUDE or LATITUDE.')
    co.STATION_CODE = station_code(co.STATION_CODE)
    if (len(co) != cfg.expected_stations or co.STATION_CODE.duplicated().any()
            or co.STATION_CODE.isna().any() or co.STATION_CODE.eq('').any()):
        raise ValueError('Coordinate count/ID mismatch. Resolve coordinates before analysis.')
    for v in ['LONGITUDE', 'LATITUDE']:
        co[v] = pd.to_numeric(co[v], errors='coerce')
    spatial_ok = (co.LONGITUDE.between(-180, 180)
                  & co.LATITUDE.between(-90, 90) & co.LONGITUDE.ne(0) & co.LATITUDE.ne(0))
    co['spatial_eligible'] = spatial_ok
    co['X_4547'], co['Y_4547'] = np.nan, np.nan
    x, y = transformer.transform(co.loc[spatial_ok, 'LONGITUDE'].to_numpy(),
                                 co.loc[spatial_ok, 'LATITUDE'].to_numpy())
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('Fixed-station projection produced nonfinite coordinates.')
    co.loc[spatial_ok, 'X_4547'], co.loc[spatial_ok, 'Y_4547'] = x, y
    refs, audits = [], []
    for month in months:
        d = pd.read_csv(release/f'Guangzhou_station_{month}_1h_QC.csv.gz',
                        dtype={'STATION_CODE': str})
        required = {'STATION_CODE', 'TIME_POINT', 'STATION_TIME_QC',
                    'PM25', 'PM10', 'PM25_QC', 'PM10_QC'}
        if not required.issubset(d):
            raise ValueError(f'{month}: station file lacks QC release columns.')
        d.STATION_CODE = station_code(d.STATION_CODE)
        d.TIME_POINT = timestamp(d.TIME_POINT)
        d['month'] = month
        base = (pd.to_numeric(d.STATION_TIME_QC, errors='coerce').eq(0)
                & d.STATION_CODE.isin(co.loc[spatial_ok, 'STATION_CODE'])
                & d.TIME_POINT.notna()
                & d.TIME_POINT.eq(d.TIME_POINT.dt.floor('h'))
                & d.TIME_POINT.dt.strftime('%Y-%m').eq(month))
        if d.loc[base].duplicated(['STATION_CODE', 'TIME_POINT']).any():
            raise ValueError('Duplicate QC0 reference station-hours: rerun station QC first.')
        for p in PM:
            val = pd.to_numeric(d[p], errors='coerce')
            good = base & pd.to_numeric(d[p+'_QC'], errors='coerce').eq(0) & np.isfinite(val) & val.ge(0)
            d[p+'_valid'] = good
            d[p+'_reference'] = val.where(good)
        audits.append(d[['month', 'STATION_CODE', 'TIME_POINT', 'PM25_valid', 'PM10_valid']])
        refs.append(d.loc[d.PM25_valid | d.PM10_valid,
                         ['month', 'STATION_CODE', 'TIME_POINT', 'PM25_reference', 'PM10_reference']])
    ref = pd.concat(refs, ignore_index=True)
    if ref.empty:
        raise ValueError('No eligible fixed-station observations; inspect station QC reports.')
    ref['hour_ns'] = ref.TIME_POINT.astype('int64')
    return co, ref, pd.concat(audits, ignore_index=True)


def init_db(path):
    con = sqlite3.connect(path)
    con.execute('PRAGMA synchronous=NORMAL')
    con.execute('PRAGMA temp_store=FILE')
    con.execute('PRAGMA cache_size=-131072')
    con.execute('CREATE TABLE nearby (station TEXT, hour_ns INTEGER, device TEXT, '
                'time_ns INTEGER, pm25 REAL, pm10 REAL, distance_m REAL)')
    return con


def ingest_mobile(con, release, co, ref, months, cfg, transformer):
    """Chunked gzip scan to the largest radius; all stations, not only nearest.

    Nearby rows are stored on disk. Aggregation happens AFTER the complete scan,
    so duplicate filtering and exact medians never depend on CSV chunk boundaries.
    """
    spatial = co.loc[co.spatial_eligible].reset_index(drop=True)
    tree = cKDTree(spatial[['X_4547', 'Y_4547']].to_numpy())
    station_ids = spatial.STATION_CODE.to_numpy()
    logs = []
    for month in months:
        started = time.monotonic()
        r = ref.loc[ref.month.eq(month)]
        active_hours = r.hour_ns.unique()
        lookup = {str(s): g.set_index('hour_ns')[['PM25_reference', 'PM10_reference']]
                  for s, g in r.groupby('STATION_CODE')}
        counts = dict(month=month, source_rows=0, base_qc_and_time_rows=0,
                      rows_in_any_reference_hour=0, station_row_assignments=0,
                      multiple_station_rows=0)
        path = release/f'Guangzhou_mobile_{month}_15s_QC.csv.gz'
        for chunk_no, d in enumerate(pd.read_csv(path, usecols=MOBILE_COLUMNS,
                dtype={'DEVICE_ID': 'string', 'TIME_POINT': 'string'}, chunksize=cfg.chunksize), 1):
            counts['source_rows'] += len(d)
            d['time'] = timestamp(d.TIME_POINT)
            for col in ['LONGITUDE', 'LATITUDE', 'PM25', 'PM10',
                        'DEVICE_TIME_QC', 'GPS_QC', 'PM25_QC', 'PM10_QC']:
                d[col] = pd.to_numeric(d[col], errors='coerce')
            d.DEVICE_ID = d.DEVICE_ID.str.strip()
            for p in PM:
                ok = d[p+'_QC'].eq(0) & np.isfinite(d[p]) & d[p].ge(0)
                d[p] = d[p].where(ok)
            base = (d.DEVICE_TIME_QC.eq(0) & d.GPS_QC.eq(0) & d.time.notna()
                    & d.time.dt.strftime('%Y-%m').eq(month)
                    & d.DEVICE_ID.notna() & d.DEVICE_ID.ne('')
                    & d.LONGITUDE.between(-180, 180) & d.LATITUDE.between(-90, 90)
                    & d.LONGITUDE.ne(0) & d.LATITUDE.ne(0)
                    & d[list(PM)].notna().any(axis=1))
            counts['base_qc_and_time_rows'] += int(base.sum())
            d = d.loc[base].copy()
            d['hour_ns'] = d.time.dt.floor('h').astype('int64')
            d = d.loc[d.hour_ns.isin(active_hours)].reset_index(drop=True)
            counts['rows_in_any_reference_hour'] += len(d)
            if len(d):
                x, y = transformer.transform(d.LONGITUDE.to_numpy(), d.LATITUDE.to_numpy())
                xy = np.column_stack([x, y])
                if not np.isfinite(xy).all():
                    raise ValueError('Mobile projection produced nonfinite coordinates.')
                neighbors = tree.query_ball_point(xy, cfg.comparison_radius_m)
                n = np.fromiter((len(a) for a in neighbors), dtype=int, count=len(d))
                if n.sum():
                    mi = np.repeat(np.arange(len(d)), n)
                    si = np.concatenate([np.asarray(a, dtype=int) for a in neighbors if len(a)])
                    out = pd.DataFrame({'station': station_ids[si], 'hour_ns': d.hour_ns.to_numpy()[mi],
                        'device': d.DEVICE_ID.to_numpy()[mi], 'time_ns': d.time.astype('int64').to_numpy()[mi],
                        'pm25': d.PM25.to_numpy()[mi], 'pm10': d.PM10.to_numpy()[mi],
                        'distance_m': np.linalg.norm(xy[mi]-tree.data[si], axis=1), 'mobile_index': mi})
                    assignments = []
                    for sid, g in out.groupby('station', sort=False):
                        if sid not in lookup:
                            continue
                        g = g.copy()
                        for p in PM:
                            valid_ref = g.hour_ns.map(lookup[sid][p+'_reference']).notna()
                            g[p.lower()] = g[p.lower()].where(valid_ref)
                        assignments.append(g.loc[g[['pm25', 'pm10']].notna().any(axis=1)])
                    if assignments:
                        out = pd.concat(assignments, ignore_index=True)
                        counts['station_row_assignments'] += len(out)
                        counts['multiple_station_rows'] += int(out.groupby('mobile_index').size().gt(1).sum())
                        out.drop(columns='mobile_index').to_sql('nearby', con, if_exists='append', index=False)
            print(f'{month} chunk {chunk_no}: read={counts["source_rows"]:,}; '
                  f'station assignments={counts["station_row_assignments"]:,}', flush=True)
        counts['seconds'] = round(time.monotonic()-started, 1)
        logs.append(counts)
    con.execute('CREATE INDEX nearby_key ON nearby(station, hour_ns)')
    con.commit()
    return pd.DataFrame(logs)


def mobile_hour(g, pollutant, cfg):
    """Device-minute medians -> median across devices/minute -> mean over minutes.

    Missing minutes are not zero and are not imputed. This is an observed-minute
    hourly summary, NOT a full-hour regulatory mean. All duplicate device/time
    rows are excluded by caller before this function.
    """
    p = pollutant.lower()
    g = g.loc[g[p].notna()].copy()
    if g.empty:
        return dict(mobile_value=np.nan, n_records=0, n_devices=0, n_minutes=0,
                    n_quarters=0, span_minutes=0., median_distance_m=np.nan,
                    max_distance_m=np.nan, support_ok=False, support_reason='NO_MOBILE_DATA')
    g['minute'] = g.time_ns // NS_MIN
    minute_device = g.groupby(['minute', 'device'])[p].median()
    minute = minute_device.groupby(level='minute').median()
    n_minutes = len(minute)
    quarters = int(len(np.unique((minute.index.to_numpy() % 60)//15)))
    span = float((g.time_ns.max()-g.time_ns.min())/NS_MIN)
    reasons = []
    if n_minutes < cfg.min_minutes:
        reasons.append('FEW_MINUTES')
    if quarters < cfg.min_quarters:
        reasons.append('FEW_QUARTERS')
    if span < cfg.min_span_minutes:
        reasons.append('SHORT_TIME_SPAN')
    return dict(mobile_value=float(minute.mean()), n_records=len(g), n_devices=g.device.nunique(),
                n_minutes=n_minutes, n_quarters=quarters, span_minutes=span,
                median_distance_m=float(g.distance_m.median()), max_distance_m=float(g.distance_m.max()),
                support_ok=not reasons, support_reason=';'.join(reasons))


def make_pairs(con, ref, cfg, radius_m=None):
    radius_m = cfg.radius_m if radius_m is None else float(radius_m)
    rows = []
    for i, r in enumerate(ref.itertuples(index=False), 1):
        g = pd.read_sql_query('SELECT device,time_ns,pm25,pm10,distance_m FROM nearby '
                             'WHERE station=? AND hour_ns=? AND distance_m<=?', con,
                             params=(str(r.STATION_CODE), int(r.hour_ns), radius_m))
        duplicate = g.duplicated(['device', 'time_ns'], keep=False)
        g_clean = g.loc[~duplicate]
        for p in PM:
            v = getattr(r, p+'_reference')
            if not np.isfinite(v):
                continue
            rec = dict(month=r.month, STATION_CODE=str(r.STATION_CODE), TIME_POINT=r.TIME_POINT,
                       pollutant=p, reference_value=float(v), duplicate_rows_excluded=int(duplicate.sum()))
            rec.update(mobile_hour(g_clean, p, cfg))
            rec['difference'] = rec['mobile_value']-v
            rec['pair_mean'] = (rec['mobile_value']+v)/2
            rows.append(rec)
        if i % 5000 == 0:
            print(f'Aggregated {i:,}/{len(ref):,} eligible reference station-hours', flush=True)
    return pd.DataFrame(rows)


PAIR_KEY = ['month', 'STATION_CODE', 'TIME_POINT', 'pollutant']


def radius_sensitivity_summary(results, co, months, cfg):
    """Compare radii on their own accepted hours and on identical hours.

    Network medians are station-equal-weighted. Common-set rows remove changes
    in station-hour composition: both radii use the exact same accepted keys.
    """
    accepted = {radius: pairs.loc[pairs.support_ok].copy()
                for radius, pairs in results.items()}
    key_sets = {radius: set(map(tuple, data[PAIR_KEY].itertuples(index=False, name=None)))
                for radius, data in accepted.items()}
    common_keys = set.intersection(*key_sets.values()) if key_sets else set()
    template = next(iter(accepted.values()))[PAIR_KEY].iloc[0:0].copy()
    common = (pd.DataFrame(sorted(common_keys), columns=PAIR_KEY)
              if common_keys else template)
    rows = []

    def add_rows(analysis_set, radius, pairs):
        _, _, network = make_tables(pairs, co, months, cfg)
        pooled = network.loc[network.scope.eq('Pooled')]
        for p in PM:
            t = pooled.loc[pooled.pollutant.eq(p)].iloc[0]
            rows.append(dict(
                analysis_set=analysis_set,
                radius_m=float(radius),
                pollutant=p,
                matched_station_hours=int(t.paired_station_hours),
                qualified_station_hours=int(t.qualified_station_hours),
                qualified_stations=int(t.qualified_stations),
                pearson_r_station_median=t.pearson_r_median,
                mean_difference_station_median=t.mean_difference_median,
                mean_absolute_difference_station_median=t.mean_absolute_difference_median,
            ))

    for radius, pairs in results.items():
        add_rows('radius_specific', radius, pairs)
    for radius, pairs in accepted.items():
        common_pairs = pairs.merge(common, on=PAIR_KEY, how='inner', validate='one_to_one')
        add_rows('common_station_hours', radius, common_pairs)
    summary = pd.DataFrame(rows).sort_values(['analysis_set', 'pollutant', 'radius_m'])
    return summary.reset_index(drop=True), common


def metrics(g):
    x = g.reference_value.to_numpy(dtype=float)
    y = g.mobile_value.to_numpy(dtype=float)
    d = y-x
    n = len(x)
    result = dict(n_hours=n, n_days=g.TIME_POINT.dt.normalize().nunique(),
                  reference_mean=np.nan, mobile_mean=np.nan, reference_sd=np.nan,
                  pearson_r=np.nan, spearman_r=np.nan, mean_difference=np.nan,
                  median_difference=np.nan, median_absolute_difference=np.nan,
                  mean_absolute_difference=np.nan, root_mean_square_difference=np.nan,
                  difference_sd=np.nan, loa_low=np.nan, loa_high=np.nan)
    if not n:
        return result
    result.update(reference_mean=float(x.mean()), mobile_mean=float(y.mean()),
                  mean_difference=float(d.mean()), median_difference=float(np.median(d)),
                  median_absolute_difference=float(np.median(abs(d))),
                  mean_absolute_difference=float(np.mean(abs(d))),
                  root_mean_square_difference=float(np.sqrt(np.mean(d*d))))
    if n >= 2:
        sd = float(d.std(ddof=1))
        result.update(reference_sd=float(x.std(ddof=1)), difference_sd=sd,
                      loa_low=float(d.mean()-1.96*sd), loa_high=float(d.mean()+1.96*sd))
        if np.ptp(x)>0 and np.ptp(y)>0:
            result.update(pearson_r=float(np.corrcoef(x, y)[0, 1]), spearman_r=float(spearmanr(x, y).statistic))
    return result


def make_tables(pairs, co, months, cfg):
    accepted = pairs.loc[pairs.support_ok].copy()
    records = []
    for scope in [*months, 'Pooled']:
        scoped = pairs if scope == 'Pooled' else pairs.loc[pairs.month.eq(scope)]
        for sid in co.STATION_CODE:
            for p in PM:
                all_hours = scoped.loc[scoped.STATION_CODE.eq(sid) & scoped.pollutant.eq(p)]
                g = all_hours.loc[all_hours.support_ok]
                rec = dict(scope=scope, STATION_CODE=sid, pollutant=p,
                           n_reference_qc0_hours=len(all_hours),
                           n_hours_with_any_mobile=int(all_hours.n_records.gt(0).sum()))
                rec.update(metrics(g))
                rec['qualified'] = rec['n_hours']>=cfg.min_station_hours and rec['n_days']>=cfg.min_station_days
                rec['status'] = ('NO_ELIGIBLE_REFERENCE' if not len(all_hours) else
                                 'NO_QUALIFIED_HOUR' if not len(g) else
                                 'QUALIFIED' if rec['qualified'] else 'LOW_SAMPLE_SUPPORT')
                for col in ['n_minutes', 'n_devices', 'span_minutes', 'median_distance_m']:
                    rec['median_'+col] = float(g[col].median()) if len(g) else np.nan
                rec['match_percent_of_reference'] = 100*len(g)/len(all_hours) if len(all_hours) else np.nan
                records.append(rec)
    station_metrics = pd.DataFrame(records)
    network = []
    for (scope, p), t in station_metrics.groupby(['scope', 'pollutant'], sort=False):
        q = t.loc[t.qualified]
        rec = dict(scope=scope, pollutant=p, total_stations=len(t),
                   stations_with_pairs=int(t.n_hours.gt(0).sum()), qualified_stations=len(q),
                   paired_station_hours=int(t.n_hours.sum()), qualified_station_hours=int(q.n_hours.sum()))
        # Each station contributes one metric; no record-count weights.
        for col in ['pearson_r', 'spearman_r', 'mean_difference', 'mean_absolute_difference',
                    'root_mean_square_difference', 'n_hours', 'n_days']:
            v = q[col].dropna()
            rec[col+'_station_count'] = len(v)
            for quant, name in [(0.05, 'p05'), (0.25, 'q25'), (0.5, 'median'), (0.75, 'q75'), (0.95, 'p95')]:
                rec[col+'_'+name] = float(v.quantile(quant)) if len(v) else np.nan
        network.append(rec)
    return accepted, station_metrics, pd.DataFrame(network)


def select_examples(station_metrics):
    selected = []
    for p in PM:
        t = station_metrics.loc[station_metrics.scope.eq('Pooled') & station_metrics.pollutant.eq(p)
                                & station_metrics.qualified].copy()
        if t.empty:
            continue
        target = t.mean_absolute_difference.median()
        high_n = t.n_hours.quantile(0.75)
        t = t.loc[t.n_hours.ge(high_n)].copy()
        t['selection_distance'] = abs(t.mean_absolute_difference-target)
        chosen = t.sort_values(['selection_distance', 'n_hours', 'STATION_CODE'],
                               ascending=[True, False, True]).iloc[0].to_dict()
        chosen['target_network_median_MAD'] = target
        chosen['overlap_upper_quartile_threshold'] = high_n
        selected.append(chosen)
    return pd.DataFrame(selected)


def plot_results(station_metrics, network, pairs, selected, co, months, cfg, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
    from matplotlib.lines import Line2D
    from matplotlib import patheffects
    from matplotlib import font_manager
    from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullLocator, MaxNLocator
    for path in FONT_DIR.glob('*'):
        if path.suffix.lower() in {'.ttf', '.otf'}:
            font_manager.fontManager.addfont(str(path))
    family = 'DejaVu Sans'
    for candidate in ('Arial', 'Helvetica', 'DejaVu Sans'):
        try:
            font_manager.findfont(font_manager.FontProperties(family=candidate), fallback_to_default=False)
            family = candidate
            break
        except ValueError:
            continue
    print(f'Font: {family}')
    if family != 'Arial':
        print(f'Optional Arial font files can be placed in {FONT_DIR}.')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [family],
        'font.size': TEXT_SIZE, 'font.weight': 'normal', 'font.style': 'normal',
        'axes.titlesize': TEXT_SIZE, 'axes.titleweight': 'normal', 'axes.labelsize': TEXT_SIZE,
        'axes.labelweight': 'normal', 'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE,
        'legend.fontsize': TEXT_SIZE, 'ps.fonttype': 42,
        'mathtext.fontset': 'custom', 'mathtext.rm': family, 'mathtext.it': f'{family}:italic',
        'mathtext.bf': f'{family}:bold', 'mathtext.sf': family, 'mathtext.cal': family,
        'mathtext.tt': family, 'mathtext.default': 'regular', 'text.usetex': False,
        'axes.edgecolor': '#495357', 'axes.linewidth': .8, 'pdf.fonttype': 42,
        'savefig.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False,
        'figure.autolayout': False})
    teal = LinearSegmentedColormap.from_list('station_teal', ['#EDF4F3', '#99CEC5', '#35918E', '#164A5A'])
    teal.set_bad('#E9E9E9')
    # Use neutral gray for hexagon density so month colors remain distinct.
    density_gray = LinearSegmentedColormap.from_list(
        'station_density_gray', ['#F3F3F3', '#D6D6D6', '#A5A5A5', '#545454'])
    ids = sorted(co.STATION_CODE, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))
    scopes = [*months, 'Pooled']

    def panel(ax, letter, pad=10):
        ax.annotate(letter, xy=(0, 1), xytext=(-18, pad), textcoords='offset points',
                    xycoords='axes fraction', fontsize=TEXT_SIZE, fontweight='bold',
                    ha='left', va='bottom', annotation_clip=False)

    def colorbar_style(cb, logarithmic=False):
        cb.ax.tick_params(labelsize=TICK_SIZE, length=3, width=.8)
        cb.ax.xaxis.set_minor_locator(NullLocator())
        cb.ax.yaxis.set_minor_locator(NullLocator())
        cb.outline.set_linewidth(.7)
        if logarithmic:
            cb.locator = LogLocator(base=10, subs=(1,), numticks=5)
            cb.formatter = LogFormatterMathtext(base=10)
            cb.update_ticks()
            cb.minorticks_off()

    def finish(fig, name):
        out.mkdir(parents=True, exist_ok=True)
        stem = out / name
        try:
            for ext in ('.png', '.pdf'):
                with BytesIO() as buffer:
                    fig.savefig(buffer, format=ext[1:], dpi=cfg.dpi,
                                bbox_inches='tight', pad_inches=.06)
                    content = buffer.getvalue()
                if ext == '.pdf' and not content.rstrip().endswith(b'%%EOF'):
                    raise RuntimeError('PDF rendering was incomplete.')
                path = stem.with_suffix(ext)
                path.write_bytes(content)
                print(f'Saved: {path}')
        finally:
            plt.close(fig)

    def matrix(p, value, selected_scopes, mask=False):
        t = station_metrics.loc[station_metrics.pollutant.eq(p)].copy()
        if mask:
            t.loc[~t.qualified, value] = np.nan
        return t.pivot(index='STATION_CODE', columns='scope', values=value).reindex(index=ids, columns=selected_scopes)

    fig = plt.figure(figsize=(12.6, 8.0), layout='constrained')
    fig.set_constrained_layout_pads(w_pad=.07, h_pad=.12, wspace=.08, hspace=.10)
    gs = fig.add_gridspec(2, 3, height_ratios=[.85, 1.25])
    ax = fig.add_subplot(gs[0, :])
    a = np.vstack([matrix(p, 'n_hours', months).to_numpy().T for p in PM])
    im = ax.imshow(np.ma.masked_where(a==0, a), aspect='auto', cmap=teal,
                   norm=LogNorm(vmin=1, vmax=max(2, np.nanmax(a))))
    tick_index = np.unique(np.r_[0, np.arange(4, len(ids), 5), len(ids)-1])
    if len(tick_index)>2 and tick_index[-1]-tick_index[-2]<3:
        tick_index = np.delete(tick_index, -2)
    ax.set_xticks(tick_index, [ids[i] for i in tick_index])
    ax.set_yticks(range(len(PM)*len(months)), [PM_LABEL[p]+'  '+LABELS[m] for p in PM for m in months])
    ax.set_xlabel('Station code')
    ax.set_title('Matched station-hours after temporal-support screening', loc='left', pad=12)
    panel(ax, 'a')
    ax.axhline(len(months)-.5, color='white', lw=3)
    colorbar_style(fig.colorbar(im, ax=ax, pad=.02, shrink=.9, label='Matched hours'), True)
    for j, (metric, title, xlabel) in enumerate([
            ('pearson_r', '$r$', '$r$'),
            ('mean_difference', 'ME', 'ME\n'+r'($\mu$g m$^{-3}$)'),
            ('mean_absolute_difference', 'MAE', 'MAE\n'+r'($\mu$g m$^{-3}$)')]):
        ax = fig.add_subplot(gs[1, j])
        y = 0
        yt, yl = [], []
        for p in PM:
            for scope in scopes:
                t = network.loc[network.scope.eq(scope) & network.pollutant.eq(p)].iloc[0]
                if scope == 'Pooled':
                    ax.axhspan(y-.4, y+.4, color='#F0F1EF', zorder=0)
                vals = [t[metric+'_'+k] for k in ['p05', 'q25', 'median', 'q75', 'p95']]
                if np.isfinite(vals).all():
                    ax.plot([vals[0], vals[4]], [y,y], color=PM_COLOR[p], alpha=.35, lw=1.6)
                    ax.plot([vals[1], vals[3]], [y,y], color=PM_COLOR[p], lw=6, solid_capstyle='round')
                    ax.plot(vals[2], y, 'D' if scope=='Pooled' else 'o', color=PM_COLOR[p], mfc='white', mew=1.8, ms=6)
                count_label = f'(n={int(t[metric+"_station_count"])})'
                yt.append(y); yl.append(PM_LABEL[p]+'  '+LABELS[scope]+'  '+count_label if j==0 else count_label)
                y += 1
            y += .7
        ax.set_yticks(yt, yl); ax.invert_yaxis(); ax.set_xlabel(xlabel)
        ax.set_title(title, loc='left', pad=12); ax.grid(axis='x', alpha=.15)
        panel(ax, chr(ord('b') + j))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if metric=='pearson_r':
            rv = network[metric+'_p05'].dropna()
            low = max(-1.02, np.floor((rv.min()-.04)*20)/20) if len(rv) else -1.02
            ax.set_xlim(low, 1.02)
        elif metric=='mean_difference':
            vals = network[[metric+'_p05', metric+'_p95']].to_numpy(dtype=float)
            bound = max(1., np.nanmax(np.abs(vals))*1.08) if np.isfinite(vals).any() else 1.
            ax.set_xlim(-bound, bound)
            ax.axvline(0, color='#555555', lw=.8, zorder=0)
        else:
            rv = network[metric+'_p95'].dropna()
            ax.set_xlim(0, max(1., float(rv.max())*1.08) if len(rv) else 1.)
    finish(fig, 'Figure_station_network_overview')

    fig, axes = plt.subplots(2, 2, figsize=(9.3, 9.6), layout='constrained')
    fig.set_constrained_layout_pads(w_pad=.07, h_pad=.12, wspace=.09, hspace=.13)
    handles = [Line2D([0],[0],color=MONTH_COLORS[m],marker=MONTH_MARKERS[m],
                     mfc='white',mew=1.05,ms=4.5,lw=2,label=LABELS[m]) for m in months]
    fig.legend(handles=handles, loc='outside upper center', ncol=len(months), frameon=False)
    for i, p in enumerate(PM):
        if selected.empty or not selected.pollutant.eq(p).any():
            for ax in axes[i]:
                ax.text(.5,.5,'No station meets sample-support criteria',transform=ax.transAxes,ha='center')
                ax.set_axis_off()
            continue
        t = selected.loc[selected.pollutant.eq(p)].iloc[0]
        g = pairs.loc[pairs.pollutant.eq(p) & pairs.STATION_CODE.eq(t.STATION_CODE)]
        for j, ax in enumerate(axes[i]):
            x = g.reference_value
            y = g.mobile_value if j==0 else g.difference
            hb = ax.hexbin(x, y, gridsize=32, mincnt=1, bins='log', cmap=density_gray,
                           linewidths=0, zorder=1)
            colorbar_style(fig.colorbar(hb, ax=ax, shrink=.6, pad=.025, label='Hours per hexagon'), True)
            for month, h in g.groupby('month'):
                xx = h.reference_value
                yy = h.mobile_value if j==0 else h.difference
                bins = pd.qcut(xx, q=min(8, max(1, len(h)//5)), duplicates='drop')
                b = pd.DataFrame({'x':xx, 'y':yy, 'bin':bins}).groupby('bin', observed=True).agg(x=('x','median'),y=('y','median'),n=('x','size'))
                b = b.loc[b.n.ge(5)]
                curve, = ax.plot(b.x, b.y, color=MONTH_COLORS[month],
                                marker=MONTH_MARKERS[month], ms=4.5, mfc='white',
                                mew=1.05, lw=2, zorder=6)
                # A thin white stroke separates monthly curves from the density layer.
                curve.set_path_effects([
                    patheffects.Stroke(linewidth=3.3, foreground='white'),
                    patheffects.Normal(),
                ])
            ax.grid(alpha=.12)
            ax.set_title(PM_LABEL[p]+f': station {t.STATION_CODE}'+('' if j==0 else ' errors'), loc='left', pad=62)
            panel(ax, chr(ord('a') + i*2+j), pad=62)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if j==0:
                lim = max(float(x.max()), float(y.max()))*1.05
                lim = max(lim, 1.)
                ax.plot([0,lim],[0,lim],color='#353D40',lw=1,zorder=3)
                ax.set(xlim=(0,lim),ylim=(0,lim),aspect='equal',
                       xlabel='Fixed-station hourly\n'+r'concentration ($\mu$g m$^{-3}$)',
                       ylabel='Mobile observed-minute\n'+r'summary ($\mu$g m$^{-3}$)')
                text = f'$N$ = {int(t.n_hours):,} hours; {int(t.n_days)} days\n$r$ = {t.pearson_r:.3f}\nMAE = {t.mean_absolute_difference:.2f} '+r'$\mu$g m$^{-3}$'
                ax.text(0,1.025,text,transform=ax.transAxes,ha='left',va='bottom',fontsize=NOTE_SIZE)
            else:
                ax.axhline(0,color='#444C4E',lw=.8,zorder=3)
                ax.axhline(t.mean_difference,color='#444C4E',lw=1.4,zorder=3)
                for v in [t.loa_low,t.loa_high]:
                    ax.axhline(v,color='#7D878A',ls='--',lw=1,zorder=3)
                ax.set(xlabel='Fixed-station hourly\nconcentration '+r'($\mu$g m$^{-3}$)',
                       ylabel='Mobile minus fixed-site\nconcentration '+r'($\mu$g m$^{-3}$)')
                ax.text(0,1.025,f'ME = {t.mean_difference:.2f}\nDescriptive limits:\n[{t.loa_low:.2f}, {t.loa_high:.2f}] '+r'$\mu$g m$^{-3}$',
                        transform=ax.transAxes,ha='left',va='bottom',fontsize=NOTE_SIZE)
    finish(fig, 'Figure_station_representative_diagnostics')

    # Complete station-level view is supplementary; no station is silently omitted.
    fig, axes = plt.subplots(1, 8, figsize=(12.6, max(8.0, len(ids)*.205)), layout='constrained')
    fig.set_constrained_layout_pads(w_pad=.045, h_pad=.10, wspace=.035)
    for i, p in enumerate(PM):
        for j, (metric, title) in enumerate([('n_hours','Matched\nhours'),('pearson_r','$r$'),
                                            ('mean_difference','ME'),
                                            ('mean_absolute_difference','MAE')]):
            ax = axes[i*4+j]
            # pandas Copy-on-Write may expose a read-only NumPy view. This
            # plotting array is modified below, so explicitly own its memory.
            a = matrix(p, metric, scopes, mask=j>0).to_numpy(dtype=float, copy=True)
            if j==0:
                a[a==0] = np.nan
                norm = LogNorm(vmin=1, vmax=max(2, np.nanmax(a) if np.isfinite(a).any() else 2))
                cmap = teal
            elif j==1:
                norm = Normalize(-1,1); cmap = plt.get_cmap('RdBu').copy(); cmap.set_bad('#E9E9E9')
            elif j==2:
                bound = max(1., np.nanmax(np.abs(a)) if np.isfinite(a).any() else 1.)
                norm = Normalize(-bound, bound)
                cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#E9E9E9')
            else:
                norm = Normalize(0,max(1, np.nanmax(a) if np.isfinite(a).any() else 1)); cmap = teal
            im = ax.imshow(np.ma.masked_invalid(a),aspect='auto',cmap=cmap,norm=norm)
            ax.set_title(PM_LABEL[p]+'\n'+title, pad=12)
            ax.text(0, 1.11, chr(ord('a')+i*4+j), transform=ax.transAxes,
                    fontsize=TEXT_SIZE, fontweight='bold', ha='left', va='bottom')
            short_labels = {'March':'Mar', 'August':'Aug', 'November':'Nov', 'Pooled':'Pooled'}
            ax.set_xticks(range(len(scopes)),[short_labels[LABELS[s]] for s in scopes],rotation=60,ha='right')
            ax.set_yticks(range(len(ids)),ids if j==0 else ['']*len(ids),fontsize=TICK_SIZE)
            ax.tick_params(axis='y', length=2 if j==0 else 0, pad=3)
            if j==0: ax.set_ylabel('Station code')
            cb = fig.colorbar(im,ax=ax,location='bottom',fraction=.025,pad=.035, aspect=12)
            colorbar_style(cb, j==0)
            if j==0:
                cb.set_ticks(sorted(set([1, 10**int(np.floor(np.log10(norm.vmax)))])))
            elif j==1:
                cb.set_ticks([-1, 0, 1])
            elif j==2:
                cb.set_ticks([-bound, 0, bound])
                cb.set_label(r'$\mu$g m$^{-3}$', fontsize=TICK_SIZE)
            else:
                cb.locator = MaxNLocator(nbins=2)
                cb.update_ticks()
                cb.set_label(r'$\mu$g m$^{-3}$', fontsize=TICK_SIZE)
    finish(fig, 'Figure_station_all_sites_supplement')


def run(release, output, cfg, months=MONTHS, transformer=None):
    release, output = Path(release), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output not empty; choose a new --output directory: {output}')
    paths = [release/'station_coordinates.csv']
    for m in months:
        paths += [release/f'Guangzhou_station_{m}_1h_QC.csv.gz', release/f'Guangzhou_mobile_{m}_15s_QC.csv.gz']
    missing = [str(p) for p in paths if not p.is_file()]
    if missing: raise FileNotFoundError('\n'.join(missing))
    if (cfg.min_minutes < 1 or cfg.radius_m <= 0 or cfg.chunksize < 1
            or cfg.comparison_radius_m <= cfg.radius_m):
        raise ValueError('Invalid settings.')
    transformer = transformer or get_transformer(cfg)
    co, ref, reference_audit = load_stations(release, months, cfg, transformer)
    output.mkdir(parents=True,exist_ok=True)
    for folder in ['tables','analysis_data','figures','cache']:
        (output/folder).mkdir(exist_ok=True)
    before = {str(p):dict(size_bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths}
    config = dict(version=VERSION, settings=asdict(cfg), months=list(months), timezone='Asia/Shanghai',
                  hour_labels='interval_start [t,t+1h), confirmed by data owner',
                  source_files=before, small_file_sha256={str(p):hash_file(p) for p in paths if 'mobile_' not in p.name},
                  python=platform.python_version(), pandas=pd.__version__,
                  numpy=np.__version__, status='running')
    source_qc_config = release/'reports/station_reference_qc/qc_config.json'
    if source_qc_config.is_file():
        config['fixed_station_qc'] = json.loads(source_qc_config.read_text(encoding='utf-8'))
    (output/'analysis_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    save_csv(co,output/'tables/station_coordinates_projected.csv')
    save_csv(reference_audit,output/'analysis_data/reference_eligibility.csv.gz')
    con = init_db(output/'cache/nearby_mobile.sqlite')
    try:
        scan = ingest_mobile(con,release,co,ref,months,cfg,transformer)
        save_csv(scan,output/'tables/mobile_scan_summary.csv')
        pair_results = {
            cfg.radius_m: make_pairs(con, ref, cfg, cfg.radius_m),
            cfg.comparison_radius_m: make_pairs(con, ref, cfg, cfg.comparison_radius_m),
        }
    finally:
        con.close()
    pairs = pair_results[cfg.radius_m]
    accepted, st, net = make_tables(pairs,co,months,cfg)
    comparison_pairs = pair_results[cfg.comparison_radius_m]
    accepted_comparison, st_comparison, net_comparison = make_tables(
        comparison_pairs, co, months, cfg)
    radius_summary, common_keys = radius_sensitivity_summary(
        pair_results, co, months, cfg)
    selected = select_examples(st)
    save_csv(pairs,output/'analysis_data/all_reference_hours_with_mobile_support.csv.gz')
    save_csv(accepted,output/'analysis_data/qualified_station_hour_pairs.csv.gz')
    save_csv(st,output/'tables/station_month_metrics.csv')
    save_csv(net,output/'tables/network_summary.csv')
    suffix = f'{cfg.comparison_radius_m:g}m'
    save_csv(comparison_pairs, output/f'analysis_data/all_reference_hours_with_mobile_support_{suffix}.csv.gz')
    save_csv(accepted_comparison, output/f'analysis_data/qualified_station_hour_pairs_{suffix}.csv.gz')
    save_csv(st_comparison, output/f'tables/station_month_metrics_{suffix}.csv')
    save_csv(net_comparison, output/f'tables/network_summary_{suffix}.csv')
    save_csv(radius_summary, output/'tables/radius_sensitivity_summary.csv')
    common_name = f'common_station_hour_keys_{cfg.radius_m:g}m_{cfg.comparison_radius_m:g}m.csv.gz'
    save_csv(common_keys, output/'analysis_data'/common_name)
    save_csv(selected,output/'tables/representative_stations.csv')
    flow = pairs.groupby(['month','pollutant','support_ok','support_reason'],dropna=False).size().reset_index(name='station_hours')
    save_csv(flow,output/'tables/matching_exclusions.csv')
    assert not pairs.duplicated(['STATION_CODE','TIME_POINT','pollutant']).any()
    assert np.isfinite(accepted[['reference_value','mobile_value']]).all().all()
    assert accepted.max_distance_m.le(cfg.radius_m+1e-6).all()
    assert accepted_comparison.max_distance_m.le(cfg.comparison_radius_m+1e-6).all()
    assert not comparison_pairs.duplicated(['STATION_CODE','TIME_POINT','pollutant']).any()
    common_counts = radius_summary.loc[radius_summary.analysis_set.eq('common_station_hours')]
    assert common_counts.groupby('pollutant').matched_station_hours.nunique().le(1).all()
    assert len(st)==len(co)*len(PM)*(len(months)+1)
    plot_results(st,net,accepted,selected,co,months,cfg,output/'figures')
    after = {str(p):dict(size_bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths}
    if before != after: raise RuntimeError('An input file changed during analysis.')
    config.update(status='completed',paired_station_hours=len(accepted),
                  comparison_paired_station_hours=len(accepted_comparison),
                  figures='network overview; representative diagnostics; all-station supplement')
    (output/'analysis_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    print(net[['scope','pollutant','stations_with_pairs','qualified_stations','paired_station_hours']].to_string(index=False))
    print('\nRadius sensitivity (pooled):')
    print(radius_summary.to_string(index=False))
    print(f'Complete: {output}\nNo source changes. Nearby SQLite cache retained for audit.',flush=True)
    return pairs,st,net




def main():
    global FONT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir',type=Path,default=RELEASE)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--font-dir',type=Path,default=FONT_DIR,
                        help='Optional directory containing publication fonts.')
    parser.add_argument('--radius-m',type=float,default=500.)
    parser.add_argument('--comparison-radius-m',type=float,default=1000.)
    parser.add_argument('--chunksize',type=int,default=300_000)
    parser.add_argument('--dpi',type=int,default=600,
                        help='Resolution of PNG figures.')
    parser.add_argument('--plot-only',action='store_true',
                        help='Redraw figures from completed tables/pairs; no mobile rescan.')
    args = parser.parse_args()
    release_dir = args.release_dir.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    FONT_DIR = args.font_dir.expanduser().resolve()
    if args.plot_only:
        redraw(output_dir, dpi=args.dpi)
        return
    run(release_dir,output_dir,Settings(radius_m=args.radius_m,
        comparison_radius_m=args.comparison_radius_m,
        chunksize=args.chunksize,dpi=args.dpi))


def redraw(output, dpi=None):
    output = Path(output)
    config = json.loads((output/'analysis_config.json').read_text(encoding='utf-8'))
    settings = dict(config['settings'])
    if dpi is not None:
        settings['dpi'] = dpi
    cfg = Settings(**settings)
    st = pd.read_csv(output/'tables/station_month_metrics.csv',dtype={'STATION_CODE':str})
    net = pd.read_csv(output/'tables/network_summary.csv')
    pairs = pd.read_csv(output/'analysis_data/qualified_station_hour_pairs.csv.gz',dtype={'STATION_CODE':str})
    pairs['TIME_POINT'] = timestamp(pairs.TIME_POINT)
    co = pd.read_csv(output/'tables/station_coordinates_projected.csv',dtype={'STATION_CODE':str})
    selected = select_examples(st)
    # Output filenames are exactly the three generated figure PNG/PDF pairs.
    plot_results(st,net,pairs,selected,co,config['months'],cfg,output/'figures')
    print('Redrew six generated figure files. Input data and metric tables unchanged.',flush=True)


if __name__=='__main__':
    main()
