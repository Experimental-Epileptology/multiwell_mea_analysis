import hashlib
import os
import re

import numpy as np
import pandas as pd
from .ephys_proc import pop_burst_detection

na = np.array


def gid_to_numbers(gid):
    """Convert the gid to sequential integer labels 0..n_unique-1."""
    _, inverse = np.unique(gid, return_inverse=True)
    return inverse


def spike_features(st, gid, T):
    """
    Calculate basic spike train features.

    Parameters:
    - st: np.ndarray
        Array of spike times.
    - gid: np.ndarray
        Array of neuron (channel) IDs corresponding to each spike time.
    - T: float
        Total recording duration (same unit as st).

    Returns:
    - mean_isi: float
        Mean inter-spike interval (averaged across channels).
    - cv_isi: float
        Mean coefficient of variation (CV) of ISI across channels.
    - fr: float
        Mean firing rate per channel (spikes per unit time per channel).
    """
    unique_gids = np.unique(gid)
    misis = []
    cv_isis = []
    spike_counts_ch = []

    for unit_id in unique_gids:
        channel_spike_times = st[gid == unit_id]
        channel_isi = np.diff(channel_spike_times)
        if len(channel_isi) > 0:
            misi = np.mean(channel_isi)
            misis.append(misi)
            cv_isis.append(np.std(channel_isi) / misi)
        spike_counts_ch.append(len(channel_spike_times))

    mean_isi = np.mean(misis)
    cv_isi = np.mean(cv_isis)
    fr = (len(st) / T) / len(unique_gids)

    return mean_isi, cv_isi, fr, spike_counts_ch


def burst_features(bursts_well, T):
    """
    Calculate burst features for spike trains recorded from multiple channels.

    Parameters:
    - bursts_well: pd.DataFrame
        DataFrame containing burst information with columns:
        'Channel ID', 'Start timestamp [µs]', 'Duration [µs]', and 'Spike Count'.
    - T: float
        Total recording duration in seconds.

    Returns:
    - mean_ibi: float
        Mean inter-burst interval across all channels (in seconds).
    - cv_ibi: float
        Coefficient of variation (CV) of inter-burst intervals across channels.
    - mean_burst_rate: float
        Mean number of bursts per second per channel.
    - mean_burst_duration: float
        Mean burst duration across channels (in seconds).
    - cv_burst_duration: float
        Coefficient of variation (CV) of burst durations across channels.
    - mean_rate_in_burst: float
        Mean firing rate inside bursts (spikes per second during bursts).
    """
    channels = na(bursts_well['Channel ID'])
    unique_channels = np.unique(channels)

    avg_rate_in_burst = []
    mibis = []
    cv_ibis = []
    cv_bds = []
    burst_rates = []
    mbd = []

    for ch in unique_channels:
        mask = (channels == ch)
        burst_start = na(bursts_well.loc[mask, 'Start timestamp [µs]']) * 1e-6
        burst_durs = na(bursts_well.loc[mask, 'Duration [µs]']) * 1e-6
        spike_count = na(bursts_well.loc[mask, 'Spike Count'])

        if len(burst_start) > 10:
            burst_times = np.squeeze(burst_start)
            burst_durs = np.squeeze(burst_durs)
            ibis = np.diff(burst_times)

            mibis.append(np.mean(ibis))
            cv_ibis.append(np.std(ibis) / np.mean(ibis))
            burst_rates.append(len(burst_times) / T)
            mbd.append(np.mean(burst_durs))
            cv_bds.append(np.std(burst_durs) / np.mean(burst_durs))
            avg_rate_in_burst.append(np.mean(spike_count / burst_durs))
        else:
            mibis.append(np.nan)
            cv_ibis.append(np.nan)
            burst_rates.append(np.nan)
            mbd.append(np.nan)
            cv_bds.append(np.nan)
            avg_rate_in_burst.append(np.nan)

    return (
        np.nanmean(mibis),
        np.nanmean(cv_ibis),
        np.nanmean(burst_rates),
        np.nanmean(mbd),
        np.nanmean(cv_bds),
        np.nanmean(avg_rate_in_burst)
    )


def netburst_features(bursts_well, T):
    """
    Calculate global (network-level) burst features.

    Parameters:
    - bursts_well: pd.DataFrame
        DataFrame containing network burst information with columns:
        'Start timestamp [µs]', 'Duration [µs]', and 'Spike Count'.
    - T: float
        Total recording duration in seconds.

    Returns:
    - mean_ibi: float
        Mean inter-burst interval (seconds).
    - cv_ibi: float
        Coefficient of variation (CV) of inter-burst intervals.
    - burst_rate: float
        Network burst rate (bursts per second).
    - mean_burst_duration: float
        Mean burst duration (seconds).
    - cv_burst_duration: float
        Coefficient of variation (CV) of burst durations.
    - total_spikes_in_bursts: int
        Total number of spikes across all bursts.
    - mean_rate_in_bursts: float
        Mean spike rate inside bursts (spikes per second).
    - n_bursts: int
        Total number of network bursts detected.
    """
    burst_start = na(bursts_well['Start timestamp [µs]']) * 1e-6
    burst_durs = na(bursts_well['Duration [µs]']) * 1e-6
    spike_count = na(bursts_well['Spike Count'])

    in_burst_rate = np.mean(spike_count / burst_durs)
    in_burst_spikes = np.sum(spike_count)
    n_bursts = len(burst_start)
    total_burst_time = np.sum(burst_durs) / T

    if n_bursts > 10:
        burst_times = np.squeeze(burst_start)
        burst_durations = np.squeeze(burst_durs)

        ibis = np.diff(burst_times)

        mean_ibi = np.mean(ibis)
        cv_ibi = np.std(ibis) / np.mean(ibis)
        burst_rate = n_bursts / T
        mean_burst_duration = np.mean(burst_durations)
        cv_burst_duration = np.std(burst_durations) / np.mean(burst_durations)
        smbr = np.sum(ibis < 1.) / len(ibis)

    else:
        mean_ibi = np.nan
        cv_ibi = np.nan
        burst_rate = np.nan
        mean_burst_duration = np.nan
        cv_burst_duration = np.nan
        smbr = np.nan

    return (
        mean_ibi,
        cv_ibi,
        burst_rate,
        mean_burst_duration,
        cv_burst_duration,
        in_burst_spikes,
        in_burst_rate,
        n_bursts,
        total_burst_time,
        smbr
    )


SPIKE_COLS = ['Well Label', 'Channel ID', 'Timestamp [µs]']
BURST_COLS = ['Well Label', 'Channel ID', 'Start timestamp [µs]', 'Duration [µs]', 'Spike Count']
NETBURST_COLS = ['Well Label', 'Start timestamp [µs]', 'Duration [µs]', 'Spike Count']


def read_ephys(path, wells=None, cache_dir=None):
    """
    Load the spikes/bursts/networkbursts CSVs for a recording file.

    Only reads the columns actually used downstream (dropping the large
    per-row string columns like 'Compound Name'/'Experiment' cuts parse
    time by >10x on a full-plate file). Pass `wells` (a well label, or a
    list of them) to also drop rows for other wells right after reading,
    which is nearly free once the unused columns are gone - useful when
    you only need one or two wells out of a full plate (e.g. for a
    one-off plot) rather than every well on it.

    Column/row pruning still can't avoid reading every byte of a large
    CSV off disk on its first touch. If the same (file, wells) pair gets
    read repeatedly (e.g. re-running an exploratory plotting cell), pass
    `cache_dir` to cache the already-narrowed result on local disk, keyed
    by the source file's path and mtime - later calls skip the CSV read
    entirely. Source recordings are treated as immutable once written; the
    cache is invalidated only if the file's mtime changes.
    """
    cache_file = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key = f"{path}|{wells}|{os.path.getmtime(path)}"
        digest = hashlib.md5(key.encode()).hexdigest()[:16]
        cache_file = os.path.join(cache_dir, f"{digest}.pkl")
        if os.path.exists(cache_file):
            return pd.read_pickle(cache_file)

    split_file = path.split('_')
    split_file[-1] = 'networkbursts.csv'
    netburst_file = '_'.join(split_file)
    split_file[-1] = 'spikes.csv'
    spike_file = '_'.join(split_file)
    split_file[-1] = 'bursts.csv'
    burst_file = '_'.join(split_file)
    spikes = pd.read_csv(spike_file, usecols=SPIKE_COLS)
    netbursts = pd.read_csv(netburst_file, usecols=NETBURST_COLS)
    bursts = pd.read_csv(burst_file, usecols=BURST_COLS)
    if wells is not None:
        wells = [wells] if isinstance(wells, str) else wells
        spikes = spikes[spikes['Well Label'].isin(wells)]
        bursts = bursts[bursts['Well Label'].isin(wells)]
        netbursts = netbursts[netbursts['Well Label'].isin(wells)]

    if cache_file is not None:
        pd.to_pickle((spikes, bursts, netbursts), cache_file)

    return spikes, bursts, netbursts


def get_pop_bursts(st_p, gid_p,
                   maxISIstart=100,
                   maxISIb=250,
                   minBdur=20,
                   minIBI=20,
                   minSburst=5,
                   onset_threshold=6,
                   min_active_electrodes=6,
                   mad_multiplier=0.5):
    """Run pop_burst_detection on spike times (seconds). Returns Nx2 array in seconds."""
    if st_p is None or len(st_p) == 0:
        return np.empty((0, 2))
    try:
        result = pop_burst_detection(
            st_p * 1000,
            gid_p,
            maxISIstart=maxISIstart,
            maxISIb=maxISIb,
            minBdur=minBdur,
            minIBI=minIBI,
            minSburst=minSburst,
            onset_threshold=onset_threshold,
            min_active_electrodes=min_active_electrodes,
        )
        if not result:
            return np.empty((0, 2))
        b_times = na(result, dtype=float) / 1000.0
        is_real, _, _ = flag_real_bursts(st_p, b_times, mad_multiplier=mad_multiplier)
        return b_times[is_real]
    except Exception as e:
        print("pop_burst_detection error:", e)
        return np.empty((0, 2))


def flag_real_bursts(st_p, b_times, mad_multiplier=0.5):
    """Flag detected bursts using log inter-spike-interval (ISI) statistics.

    No binning is used. Both the baseline and each burst's activity are
    characterised directly from consecutive-spike intervals in the pooled
    (population) spike train, worked in log-ISI space: raw ISI is heavily
    right-skewed (a few long gaps inflate the MAD and compress every
    burst's z-score toward zero), so log-ISI is used instead - the same
    reasoning behind the log-ISI valley method in auto_maxISI
    (ephys_proc.py).

    - Baseline log-ISI: log(interval) for consecutive spikes that both
      fall outside every detected burst (the quiet, between-burst
      activity).
    - Burst log-ISI: the median log(interval) between consecutive spikes
      that fall inside a given burst window.

    A burst is "real" if its own spikes are packed significantly tighter
    (smaller log-ISI => higher instantaneous rate) than baseline, using
    the median absolute deviation robust z-score
    (https://en.wikipedia.org/wiki/Median_absolute_deviation):

        M_i = 0.6745 * (burst_median_log_isi_i - baseline_median_log_isi) / baseline_mad_log_isi

    A burst is considered real if M_i < -mad_multiplier.

    Parameters
    ----------
    st_p : array-like
        All spike times for the well (seconds), across all channels.
    b_times : array-like, shape (N, 2)
        Burst [start, end] times (seconds), e.g. from get_pop_bursts.
    mad_multiplier : float
        Modified z-score threshold (default 0.5).

    Returns
    -------
    is_real : np.ndarray of bool, shape (N,)
        True where a burst's log-ISI is significantly below baseline.
    burst_median_isi : np.ndarray of float, shape (N,)
        Median inter-spike interval (seconds, linear scale) within each
        burst (NaN if fewer than 2 usable spikes).
    modified_z : np.ndarray of float, shape (N,)
        The robust z-score computed for each burst's median log-ISI.
    """
    b_times = np.asarray(b_times)
    if b_times.size == 0:
        return (np.empty((0,), dtype=bool), np.empty((0,), dtype=float),
                np.empty((0,), dtype=float))

    st_p = np.sort(np.asarray(st_p))
    n = len(b_times)
    if len(st_p) < 2:
        return (np.zeros(n, dtype=bool), np.full(n, np.nan), np.full(n, np.nan))

    in_burst = np.zeros(len(st_p), dtype=bool)
    for start, end in b_times:
        in_burst |= (st_p >= start) & (st_p <= end)

    pop_isi = np.diff(st_p)
    # An ISI counts toward the baseline only if both spikes bounding it sit
    # outside every burst.
    baseline_isis = pop_isi[~in_burst[:-1] & ~in_burst[1:]]
    baseline_isis = baseline_isis[baseline_isis > 0]

    if len(baseline_isis) == 0:
        return (np.zeros(n, dtype=bool), np.full(n, np.nan), np.full(n, np.nan))

    baseline_log_isi = np.log(baseline_isis)
    baseline_median = np.median(baseline_log_isi)
    baseline_mad = np.median(np.abs(baseline_log_isi - baseline_median))

    burst_median_isi = np.full(n, np.nan)
    burst_median_log_isi = np.full(n, np.nan)
    for i, (start, end) in enumerate(b_times):
        spikes_in_burst = np.sort(st_p[(st_p >= start) & (st_p <= end)])
        isis = np.diff(spikes_in_burst)
        isis = isis[isis > 0]
        if len(isis) > 0:
            burst_median_isi[i] = np.median(isis)
            burst_median_log_isi[i] = np.median(np.log(isis))

    deviations = burst_median_log_isi - baseline_median
    if baseline_mad > 0:
        modified_z = 0.6745 * deviations / baseline_mad
    else:
        # MAD == 0: any below-baseline deviation is an effectively infinite
        # (negative) z-score; at/above baseline is not an outlier.
        modified_z = np.where(deviations < 0, -np.inf, 0.0)

    is_real = np.where(np.isfinite(burst_median_log_isi), modified_z < -mad_multiplier, False)
    return is_real, burst_median_isi, modified_z


# Defaults for get_pop_bursts, used whenever get_new_bursts/get_features/
# process_recording_file/process_multidrug_recording_file are called without
# an explicit `burst_params` dict. Pass burst_params through from a notebook
# to use different burst-detection settings for a different dataset without
# touching this file - see get_pop_bursts for what each key means.
DEFAULT_BURST_PARAMS = dict(
    maxISIstart=100,
    maxISIb=250,
    minBdur=20,
    minIBI=20,
    minSburst=5,
    onset_threshold=6,
    min_active_electrodes=6,
    mad_multiplier=0.5,
)


def get_new_bursts(st, gid, burst_params=None):
    st_flat = np.asarray(st).ravel()
    gid_flat = np.asarray(gid).ravel()

    params = DEFAULT_BURST_PARAMS if burst_params is None else burst_params
    b_times = get_pop_bursts(st_flat, gid_flat, **params)

    n_bursts = len(b_times)
    if n_bursts > 3:
        burst_durations = b_times[:, 1] - b_times[:, 0]
        ibis = b_times[1:, 0] - b_times[:-1, 1]
        mean_ibi = np.mean(ibis)
        cv_ibi = np.std(ibis) / np.mean(ibis)
        mean_burst_duration = np.mean(burst_durations)
        cv_burst_duration = np.std(burst_durations) / np.mean(burst_durations)
    else:
        mean_ibi = np.nan
        cv_ibi = np.nan
        cv_burst_duration = np.nan
        mean_burst_duration = np.nan

    return (
        mean_ibi,
        cv_ibi,
        mean_burst_duration,
        cv_burst_duration,
    )


def get_features(well_name, spikes, bursts, netbursts, burst_params=None):
    """ Collect features for a well.

    burst_params: optional dict of get_pop_bursts kwargs (maxISIstart,
    maxISIb, minBdur, minIBI, minSburst, onset_threshold,
    min_active_electrodes, mad_multiplier) used for the "(custom)"
    network-burst features. Defaults to DEFAULT_BURST_PARAMS when not given.
    """
    spikes_well = spikes[spikes['Well Label'] == well_name]
    bursts_well = bursts[bursts['Well Label'] == well_name]
    netbursts_well = netbursts[netbursts['Well Label'] == well_name]
    st = na((spikes_well.filter(items=['Timestamp [µs]'])[spikes_well['Well Label'] == well_name])) * 1e-6
    gid = na(spikes_well.filter(items=['Channel ID'])[spikes_well['Well Label'] == well_name])
    try:
        T = np.max(na(st))
        mean_isi, cv_isi, frN, spike_counts = spike_features(st, gid, T)
        mibis, cv_ibis, burst_rates, mbd, cv_bds, avg_rate_in_burst = burst_features(bursts_well, T)
        netmibi, netcvIBI, net_burst_rate, net_mbd, net_cvbs, in_burst_spikes, in_net_burst_rate, n_bursts, total_burst_time, smbr = netburst_features(netbursts_well, T)
        mean_ibi_new, cv_ibi_new, mean_burst_duration_new, cv_burst_duration_new = get_new_bursts(st, gid, burst_params=burst_params)
        random_spikes = ((np.sum(spike_counts) - in_burst_spikes) / np.sum(spike_counts)) * 100
        feats = [mean_isi, cv_isi, frN,
                 mibis, cv_ibis, burst_rates,
                 mbd, cv_bds, avg_rate_in_burst,
                 netmibi,
                 netcvIBI, net_burst_rate, net_mbd,
                 net_cvbs, mean_ibi_new, cv_ibi_new,
                 mean_burst_duration_new, cv_burst_duration_new, in_net_burst_rate,
                 random_spikes, total_burst_time, smbr]
    except Exception as e:
        print(f"get_features error (well {well_name}):", e)
        feats = [np.nan] * len(feature_names)
    return feats


def read_layout(filename):
    """Read a well layout file (csv or xlsx) into a dict of well -> condition."""
    try:
        layout_df = pd.read_csv(filename, sep=';')
    except Exception:
        # Compatibility with hannas files
        layout_df = pd.read_excel(filename).dropna()

    result = {}
    for _, row in layout_df.iterrows():
        row_label = row.iloc[0]  # First column (A, B, C, D)
        for col_idx, value in enumerate(row.iloc[1:], start=1):
            result[f"{row_label}{col_idx}"] = value
    return result


def process_recording_file(main_path, layouts_path, diff, week, file, burst_params=None):
    """Process one recording file: parse its name, load its well layout, read its
    spikes/bursts/netbursts CSVs once, and compute features for every well.

    This is the unit of parallelism for building the conditions table - each
    file is independent, so it can run inside a ProcessPoolExecutor worker.
    Must stay a plain module-level function (not a notebook closure) so it can
    be pickled and sent to worker processes.

    burst_params: optional dict of get_pop_bursts kwargs, forwarded to
    get_features for every well in this file - see get_features/
    DEFAULT_BURST_PARAMS. Lets one notebook use different burst-detection
    settings than another without editing this file.

    Returns:
        (results, skipped): results is a list of (id_, row) tuples; skipped is
        the filename if it didn't match the expected 9-field naming
        convention, otherwise None.
    """
    sp_file = file.split('_')
    if len(sp_file) != 9:
        return [], file

    exp, line, DIV, batch, drug, treatment_time, diff_date, rec_date, _ = sp_file
    layout_name = f"{line}_{drug}_{diff_date}"
    layout = read_layout(os.path.join(layouts_path, layout_name + '.xlsx'))

    file_path = os.path.join(main_path, diff, week, 'Export', file)
    spikes, bursts, netbursts = read_ephys(file_path)

    results = []
    for well, LTXcondition in layout.items():
        if LTXcondition in ('X', 'Y'):
            continue
        id_ = f"{rec_date}_{well}_{batch}_{line}_{treatment_time}_{DIV}_{drug}"
        features = get_features(well, spikes, bursts, netbursts, burst_params=burst_params)
        row = [file_path, line, diff_date, batch, week, DIV, treatment_time, rec_date,
               layout_name, well, drug, LTXcondition, *features]
        results.append((id_, row))

    return results, None


def read_layout_single(filename, sheet_name):
    """Read one sheet of a per-cell-line layout workbook into well -> condition.

    Used for experiments where multiple drugs share one plate/layout, so the
    layout lives in one Excel workbook per cell line with one sheet per
    "<diff_date>_<letter>" batch, rather than one file per (line, drug, diff).
    """
    try:
        layout_df = pd.read_csv(filename, sep=';')
    except Exception:
        # Compatibility with hannas files
        layout_df = pd.read_excel(filename, sheet_name=sheet_name).dropna()

    result = {}
    for _, row in layout_df.iterrows():
        row_label = row.iloc[0]  # First column (A, B, C, D)
        for col_idx, value in enumerate(row.iloc[1:], start=1):
            result[f"{row_label}{col_idx}"] = value
    return result


def parse_condition(raw):
    """Split a combined "DRUG (DOSE)" layout cell into (drug, dose).

    Plain values with no parenthesised dose (e.g. 'untreated', 'DMSO') return
    (None, raw) - they aren't tied to one specific drug on a multi-drug plate.
    Drug names have internal hyphens stripped ('LTX-2' -> 'LTX2') to match
    the drug naming used for single-drug-per-plate experiments.
    """
    raw = str(raw).strip()
    m = re.match(r'^(.*?)\s*\(([^)]+)\)\s*$', raw)
    if m:
        return m.group(1).replace('-', ''), m.group(2).strip()
    return None, raw


def process_multidrug_recording_file(main_path, layouts_path, diff, week, file, burst_params=None):
    """Process one recording file from a multi-drug-per-plate experiment:
    parse its name, load its well layout (one sheet of a per-cell-line
    workbook), read its spikes/bursts/netbursts CSVs once, and compute
    features for every well.

    Unlike process_recording_file, a single file/plate here isn't dedicated
    to one drug - each well's layout cell is a combined "DRUG (DOSE)" string
    (or 'untreated'/'DMSO' with no drug), parsed via parse_condition into
    separate drug/dose columns.

    Filename convention: exp_line_DIV_letter_treatmenttime_diffdate_recdate_*
    (8 fields, no separate drug/batch field - `week` is expected to already
    include any trailing path segment, e.g. 'week4/Export').

    This is the unit of parallelism for building the conditions table - each
    file is independent, so it can run inside a ProcessPoolExecutor worker.
    Must stay a plain module-level function (not a notebook closure) so it
    can be pickled and sent to worker processes.

    burst_params: optional dict of get_pop_bursts kwargs, forwarded to
    get_features for every well in this file - see get_features/
    DEFAULT_BURST_PARAMS. Lets this dataset use different burst-detection
    settings than another notebook without editing this file.

    Returns:
        (results, skipped): results is a list of (id_, row) tuples; skipped
        is the filename if it didn't match the expected 8-field naming
        convention or had no matching layout sheet, otherwise None.
    """
    sp_file = file.split('_')
    if len(sp_file) != 8:
        return [], file

    exp, line, DIV, letter, treatment_time, diff_date, rec_date, _ = sp_file
    layout_name = line
    try:
        layout = read_layout_single(os.path.join(layouts_path, layout_name + '.xlsx'),
                                     f'{diff_date}_{letter}')
    except Exception as e:
        print(f"read_layout_single error ({file}):", e)
        return [], file

    file_path = os.path.join(main_path, diff, week, file)
    spikes, bursts, netbursts = read_ephys(file_path)

    results = []
    for well, condition in layout.items():
        if condition in ('X', 'Y'):
            continue
        drug, dose = parse_condition(condition)
        id_ = f"{rec_date}_{well}_{line}_{treatment_time}_{DIV}_{letter}"
        features = get_features(well, spikes, bursts, netbursts, burst_params=burst_params)
        row = [file_path, line, diff_date, week, DIV, treatment_time, rec_date,
               layout_name, well, drug, dose, condition, *features]
        results.append((id_, row))

    return results, None


def parse_fused_condition(raw):
    """Split a "DRUG_DOSEuM" layout cell (e.g. 'LTX5_0.1uM') into (drug, dose).

    Used for single-compound follow-up datasets (e.g. LTX5) where the layout
    cell's dose has an explicit unit suffix and underscore separator, unlike
    the "DRUG (DOSE)" convention in parse_condition. Plain values with no
    concentration (e.g. 'DMSO', 'NSC') are conditions in their own right -
    the whole label becomes `drug` with dose '' (not a numeric dose, and not
    None either - `dose` gets cast via na(..., dtype=str) downstream, which
    would turn a None into the literal string "None").
    """
    raw = str(raw).strip()
    m = re.match(r'^([A-Za-z0-9]+)_([\d.]+)uM$', raw)
    if m:
        return m.group(1), m.group(2)
    return raw, ''


def process_fused_drug_recording_file(main_path, layouts_path, drug, diff, week, file, burst_params=None):
    """Process one recording file from a single-compound follow-up experiment
    (e.g. LTX5): one plate per cell line, `drug` is the experiment's primary
    compound, and each well's condition varies by *dose* (or a reference
    comparator drug / vehicle), not by drug identity - unlike
    process_multidrug_recording_file where the whole point is drug varying
    per well.

    Two conventions this handles that the other processors don't:
    - The filename's treatment-time field has `drug` fused onto it for every
      non-baseline timepoint (e.g. 'LTX51h' -> drug 'LTX5', treatment_time
      '1h'; 'LTX524h' -> '24h'; 'baseline' is left bare, since it predates
      any treatment). A prefix that doesn't match `drug` or 'baseline' is
      kept as-is rather than guessed at, so an unexpected value surfaces
      instead of being silently mislabeled.
    - Layout files are per-cell-line `.ods` (LibreOffice/OpenDocument)
      workbooks with a single, unnamed-convention sheet - unlike the `.xlsx`
      workbooks with one sheet per diff-batch used elsewhere. Reading `.ods`
      requires the `odfpy` package (`pd.read_excel(..., engine='odf')`).
      Because a follow-up experiment like this only spans one differentiation,
      there's no per-batch sheet to look up - the first (only) sheet is used
      directly. The layout filename is reconstructed from the recording
      file's own parsed fields (`{exp}_{line}_{letter}_{diffdate}.ods`)
      rather than assumed to be just `{line}.ods`.
    Each well's condition is parsed with parse_fused_condition (DRUG_DOSEuM),
    not parse_condition (DRUG (DOSE)) - the layout can list a reference
    comparator drug (e.g. 'LTX1_10uM') alongside the primary compound, DMSO/
    vehicle wells, and excluded wells ('X'/'Y').

    Filename convention: exp_line_DIV_letter_treatmenttimeFused_diffdate_recdate_*
    (8 fields; `letter` is just a per-plate id here, not necessarily
    alphabetic).

    This is the unit of parallelism for building the conditions table - each
    file is independent, so it can run inside a ProcessPoolExecutor worker.
    Must stay a plain module-level function (not a notebook closure) so it
    can be pickled and sent to worker processes.

    burst_params: optional dict of get_pop_bursts kwargs, forwarded to
    get_features for every well in this file - see get_features/
    DEFAULT_BURST_PARAMS.

    Returns:
        (results, skipped): results is a list of (id_, row) tuples; skipped
        is the filename if it didn't match the expected 8-field naming
        convention or had no matching layout file, otherwise None.
    """
    sp_file = file.split('_')
    if len(sp_file) != 8:
        return [], file

    exp, line, DIV, letter, tt_raw, diff_date, rec_date, _ = sp_file
    if tt_raw == 'baseline':
        treatment_time = 'baseline'
    elif tt_raw.startswith(drug):
        treatment_time = tt_raw[len(drug):]
    else:
        treatment_time = tt_raw

    layout_name = f'{exp}_{line}_{letter}_{diff_date}'
    try:
        layout_df = pd.read_excel(os.path.join(layouts_path, layout_name + '.ods'),
                                   sheet_name=0, engine='odf')
    except Exception as e:
        print(f"layout read error ({file}):", e)
        return [], file

    layout = {}
    for _, row_ in layout_df.iterrows():
        row_label = row_.iloc[0]
        for col_idx, value in enumerate(row_.iloc[1:], start=1):
            layout[f"{row_label}{col_idx}"] = value

    file_path = os.path.join(main_path, diff, week, file)
    spikes, bursts, netbursts = read_ephys(file_path)

    results = []
    for well, condition in layout.items():
        if condition in ('X', 'Y'):
            continue
        cond_drug, dose = parse_fused_condition(condition)
        id_ = f"{rec_date}_{well}_{line}_{treatment_time}_{DIV}_{letter}"
        features = get_features(well, spikes, bursts, netbursts, burst_params=burst_params)
        row = [file_path, line, diff_date, week, DIV, treatment_time, rec_date,
               layout_name, well, cond_drug, dose, condition, *features]
        results.append((id_, row))

    return results, None


def process_plate_recording_file(main_path, diff, subdir, file, burst_params=None):
    """Process one recording file from a layout-less, one-condition-per-plate
    experiment (e.g. an ASO time-course): every well on the plate shares the
    same cell line and treatment condition, both already encoded in the
    filename, so - unlike process_recording_file/process_multidrug_recording_file/
    process_fused_drug_recording_file - there is no external layout file to
    read. The wells to process are simply whatever 'Well Label' values appear
    in the recording's own spikes CSV.

    Filename convention: exp_line_DIV_plate_timepoint_daysaftertreatment_diffdate_recdate_suffix
    (9 fields, same shape as process_recording_file's). `timepoint` is a
    condition label (e.g. 'baseline'/'ASO') and
    `daysaftertreatment` is the day count as a string (e.g. '0'/'4'/'7') -
    together they distinguish the plate's timepoints within one diff, since
    unlike process_recording_file's 'drug' field, there's no separate compound
    identity here.

    This is the unit of parallelism for building the conditions table - each
    file is independent, so it can run inside a ProcessPoolExecutor worker.
    Must stay a plain module-level function (not a notebook closure) so it can
    be pickled and sent to worker processes.

    burst_params: optional dict of get_pop_bursts kwargs, forwarded to
    get_features for every well in this file - see get_features/
    DEFAULT_BURST_PARAMS.

    Returns:
        (results, skipped): results is a list of (id_, row) tuples; skipped is
        the filename if it didn't match the expected 9-field naming
        convention, otherwise None.
    """
    sp_file = file.split('_')
    if len(sp_file) != 9:
        return [], file

    exp, line, DIV, plate, timepoint, days_after, diff_date, rec_date, _ = sp_file

    file_path = os.path.join(main_path, diff, subdir, file)
    spikes, bursts, netbursts = read_ephys(file_path)

    results = []
    for well in sorted(spikes['Well Label'].unique()):
        id_ = f"{diff_date}_{rec_date}_{well}_{plate}_{line}_{timepoint}_{days_after}_{DIV}"
        features = get_features(well, spikes, bursts, netbursts, burst_params=burst_params)
        row = [file_path, line, plate, DIV, timepoint, days_after, diff_date, rec_date,
               well, *features]
        results.append((id_, row))

    return results, None


def parse_layout_condition(raw):
    """Split a "CellLine_Construct_Dose" layout cell (e.g. 'P22-5_ASO4_1µM',
    'P22-5_None_None') into (cell_line, construct, dose).

    Used for process_plate_layout_recording_file's per-well layout, where
    every field is always present and underscore-joined (unlike
    parse_condition's "DRUG (DOSE)" or parse_fused_condition's
    "DRUG_DOSEuM"). The literal string 'None' in either the construct or
    dose position means that field doesn't apply to this well (e.g. an
    untreated/baseline well) and is returned as Python None rather than the
    string.
    """
    parts = str(raw).strip().split('_')
    if len(parts) != 3:
        return raw, None, None
    cell_line, construct, dose = parts
    return cell_line, (None if construct == 'None' else construct), (None if dose == 'None' else dose)


def process_plate_layout_recording_file(main_path, layouts_path, diff, subdir, file, burst_params=None):
    """Process one recording file from a layout-backed, one-condition-per-well
    experiment (e.g. an ASO time-course where each plate has its own
    per-well layout workbook): like process_plate_recording_file, one file =
    one plate, but here individual wells within the plate can carry
    different conditions (construct/dose) or be excluded, so a layout is
    required.

    Filename convention: exp_line_DIV_plate_timepoint_daysaftertreatment_diffdate_recdate_suffix
    (9 fields, same shape as process_plate_recording_file's) - `plate` is
    'Plate<N>' in the recording filename but bare '<N>' in the layout
    filename below.

    Layout files live at layouts_path/Diff<diffdate>/, one workbook per
    plate, named like the plate's own *baseline* recording but with the
    bare plate number (exp_cellgroup_DIV_N_baseline_0_diffdate_recdate.xlsx)
    - located here by matching just the plate number and diff date, since a
    day4/day7 recording's own DIV/rec_date differ from the (always-baseline)
    ones in the layout's filename. Each workbook has one sheet per
    timepoint ('baseline', '<days>days'), read with read_layout_single.
    Wells marked 'X' (or 'Y') in the layout are excluded entirely; every
    other cell is a "CellLine_Construct_Dose" string parsed with
    parse_layout_condition.

    This is the unit of parallelism for building the conditions table - each
    file is independent, so it can run inside a ProcessPoolExecutor worker.
    Must stay a plain module-level function (not a notebook closure) so it
    can be pickled and sent to worker processes.

    burst_params: optional dict of get_pop_bursts kwargs, forwarded to
    get_features for every well in this file - see get_features/
    DEFAULT_BURST_PARAMS.

    Returns:
        (results, skipped): results is a list of (id_, row) tuples; skipped
        is the filename if it didn't match the expected 9-field naming
        convention or had no matching layout file, otherwise None.
    """
    sp_file = file.split('_')
    if len(sp_file) != 9:
        return [], file

    exp, line, DIV, plate, timepoint, days_after, diff_date, rec_date, _ = sp_file
    plate_num = plate[len('Plate'):] if plate.startswith('Plate') else plate

    diff_layout_dir = os.path.join(layouts_path, f'Diff{diff_date}')
    candidates = []
    if os.path.isdir(diff_layout_dir):
        for name in os.listdir(diff_layout_dir):
            stem, ext = os.path.splitext(name)
            if ext.lower() != '.xlsx':
                continue
            fields = stem.split('_')
            if len(fields) == 8 and fields[3] == plate_num:
                candidates.append(name)

    if len(candidates) != 1:
        print(f"layout match error ({file}): found {len(candidates)} candidates in {diff_layout_dir}")
        return [], file

    layout_path = os.path.join(diff_layout_dir, candidates[0])
    sheet_name = 'baseline' if timepoint == 'baseline' else f'{days_after}days'
    try:
        layout = read_layout_single(layout_path, sheet_name)
    except Exception as e:
        print(f"layout read error ({file}):", e)
        return [], file

    file_path = os.path.join(main_path, diff, subdir, file)
    spikes, bursts, netbursts = read_ephys(file_path)

    results = []
    for well, raw_condition in layout.items():
        condition = str(raw_condition).strip()
        if condition in ('X', 'Y'):
            continue
        cell_line, construct, dose = parse_layout_condition(condition)
        id_ = f"{diff_date}_{rec_date}_{well}_{plate}_{line}_{timepoint}_{days_after}_{DIV}"
        features = get_features(well, spikes, bursts, netbursts, burst_params=burst_params)
        row = [file_path, layout_path, condition, cell_line, construct, dose, line, plate, DIV,
               timepoint, days_after, diff_date, rec_date, well, *features]
        results.append((id_, row))

    return results, None


# Features suffixed "(MCS)" come straight from the Multi Channel Systems
# bursts.csv/networkbursts.csv exports (burst_features/netburst_features);
# features suffixed "(custom)" come from this package's own from-scratch
# population burst detector (get_new_bursts -> get_pop_bursts ->
# pop_burst_detection in ephys_proc.py) instead of MCS's native detection.
feature_names = ['ISI [s]', 'CV_ISI', 'FR [Hz]',
                 'IBI (MCS) [s]', 'CV_IBI (MCS)', 'Burst_Rate (MCS) [Hz]',
                 'BD (MCS) [s]', 'CV_BD (MCS)', 'In-burst_Rate (MCS) [Hz]',
                 'NIBI (MCS) [s]', 'CV_NIBI (MCS)', 'NB_Rate (MCS) [Hz]', 'NBD (MCS) [s]', 'CV_NBD (MCS)',
                 'NIBI (custom) [s]', 'CV_NIBI (custom)', 'NBD (custom) [s]', 'CV_NBD (custom)',
                 'In-NB_Rate (MCS) [Hz]', 'Random_Spikes (MCS) [%]', 'Total_Burst_Time (MCS)', 'smbr (MCS)']
