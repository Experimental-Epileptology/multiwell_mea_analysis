import os

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


def read_ephys(path):
    split_file = path.split('_')
    split_file[-1] = 'networkbursts.csv'
    netburst_file = '_'.join(split_file)
    split_file[-1] = 'spikes.csv'
    spike_file = '_'.join(split_file)
    split_file[-1] = 'bursts.csv'
    burst_file = '_'.join(split_file)
    spikes = pd.read_csv(spike_file)
    netbursts = pd.read_csv(netburst_file)
    bursts = pd.read_csv(burst_file)
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


def get_new_bursts(st, gid):
    st_flat = np.asarray(st).ravel()
    gid_flat = np.asarray(gid).ravel()

    b_times = get_pop_bursts(st_flat, gid_flat)

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


def get_features(well_name, spikes, bursts, netbursts):
    """ Collect features for a well """
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
        mean_ibi_new, cv_ibi_new, mean_burst_duration_new, cv_burst_duration_new = get_new_bursts(st, gid)
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


def process_recording_file(main_path, layouts_path, diff, week, file):
    """Process one recording file: parse its name, load its well layout, read its
    spikes/bursts/netbursts CSVs once, and compute features for every well.

    This is the unit of parallelism for building the conditions table - each
    file is independent, so it can run inside a ProcessPoolExecutor worker.
    Must stay a plain module-level function (not a notebook closure) so it can
    be pickled and sent to worker processes.

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
        features = get_features(well, spikes, bursts, netbursts)
        row = [file_path, line, diff_date, batch, week, DIV, treatment_time, rec_date,
               layout_name, well, drug, LTXcondition, *features]
        results.append((id_, row))

    return results, None


feature_names = ['ISI', 'CV_ISI', 'FR',
                 'IBI', 'CV_IBI', 'Burst_Rate',
                 'BD', 'CV_BD', 'In-burst_Rate',
                 'NIBI', 'CV NIBI', 'NB_Rate', 'NBD', 'CV NBD',
                 'NIBInew', 'CV NIBInew', 'NBDnew', 'CV NBDnew',
                 'In-NB_Rate', 'Random_Spikes', 'Total_Burst_Time', 'smbr']
