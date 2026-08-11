""" Helpers for burst detection """

from scipy import stats
import sys
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
from numba import jit
from scipy.ndimage import uniform_filter1d

na = np.array


def auto_maxISI(st, gid, n_bins=50):
    """Estimate maxISI threshold using the log-ISI valley method.

    Pools per-electrode ISIs, builds a log-ISI histogram, and finds the
    valley between the intra-burst (short ISI) and inter-burst (long ISI)
    peaks.  Falls back to the 75th-percentile ISI when the distribution
    looks unimodal (no clear valley).

    Parameters
    ----------
    st : array-like
        Spike times in **milliseconds**.
    gid : array-like
        Electrode / neuron IDs corresponding to each spike time.
    n_bins : int
        Number of histogram bins in log-ISI space.

    Returns
    -------
    threshold_ms : float
        Estimated maxISI in milliseconds.
    """
    st = np.asarray(st, dtype=float)
    gid = np.asarray(gid)

    # Pool per-electrode ISIs
    isis = []
    for uid in np.unique(gid):
        st_ch = np.sort(st[gid == uid])
        ch_isi = np.diff(st_ch)
        if len(ch_isi) > 0:
            isis.append(ch_isi)

    if not isis:
        return np.percentile(np.diff(np.sort(st)), 75) if len(st) > 1 else 100.0

    all_isi = np.concatenate(isis)
    all_isi = all_isi[all_isi > 0]

    # if len(all_isi) < 10:
    #     return float(np.percentile(all_isi, 10))

    # log_isi = np.log10(all_isi)
    # counts, bin_edges = np.histogram(log_isi, bins=n_bins)

    # # Smooth histogram to reduce noise
    # smoothed = uniform_filter1d(counts.astype(float), size=4)

    # # Find peaks (modes) in the smoothed histogram
    # peaks, _ = find_peaks(smoothed, prominence=smoothed.max() * 0.05)

    # if len(peaks) >= 2:
    #     # Take the two tallest peaks and find the valley between them
    #     peak_heights = smoothed[peaks]
    #     top2_idx = np.argsort(peak_heights)[-2:]
    #     p_left = peaks[top2_idx].min()
    #     p_right = peaks[top2_idx].max()

    #     valley_region = smoothed[p_left:p_right + 1]
    #     valley_offset = np.argmin(valley_region)
    #     valley_bin = p_left + valley_offset

    #     threshold_log = (bin_edges[valley_bin] + bin_edges[valley_bin + 1]) / 2
    #     threshold_ms = 10 ** threshold_log
    # else:
    #     # Unimodal — fall back to 75th percentile
    #     threshold_ms = float(np.percentile(all_isi, 10))

    threshold_ms = float(np.percentile(all_isi, 50))
    # Clamp to a sensible range
    threshold_ms = float(np.clip(threshold_ms, 20.0, 200.0))
    return threshold_ms


def bimodality_index(x):
    """computes the sample bimodality index based on
    https://en.wikipedia.org/wiki/Multimodal_distribution """
    gamma = skew(x)
    kappa = kurtosis(x)
    n = len(x)
    return ((gamma**2)+1) / (kappa + (3*(n-1)**2) / ((n-2)*(n-3)))


@jit(nopython=True)
def detect_onsets(burst_vectors, thr_burst_vector, min_active_electrodes=5):
    """Detect the start and end of the network burst"""
    pop_bursts = []
    df_vec = np.diff(thr_burst_vector)
    start = -1
    for i, k in enumerate(df_vec):
        if k > 0:
            start = i
        elif k < 0 and start >= 0:
            end = i
            pop_bursts.append([start, end])
    pop_bursts_final = []
    for burst in pop_bursts:
        if np.max(burst_vectors[burst[0]:burst[1], 0]) > min_active_electrodes:
            pop_bursts_final.append(burst)
    return pop_bursts_final


def fixed_bursted_detection(st, gid, sc):
    bi = bimodality_index(sc)

    if np.median(sc) <= 3 and bi > 0.5:
        isi_thr = min(max(np.mean(np.diff(st)), 10), 500)
        bursts = na(pop_burst_detection(na(st), (gid),
                                        maxISIstart=isi_thr,
                                        min_active_electrodes=3,
                                        )) / 1000
    else:
        bursts = []

    return bursts


def _filter_bursts_by_activity(st, pop_bursts, n_std=5, bin_size_ms=500):
    """Discard bursts whose spike density does not exceed baseline by n_std.

    Baseline is estimated from fixed-size bins outside all detected bursts.
    Each burst's spike count is normalised to the same bin_size_ms so the
    comparison is on the same scale.

    Parameters
    ----------
    st : np.ndarray
        All spike times in milliseconds (need not be sorted).
    pop_bursts : list of [start_ms, end_ms]
        Candidate bursts from detect_onsets.
    n_std : float
        Number of standard deviations above baseline required to keep a burst.
    bin_size_ms : float
        Bin width used to estimate baseline spike-count statistics.

    Returns
    -------
    kept : list of [start_ms, end_ms]
    """
    if not pop_bursts:
        return pop_bursts

    st = np.sort(np.asarray(st, dtype=float))
    t_max = st[-1]

    bins = np.arange(0, t_max + bin_size_ms, bin_size_ms)
    counts, _ = np.histogram(st, bins=bins)
    bin_starts = bins[:-1]

    # Mark bins that overlap with any burst
    non_burst = np.ones(len(counts), dtype=bool)
    for b in pop_bursts:
        non_burst &= ~((bin_starts + bin_size_ms > b[0]) & (bin_starts < b[1]))

    if non_burst.sum() < 3:
        return pop_bursts  # not enough baseline — keep all

    baseline_mean = np.mean(counts[non_burst])
    baseline_std = np.std(counts[non_burst])
    threshold = baseline_mean + n_std * baseline_std

    kept = []
    for burst in pop_bursts:
        n_spikes = np.sum((st >= burst[0]) & (st <= burst[1]))
        dur = max(burst[1] - burst[0], 1.0)
        # Normalise to bin_size_ms for fair comparison with baseline counts
        normalised = n_spikes * bin_size_ms / dur
        if normalised > threshold:
            kept.append(burst)

    return kept


def pop_burst_detection(st, gid,
                maxISIstart=100,
                maxISIb=250,
                minBdur=50,
                minIBI=500,
                minSburst=6,
                onset_threshold=3,
                min_active_electrodes=10,
                        ):
    """Detect network burst based on the detection of bursts in
    each electrode and further thresholding

    Args:
        st (list): spike times (in ms)
        gid (list): neuron ids
    Returns:
        pop_bursts_final (list): list of tuples with the start and end of each burst
    """
    st = na(st)
    gid = na(gid)

    if len(st) == 0:
        return []

    gid_ids = np.unique(gid)
    # +1 so the last spike time is a valid index
    n_bins = int(np.max(st)) + 1
    burst_vectors = np.ones(shape=(n_bins, 1))
    for i, u_id in enumerate(gid_ids):
        st_ = st[gid == u_id]
        bursts = MI_bursts(st_,
                maxISIstart=maxISIstart,
                maxISIb=maxISIb,
                minBdur=minBdur,
                minIBI=minIBI,
                minSburst=minSburst)
        for burst in bursts:
            start = max(0, int(burst[0]) - 5)
            end = min(n_bins, int(burst[1]) + 5)
            burst_vectors[start:end, 0] += 1

    # Use sorted population spike train so np.diff gives valid ISIs
    isi_thr = min(max(np.mean(np.diff(np.sort(st))), 1), 500)
    bursts = MI_bursts(st,
            maxISIstart=isi_thr,
            maxISIb=isi_thr,
            minBdur=50,
            minIBI=800,
            minSburst=200)
    for burst in bursts:
        start = int(burst[0])
        end = min(n_bins, int(burst[1]))
        burst_vectors[start:end, 0] += 10

    pop_bursts_final = detect_onsets(
        burst_vectors,
        na(burst_vectors[:, 0] > onset_threshold, dtype=int),
        min_active_electrodes)
    # pop_bursts_final = _filter_bursts_by_activity(st, pop_bursts_final)
    return pop_bursts_final


@jit(nopython=True)
def find_burstlets(spikes, r_spikes, isi_pop,
              maxISIstart=4.5,
              maxISIb=4.5,
              minSburst=100):
    """
    Helper to find burstlets
    Args:
        spikes (arr): spike times
        r_spikes(arr): rounded spike times
        isi_pop(arr): isi
    Returns:
        burst_ (list of tuples): Burst start, burst end
    """
    b_spikes = None
    burst_ = []
    b_start = 0
    b_size = 0
    for i, s in enumerate(spikes[:-1]):

        if isi_pop[i] < maxISIstart and b_start == 0:
            b_size = 0
            b_start += 1
            b_spikes = s

        elif isi_pop[i] <= maxISIb and b_start > 0:
            b_start += 1
            b_size += 1

        elif b_start >= minSburst:
            burst_.append((b_spikes, s))
            b_spikes = None
            b_size = 0
            b_start = 0

        else:
            b_spikes = None
            b_size = 0
            b_start = 0

    return burst_


def MI_bursts(st,
              maxISIstart=4.5,
              maxISIb=4.5,
              minBdur=40,
              minIBI=40,
              minSburst=50):
    """Min Interval method [1,2] for burst detection
    Optimized version from 03.03.20
    OV

    Args:
            [1] spike times (list) (single neuron or population) (ms)
            [2] (float) max ISI at start of burst (ms)
            [3] (float) max ISI in burst (ms)
            [4] (float) min burst duration (ms)
            [5] (float) min inter-burst interval (ms)
            [6] (float) min number of spikes in a burst
    Returns:
            burst (list of tuples): burst start, burst end
    [1] Nex Technologies. NeuroExplorer Manual. Nex Technologies, 2014
    [2] Cotterill, E., and Eglen, S.J. (2018). Burst detection methods. ArXiv:1802.01287.
    """
    spikes = np.sort(st)
    r_spikes = np.round(spikes, -1)
    isi_pop = np.diff(spikes)
    burst_ = find_burstlets(spikes, r_spikes, isi_pop, maxISIstart, maxISIb, minSburst)
    bursts = []
    if burst_:
        bursts.append(burst_[0])
        for i, b in enumerate(burst_[1:]):
            if b[1]-b[0] >= minBdur and b[0]-bursts[-1][1] >= minIBI:
                bursts.append(b)
            elif b[0]-bursts[-1][1] <= minIBI:
                bursts[-1] = (bursts[-1][0], b[1])

    return bursts
