"""Interactive QC widget for tuning MI_bursts parameters well by well."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import clear_output, display

from .ephys_proc import auto_maxISI
from .MEA_features import gid_to_numbers, read_ephys, get_pop_bursts

na = np.array

WELL_NAMES = [
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6',
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
    'D1', 'D2', 'D3', 'D4', 'D5', 'D6',
]

PARAM_SAMPLE_SIZE = 30


def normalize_b_times(b_times_raw):
    """Coerce MI_bursts output into an (N, 2) array of [start, end]."""
    if b_times_raw is None:
        return np.empty((0, 2))
    arr = np.asarray(b_times_raw)
    if arr.size == 0:
        return np.empty((0, 2))
    arr = np.squeeze(arr)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 2:
        return arr.T
    if arr.ndim == 1 and len(arr) % 2 == 0:
        return arr.reshape(-1, 2)
    raise ValueError(f"Unexpected MI_bursts output shape: {arr.shape}")

def create_burst_qc_widget(df):
    """
    Create and return an interactive burst QC widget.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'file_path' and 'well' columns. The widget writes
        'accepted' and 'burst parameters' back into this DataFrame in-place.

    Returns
    -------
    ipywidgets.VBox
    """
    if 'accepted' not in df.columns:
        # Default True, not unlabeled: rows already reaching this df have
        # already survived any upstream exclusion (e.g. a layout's 'X'
        # wells), so "not yet visually reviewed" should read as accepted
        # until a Reject click says otherwise, not as missing/NaN.
        df['accepted'] = True
    if 'burst parameters' not in df.columns:
        df['burst parameters'] = ''

    unique_files = df['file_path'].drop_duplicates().tolist()
    ephys_cache = {}

    # pre-draw 30 random bursty samples at widget creation
    if 'bursty' in df.columns:
        candidates = df[df['bursty'].fillna(0).astype(bool)]
    else:
        candidates = df
    candidates = candidates[['file_path', 'well']].dropna()
    if len(candidates) == 0:
        candidates = df[['file_path', 'well']].dropna()
    _sampled = candidates.sample(n=min(PARAM_SAMPLE_SIZE, len(candidates)), random_state=None)
    _param_samples = list(zip(_sampled['file_path'], _sampled['well']))

    _state = {'mode': 'QC Mode'}

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def load_ephys_by_path(file_path):
        if file_path not in ephys_cache:
            ephys_cache[file_path] = read_ephys(file_path)
        return ephys_cache[file_path]

    def wells_for_file(file_path):
        """Wells to offer for this file: df's own 'well' rows for it (so a
        layout that dropped 'X'-marked wells upstream is respected here too),
        falling back to the full plate if df has none (e.g. no 'well' column,
        or this file/df wasn't built from a layout)."""
        if 'well' in df.columns:
            wells = sorted(df.loc[df['file_path'] == file_path, 'well'].dropna().unique().tolist())
            if wells:
                return wells
        return list(WELL_NAMES)

    def _mask(file_path, well_name):
        if 'well' in df.columns:
            return (df['file_path'] == file_path) & (df['well'] == well_name)
        return df['file_path'] == file_path

    def get_accept_value(file_index, well_name):
        fp = unique_files[file_index]
        vals = df.loc[_mask(fp, well_name), 'accepted']
        return vals.iloc[0] if len(vals) > 0 else np.nan

    def set_accept_value(file_index, well_name, value):
        df.loc[_mask(unique_files[file_index], well_name), 'accepted'] = bool(value)

    def set_burst_params(file_index, well_name, params):
        df.loc[_mask(unique_files[file_index], well_name), 'burst parameters'] = str(params)

    def clear_burst_params(file_index, well_name):
        df.loc[_mask(unique_files[file_index], well_name), 'burst parameters'] = ''

    def get_current_params():
        return {
            'maxISIstart': maxISIstart_slider.value,
            'maxISIb': maxISIb_slider.value,
            'minBdur': minBdur_slider.value,
            'minIBI': minIBI_slider.value,
            'minSburst': minSburst_slider.value,
            'onset_threshold': onset_threshold_slider.value,
            'min_active_electrodes': min_active_electrodes_slider.value,
            'mad_multiplier': mad_multiplier_slider.value,
        }

    # ------------------------------------------------------------------
    # Shared parameter + viz sliders
    # ------------------------------------------------------------------

    maxISIstart_slider = widgets.FloatSlider(
        value=100, min=1, max=500, step=1, description='maxISIstart',
        continuous_update=False, readout_format='.0f',
        layout=widgets.Layout(width='320px'))
    maxISIb_slider = widgets.FloatSlider(
        value=250, min=1, max=1000, step=1, description='maxISIb',
        continuous_update=False, readout_format='.0f',
        layout=widgets.Layout(width='320px'))
    minBdur_slider = widgets.FloatSlider(
        value=20, min=1, max=500, step=1, description='minBdur',
        continuous_update=False, readout_format='.0f',
        layout=widgets.Layout(width='320px'))
    minIBI_slider = widgets.FloatSlider(
        value=20, min=1, max=2000, step=1, description='minIBI',
        continuous_update=False, readout_format='.0f',
        layout=widgets.Layout(width='320px'))
    minSburst_slider = widgets.IntSlider(
        value=5, min=1, max=100, step=1, description='minSburst',
        continuous_update=False, layout=widgets.Layout(width='320px'))
    onset_threshold_slider = widgets.IntSlider(
        value=3, min=1, max=20, step=1, description='onset_thr',
        continuous_update=False, layout=widgets.Layout(width='320px'))
    min_active_electrodes_slider = widgets.IntSlider(
        value=6, min=1, max=50, step=1, description='min active el.',
        continuous_update=False, layout=widgets.Layout(width='320px'))
    mad_multiplier_slider = widgets.FloatSlider(
        value=1.5, min=0.1, max=10.0, step=0.1, description='MAD mult.',
        continuous_update=False, readout_format='.1f',
        layout=widgets.Layout(width='320px'))
    bin_slider = widgets.FloatSlider(
        value=0.05, min=0.01, max=1.0, step=0.01,
        description='Bin (s):', continuous_update=False, readout_format='.2f',
        layout=widgets.Layout(width='320px'))
    alpha_slider = widgets.FloatSlider(
        value=0.10, min=0.01, max=1.0, step=0.01,
        description='Alpha:', continuous_update=False, readout_format='.2f',
        layout=widgets.Layout(width='320px'))
    show_native_cb = widgets.Checkbox(value=False, description='Show native nbursts')
    show_mi_cb = widgets.Checkbox(value=True, description='Show pop bursts')
    default_btn = widgets.Button(description='Defaults')
    auto_isi_btn = widgets.Button(description='Auto ISI', button_style='warning')

    shared_param_box = widgets.VBox([
        widgets.HBox([maxISIstart_slider, maxISIb_slider]),
        widgets.HBox([minBdur_slider, minIBI_slider, minSburst_slider]),
        widgets.HBox([onset_threshold_slider, min_active_electrodes_slider]),
        widgets.HBox([mad_multiplier_slider]),
        widgets.HBox([bin_slider, alpha_slider]),
        widgets.HBox([show_native_cb, show_mi_cb, default_btn, auto_isi_btn]),
    ])

    # ------------------------------------------------------------------
    # Core plot function
    # ------------------------------------------------------------------

    output = widgets.Output()

    # Created once and reused for every redraw (clear + redraw onto the same
    # Axes), rather than a fresh plt.subplots() per call. Redraws happen from
    # ipywidgets callbacks, not real cell executions, so IPython's usual
    # "close figures after the cell runs" hook never fires for them; with the
    # interactive ipympl/'widget' backend specifically, creating a brand-new
    # Figure inside a callback like that is a known trigger for the canvas
    # failing to (re)attach as a live widget and silently falling back to a
    # flat/static render instead - a persistent Axes plus display(fig)
    # sidesteps that, and works the same way under the plain 'inline' backend.
    with plt.ioff():
        fig, ax = plt.subplots(1, 1, figsize=(5, 2), dpi=200)
    plt.close(fig)  # detach from pyplot's global state; we display it manually below

    def _plot(file_path, well_name, title_prefix=''):
        """Plot into the current output context (call inside `with output:`)."""
        spikes, bursts, nbursts = load_ephys_by_path(file_path)
        spikes_well = spikes[spikes['Well Label'] == well_name]
        nbursts_well = nbursts[nbursts['Well Label'] == well_name]

        ax.clear()
        base_title = f"{title_prefix}Well {well_name} | {os.path.basename(str(file_path))}"

        if len(spikes_well) == 0:
            ax.set_title(base_title, fontsize=9)
            ax.text(0.5, 0.5, 'No spikes found', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_ylabel("Channel")
            fig.tight_layout()
            display(fig)
            return

        st_p = np.squeeze(na(spikes_well.filter(items=['Timestamp [µs]'])) * 1e-6)
        gid_p = np.squeeze(na(spikes_well.filter(items=['Channel ID'])))
        gid_p = gid_to_numbers(gid_p)

        if len(nbursts_well) > 0:
            nb_start = np.squeeze(
                na(nbursts_well.filter(items=['Start timestamp [µs]'])) * 1e-6)
            nb_dur = np.squeeze(
                na(nbursts_well.filter(items=['Duration [µs]'])) * 1e-6)
            native_b_times = np.column_stack([nb_start, nb_start + nb_dur])
        else:
            native_b_times = np.empty((0, 2))

        mi_b_times = get_pop_bursts(st_p, gid_p, **get_current_params())

        ax.plot(st_p, gid_p, '|', alpha=alpha_slider.value, markersize=5, linewidth=3)

        if len(st_p) > 1 and np.max(st_p) > bin_slider.value:
            bins = np.arange(0, np.max(st_p) + bin_slider.value, bin_slider.value)
            sc_c, bin_e = np.histogram(st_p, bins)
            if np.max(sc_c) > 0:
                ax.plot(bin_e[1:], (8 * (sc_c / np.max(sc_c))) - 10, linewidth=0.5)

        if show_native_cb.value:
            for b in native_b_times:
                ax.axvspan(b[0], b[1], color='deepskyblue', alpha=0.18)
        if show_mi_cb.value:
            for b in mi_b_times:
                ax.axvspan(b[0], b[1], color='r', alpha=0.2)

        ax.set_ylabel("Channel")
        ax.set_title(f"{base_title} | MI bursts: {len(mi_b_times)}", fontsize=9)
        fig.tight_layout()
        display(fig)

    # ------------------------------------------------------------------
    # Single redraw dispatcher
    # ------------------------------------------------------------------

    def redraw(*args):
        if _state['mode'] == 'QC Mode':
            _redraw_qc()
        else:
            _redraw_param()

    # ==================================================================
    # QC MODE
    # ==================================================================

    file_slider = widgets.IntSlider(
        value=0, min=0, max=len(unique_files) - 1, step=1,
        description='File:', continuous_update=False,
        layout=widgets.Layout(width='500px'))
    _initial_wells = wells_for_file(unique_files[0]) if unique_files else list(WELL_NAMES)
    well_dropdown = widgets.Dropdown(
        options=_initial_wells, value=_initial_wells[0], description='Well:',
        layout=widgets.Layout(width='220px'))
    prev_file_btn = widgets.Button(description='← File')
    next_file_btn = widgets.Button(description='File →')
    prev_well_btn = widgets.Button(description='← Well')
    next_well_btn = widgets.Button(description='Well →')
    accept_btn = widgets.Button(description='Accept', button_style='success')
    reject_btn = widgets.Button(description='Reject', button_style='danger')
    status_html = widgets.HTML()

    def _redraw_qc():
        fp = unique_files[file_slider.value]
        val = get_accept_value(file_slider.value, well_dropdown.value)
        status_txt = ("unlabeled" if pd.isna(val)
                      else ("accepted" if bool(val) else "rejected"))
        with output:
            clear_output(wait=True)
            try:
                _plot(fp, well_dropdown.value,
                      title_prefix=f"File {file_slider.value} | {status_txt} | ")
            except Exception as e:
                print("Error while plotting:", e)
        status_html.value = f"<b>Status:</b> {status_txt}"

    def on_prev_file(_):
        if file_slider.value > file_slider.min:
            file_slider.value -= 1

    def on_next_file(_):
        if file_slider.value < file_slider.max:
            file_slider.value += 1

    def on_prev_well(_):
        opts = well_dropdown.options
        idx = opts.index(well_dropdown.value)
        if idx > 0:
            well_dropdown.value = opts[idx - 1]

    def on_next_well(_):
        opts = well_dropdown.options
        idx = opts.index(well_dropdown.value)
        if idx < len(opts) - 1:
            well_dropdown.value = opts[idx + 1]

    def on_file_change(change):
        """Refresh the well list for the newly selected file before redrawing,
        so QC mode only ever offers wells df actually has for this file.
        Reassigning .options alone resets .value to the first new option
        when the old value isn't in the new list (ipywidgets Selection
        behavior) - that itself fires well_dropdown's own redraw observer,
        so the explicit redraw() below is what actually repaints the plot
        for cases where the value happens to stay valid across files."""
        wells = wells_for_file(unique_files[change['new']])
        well_dropdown.options = wells
        redraw()

    def on_accept(_):
        set_accept_value(file_slider.value, well_dropdown.value, True)
        set_burst_params(file_slider.value, well_dropdown.value, get_current_params())
        redraw()

    def on_reject(_):
        set_accept_value(file_slider.value, well_dropdown.value, False)
        clear_burst_params(file_slider.value, well_dropdown.value)
        redraw()

    prev_file_btn.on_click(on_prev_file)
    next_file_btn.on_click(on_next_file)
    prev_well_btn.on_click(on_prev_well)
    next_well_btn.on_click(on_next_well)
    accept_btn.on_click(on_accept)
    reject_btn.on_click(on_reject)
    file_slider.observe(on_file_change, names='value')

    qc_controls = widgets.VBox([
        widgets.HBox([prev_file_btn, next_file_btn, file_slider]),
        widgets.HBox([prev_well_btn, next_well_btn, well_dropdown]),
        shared_param_box,
        widgets.HBox([accept_btn, reject_btn, status_html]),
    ])

    # ==================================================================
    # PARAMETER ID MODE
    # ==================================================================

    sample_slider = widgets.IntSlider(
        value=0, min=0, max=max(0, len(_param_samples) - 1), step=1,
        description='Sample:', continuous_update=False,
        layout=widgets.Layout(width='500px'))

    prev_sample_btn = widgets.Button(description='← Prev')
    next_sample_btn = widgets.Button(description='Next →')
    copy_params_btn = widgets.Button(description='Print params', button_style='info')

    def _redraw_param():
        if not _param_samples:
            with output:
                clear_output(wait=True)
                print("No samples available.")
            return
        fp, well = _param_samples[sample_slider.value]
        with output:
            clear_output(wait=True)
            try:
                _plot(fp, well,
                      title_prefix=f"Sample {sample_slider.value + 1}/{len(_param_samples)} | ")
            except Exception as e:
                print(f"Error while plotting: {e}")

    def on_prev_sample(_):
        if sample_slider.value > sample_slider.min:
            sample_slider.value -= 1

    def on_next_sample(_):
        if sample_slider.value < sample_slider.max:
            sample_slider.value += 1

    def on_copy_params(_):
        params = get_current_params()
        with output:
            print("\nCurrent MI_bursts parameters:")
            for k, v in params.items():
                print(f"  {k} = {v}")

    prev_sample_btn.on_click(on_prev_sample)
    next_sample_btn.on_click(on_next_sample)
    copy_params_btn.on_click(on_copy_params)

    param_id_controls = widgets.VBox([
        widgets.HBox([prev_sample_btn, next_sample_btn, sample_slider, copy_params_btn]),
        shared_param_box,
    ])

    # ==================================================================
    # Mode toggle + wiring
    # ==================================================================

    mode_toggle = widgets.ToggleButtons(
        options=['QC Mode', 'Parameter ID Mode'],
        value='QC Mode',
        layout=widgets.Layout(margin='0 0 8px 0'),
    )

    panel = widgets.VBox([qc_controls])

    def on_mode_change(change):
        _state['mode'] = change['new']
        if change['new'] == 'QC Mode':
            panel.children = [qc_controls]
        else:
            panel.children = [param_id_controls]
        redraw()

    mode_toggle.observe(on_mode_change, names='value')

    def on_default(_):
        maxISIstart_slider.value = 100
        maxISIb_slider.value = 250
        minBdur_slider.value = 50
        minIBI_slider.value = 500
        minSburst_slider.value = 6
        onset_threshold_slider.value = 3
        min_active_electrodes_slider.value = 10
        mad_multiplier_slider.value = 1.5

    def on_auto_isi(_):
        """Compute log-ISI valley threshold for the currently displayed well."""
        try:
            if _state['mode'] == 'QC Mode':
                fp = unique_files[file_slider.value]
                well = well_dropdown.value
            else:
                if not _param_samples:
                    return
                fp, well = _param_samples[sample_slider.value]

            spikes, _, _ = load_ephys_by_path(fp)
            spikes_well = spikes[spikes['Well Label'] == well]
            if len(spikes_well) == 0:
                return

            st_ms = np.squeeze(na(spikes_well.filter(items=['Timestamp [µs]'])) * 1e-3)
            gid = np.squeeze(na(spikes_well.filter(items=['Channel ID'])))

            thr = auto_maxISI(st_ms, gid)
            maxISIstart_slider.value = round(thr)
            maxISIb_slider.value = 200#float(np.clip(thr * 2.5, 20.0, 200.0))#np.clip(round(thr * 2.5))
        except Exception as e:
            print("Auto ISI error:", e)

    default_btn.on_click(on_default)
    auto_isi_btn.on_click(on_auto_isi)

    # All interactive controls call a single redraw() (file_slider is wired
    # separately via on_file_change, which refreshes well_dropdown's options
    # first, then redraws).
    for w in [well_dropdown, sample_slider,
              maxISIstart_slider, maxISIb_slider, minBdur_slider,
              minIBI_slider, minSburst_slider,
              onset_threshold_slider, min_active_electrodes_slider,
              mad_multiplier_slider,
              bin_slider, alpha_slider,
              show_native_cb, show_mi_cb]:
        w.observe(redraw, names='value')

    ui = widgets.VBox([mode_toggle, panel, output])
    redraw()
    return ui
