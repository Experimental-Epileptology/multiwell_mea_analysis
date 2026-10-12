# multiwell-mea-analysis

Shared toolkit for multiwell MEA (multielectrode array) burst detection, feature
extraction, and interactive QC widgets. Used across multiple iPSC-neuron
electrophysiology projects so improvements and bug fixes land everywhere at once.

## Contents

- `ephys_proc.py` — burst detection primitives (MI_bursts, population burst
  detection, auto ISI threshold estimation).
- `MEA_features.py` — per-well feature extraction (spike/burst/network-burst
  features), MAD-based real-burst filtering, and the batch file-processing
  pipeline used to build conditions tables.
- `burst_qc_widget.py` — interactive Jupyter widget for tuning burst-detection
  parameters well by well and accepting/rejecting wells.

## Installation

In the conda env for a given project:

```bash
pip install -e /path/to/MEA-analysis-base
```

This is an editable install: changes made to this repo take effect immediately
in every project that installed it this way, without reinstalling. Pull latest
with `git pull` from this repo to update.

Then in notebooks:

```python
from multiwell_mea_analysis.ephys_proc import pop_burst_detection
from multiwell_mea_analysis.MEA_features import get_features, get_pop_bursts
from multiwell_mea_analysis.burst_qc_widget import create_burst_qc_widget
```

## Credits 

Original code is written by Oleg Vinogradov. Claude code was used to refactor and improve the code. 