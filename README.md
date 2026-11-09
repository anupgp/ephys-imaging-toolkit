# ephys-imaging-toolkit

[![tests](https://github.com/anupgp/ephys-imaging-toolkit/actions/workflows/tests.yml/badge.svg)](https://github.com/anupgp/ephys-imaging-toolkit/actions/workflows/tests.yml)

A Python package that loads, checks and views patch-clamp electrophysiology and two-photon imaging data
from three proprietary formats (Axon `.abf`, Zeiss `.czi`, Olympus `.oib`/`.oif`), from local folders or
from the Box cloud API, through one function: `load_recording()`.

**At a glance**

| | |
|---|---|
| Size | about 15,000 lines of Python in 8 sub-packages |
| Tests | 131 pytest cases on synthetic data, run on every push by GitHub Actions |
| Cloud access | OAuth 2.0 client for the Box REST API: token refresh, owner-only token file, paginated listing, size-limited cache with SHA-1 change checks |
| Verification | 119 Zeiss files pixel-identical to the reference reader, stimulus times within 0.05 ms of recorded triggers, Box downloads byte-identical to local copies |
| Viewers | interactive Bokeh/Panel viewers for traces, image stacks and line-scan kymographs, plus static figure export |

**How it was built**

I built this package in Sep–Nov 2026 by directing an AI coding agent (Claude Code). I wrote the scope and
the agent's operating rules: stay inside the project folder, never expose credentials, and leave the legacy
code untouched. I reviewed the code and accepted each part only after it passed tests against real
instrument files. The agent first audited the lab's legacy analysis scripts. It found bugs that changed
results, such as stimulus times shifted by the acquisition software's pre-sweep holding period
(62.5 ms in a 4 s sweep). This package fixes them.

This repository contains code only. It holds no recorded data and no experiment sheets; all test data are synthetic.

---

## Quick start

```python
from ephys_imaging_data_loader import load_recording

# 1) any local file
ephys = load_recording("/path/to/data/20260322/c1/26322010.abf")

# 2) root + relative path — choose LOCAL or CLOUD with `mode`
ephys = load_recording("20260322/c1/26322010.abf", root="ephys_olympus2p", mode="local")
ephys = load_recording("20260322/c1/26322010.abf", root="ephys_olympus2p", mode="box")
ephys = load_recording("20260322/c1/26322010.abf", root="ephys_olympus2p", mode="auto")  # local, else Box

print(ephys.summary())
ephys.show()                                   # interactive viewer (notebook)
ephys.stimulus_onsets_s(sweep=0)               # decoded TTL stimulus times (s)

image = load_recording("20251227/c1/Image0004.oib", root="imaging_olympus2p")
image.uncaging_points                          # x_px, y_px of the uncaging targets
image.show()
```

The file names, dates and paths in these examples are made up. Point the settings file at your own data.

## Installation (once per computer)

```bash
# editable install: the package points at this folder, code changes apply immediately
python -m pip install -e ".[box,test]"

# machine-specific settings (git-ignored), then edit the folder paths inside
cp data_source_settings.example.toml data_source_settings.toml
```

In VS Code / Jupyter, select the your virtual environment Python kernel. No `sys.path` changes are needed.

### Viewers in VS Code

The interactive viewers use Panel, which inside VS Code draws via Jupyter widgets (`jupyter_bokeh`):

* `pip install jupyter_bokeh` into the kernel's environment (already done in your virtual environment).
* Allow VS Code to download third-party widget code: user setting
  `"jupyter.widgetScriptSources": ["jsdelivr.com", "unpkg.com"]`.
* **Code - OSS (Arch `code` package) only:** its `product.json` does not grant the Jupyter extension
  the "proposed APIs" that widgets need. Add `"enable-proposed-api": ["ms-toolsai.jupyter"]` to
  `~/.vscode-oss/argv.json` and restart the editor completely — or use JupyterLab
  (`jupyter lab`), where the viewers work without any extra setting.

## Choosing local or cloud: roots and modes

Data are addressed by a **root name** + **relative path**. Roots are defined in `data_source_settings.toml`:

```toml
[roots.ephys_olympus2p]
local_folder  = "/path/to/data/olympus2p_ephys"
box_folder_id = "123456789012"      # the number at the end of the Box folder URL
```

| `mode` | Where files come from |
|---|---|
| `"local"` | Local folders only. Works offline and never contacts Box. |
| `"box"` | Box only. Each file is downloaded **once** into the cache and re-downloaded only if it changed on Box (SHA-1 check). |
| `"auto"` | The local copy if it exists, otherwise Box. Box is contacted only when a file is missing locally. |
| *(omitted)* | `default_mode` from the settings file. |

The same analysis code therefore runs unchanged on a laptop with a partial local copy, on the lab
workstation, or entirely from Box.

### Box login

* The credentials file (`CLIENT_ID`, `CLIENT_SECRET`, optional `BOX_OAUTH_LOC`) lives **outside git**, at
  `~/.config/ephys_imaging_data_loader/box_credentials`. `box_credentials.example.env` is a template.
* Tokens are kept in `oauth_box_auth.json`, in a format shared with an older
  in-house Box login script, so both tools share one login. Avoid using both at
  the very same moment, because Box refresh tokens are single-use.
* If the saved login has expired, the first Box call prints an authorize URL and asks for the code.
* Downloads are cached in `~/.cache/ephys_imaging_data_loader` (limit set by `cache_size_limit_gb`;
  least-recently-used files are removed first). `open_data_source("box").cache.clear()` empties it.

## Choosing recordings from the experiment sheet

`ExperimentCatalog` reads the experiment sheet (CSV) (path in the `[catalog]` section of the settings;
the sheet is only read, never modified):

```python
from ephys_imaging_data_loader import ExperimentCatalog
catalog = ExperimentCatalog.from_settings(mode="auto")

catalog.browse()                     # interactive: experiment → date → neuron → dendrite → spine, table, buttons
rows = catalog.select(expdate=20260322, neuronid=1, dendriteid=1, spineid=0, isi=[30, 100])
recording = catalog.load_ephys(rows.index[0])          # the sheet's bad sweeps are already removed

catalog.show_traces_by_isi(20260322, neuronid=1, dendriteid=1, spineid=0)                # all ISIs
catalog.show_traces_by_isi(20260322, 1, 1, 0, isis=[30, 50, 100, 1000])                   # chosen ISIs
catalog.save_traces_by_isi_figure("fig.svg", 20260322, 1, 1, 0)
```

* **File names:** `ephysfile_prefix` + `ephysid`, with the run of `x` replaced by the zero-padded
  id (`26322xxx` + 7 → `26322007.abf`; `2027_02_18_xxxx` + 3 → `2027_02_18_0003.abf`). A prefix
  without `x` is a full name. When the prefix is empty, the classic Clampex name is **inferred** from
  the date (`24413008` for 2024-04-13, id 8; month codes 1–9, o, n, d) and flagged as inferred.
  Path: `<expdate>/c<neuronid>/<file>.abf`, with folder names matched ignoring case (`c1`/`C1`).
* **Bad sweeps:** `badsweeps` is a semicolon-separated list, counted from 1 like Clampex (checked
  against the files: `1;2;3;4` appears for 4-sweep files). Change with
  `badsweeps_numbering="from_0"`. Numbers that do not exist in a file are ignored with a warning.
* **`unselect`:** not used unless asked — `select(..., unselect=0)`, `unselect=[0, 1]`, or the
  "unselect filter" in the browser. Default: column ignored.
* **Traces by ISI:** one panel per ISI (ascending); thin = individual good sweeps, thick = mean, one
  colour per `group` (e.g. pre-apv / post-apv); time 0 = first stimulus; window 50 ms before the first
  pulse to ISI + 150 ms (change with `time_window_ms=(before, after)`); shared y-axis. The header shows
  date, neuron, dendrite, spine, location (region/area/side), clamp, stim power, temperature, access
  resistance … and the sweeps averaged per ISI, e.g. `pre-apv: 30 ms (3), 40 ms (3), … | post-apv: …`.
* **Finding the stimulus:** a protocol often drives several digital outputs (Olympus rig: DO0 = a
  200 ms gate during the test step, DO1 = uncaging; Zeiss rig: DO1 = microscope start, DO2 =
  stimulation). Every recorded channel is scanned for brief trigger / light pulses, and a digital
  output is **confirmed** when a recorded channel (e.g. `IN7`, `trigger2`) shows pulses at exactly its
  onsets. The stimulus is the output matching the sheet's ISI, preferring confirmed outputs and short
  pulses. Alternatives: `digital_output_bit=1`, `stimulus_channel="IN7"`, or `stimulus_channel="catalog"`
  (the sheet's `chan_stim`; if that channel shows no trigger pulses, the other channels are checked and
  the one matching the ISI is used, with a warning).
* **Checking the sheet:** `catalog.check_stimulus_channels(expdate=20260322)` loads the selected files
  and reports, per row, the channels that really carry trigger pulses, the identified stimulus, its
  IPI, and whether `chan_stim` and `isi` in the sheet agree with the file.
* **Single-recording viewer:** `recording.show()` title:
  `26322005.abf | example_protocol.pro | 3/3 sweeps | 2 pulses | IPI: 100 ms | CA3 | basal | female | stim power 30 | P40 | 32.0 °C`
  — pulses and IPI of the identified stimulus only; sheet details (area, side, sex, stim power, age
  from `dob`, temperature) are added when the file is in the sheet (also for plain `load_recording`).
  The line below says how the stimulus was identified, e.g. `digital output 1, confirmed by IN7`.

## Package layout

```
ephys_imaging_data_loader/
├── load_recording.py                  load_recording(), list_recording_files()  ← start here
├── recording_data_containers/
│   ├── ephys_recording.py             EphysRecording  (sweeps, channels, samples) + protocol + stimuli
│   └── image_recording.py             ImageRecording  (T, C, Z, Y, X) + calibration + events + targets
├── file_readers/                      local file → container (never downloads)
│   ├── axon_abf_file_reader.py
│   ├── axon_abf_protocol_decoder.py   Clampex epochs + digital (TTL) pulses
│   ├── olympus_oib_oif_file_reader.py
│   ├── zeiss_czi_file_reader.py
│   └── choose_reader_by_file_type.py
├── data_sources/                      (root, relative path) → file on local disk
│   ├── base_data_source.py            common interface
│   ├── local_folder_data_source.py
│   ├── box_cloud_data_source.py
│   ├── box_download_cache.py
│   ├── box_login_and_token_storage.py
│   ├── local_then_box_data_source.py  the "auto" mode
│   └── data_source_settings.py        reads data_source_settings.toml, open_data_source(mode)
├── experiment_catalog/                experiment sheet (CSV) → rows → recordings
│   ├── ephys_experiment_catalog.py    ExperimentCatalog: select, load_ephys/imaging, traces by ISI, browse()
│   ├── catalog_file_names.py          prefix + id → file name; badsweeps parsing
│   ├── stimulus_onset_detection.py    when was the stimulus delivered (digital output / recorded channel)
│   └── traces_by_isi.py               align and group the sweeps of one spine by ISI
└── interactive_viewers/               container → picture (never reads files)
    ├── ephys_trace_viewer.py          stacked channels, sweeps, mean, stimulus bands
    ├── image_stack_viewer.py          planes / projections, uncaging targets, scale bar
    ├── linescan_kymograph_viewer.py   kymograph + ΔF/F₀ trace + ZEN stim events
    ├── traces_by_isi_viewer.py        one panel per ISI (interactive + static figure)
    ├── ephys_recording_title.py       the '|'-separated title of recording.show()
    ├── experiment_catalog_browser.py  dropdowns + table + buttons (catalog.browse())
    ├── save_static_figure.py          PNG / SVG / PDF with matplotlib (no browser needed)
    └── display_helpers.py
```

The design principle: **where** a file is (data source), **how** it is read (file reader), **what** you get
(container) and **how** it is shown (viewer) are separate. Adding a new file format means adding one reader;
analysis code written against the containers keeps working.

## What was verified

Run the test suite with `python -m pytest -v` from this folder. The 131 public tests use only synthetic
data generated inside the tests: simulated EPSP traces, made-up experiment sheets, a fake Box client and
synthetic file metadata. No recorded data, experiment sheet or instrument file is part of this repository.

During development the readers were also checked against the lab's own instrument files, which are
private and not included:

* **ABF stimulus timing.** Decoded digital-output pulse times match the trigger signals recorded
  on input channels (`trigger1`/`trigger2` on the Zeiss rig, the `IN7` photodiode on the Olympus rig)
  to within one sample (0.05 ms). This includes Clampex's pre-sweep holding period of `n_samples // 64`,
  which the old code ignored.
* **Zeiss.** All 119 sample `.czi` files load (5 acquisition types), and every pixel equals `czifile`'s own
  output.
* **Olympus.** The uncaging target from the Bleach settings matches the independent `.roi` file inside the
  `.oib`, and 12-bit data keep their full range (max 4095).
* **Box.** A file downloaded from Box is byte-identical to the local copy and yields identical data.
  Paging through large folders, SHA-1 verification, cache reuse and eviction, and `.oif` companion
  folders are covered by tests that use a fake Box client.

## Differences from the older code (why results may shift slightly)

| Older code | Now |
|---|---|
| Stimulus times from "epochs lasting 0.1 ms", with no pre-sweep holding offset | Exact digital-output decoding, including the 1/64 holding offset, per-sweep duration increments and trains |
| Zeiss data converted to 8-bit on load | Original bit depth kept; conversion happens only for display |
| Zeiss loader indexed an 8-D array; broken with czifile ≥ 2026 | Axes rearranged by name; works with the current czifile |
| `export_to_csv` time = sample × sampling rate | time = sample / sampling rate |
| Box downloads written to never-deleted temp files; first result page only | Verified, size-limited cache; all pages read |
| Viewers sent whole 5-D stacks to the browser | Only the displayed plane is sent |

## Not included yet (next steps)

* A combined view: an imaging frame or line scan next to the matching ephys sweep, aligned in time.
* Porting the analysis code (peak detection, series resistance, deconvolution, interactive decay
  fitting, ROI selection) onto the two containers.
* Locating the Zeiss scan line on its reference image. The file stores rotation, offset and zoom
  (in `metadata`) but not the line's endpoints.
* A Box folder id for the `zeiss_lsm880` root (left empty in the settings).
