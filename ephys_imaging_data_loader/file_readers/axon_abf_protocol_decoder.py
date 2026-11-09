"""
axon_abf_protocol_decoder.py
============================

Turns the raw Clampex protocol stored inside an ABF2 file into two readable
tables:

1. ``protocol_epochs``  – the "Waveform" tab of Clampex as a table:
   one row per (sweep, analog output, epoch) with start time, duration and level.
2. ``stimulus_pulses``  – every digital (TTL) output pulse, one row per pulse:
   which sweep, which digital output, onset and duration in seconds.

Background: how Clampex timing works (important for correct stimulus times)
---------------------------------------------------------------------------
* A sweep of N samples does NOT start with epoch A. Clampex first holds the
  output at the holding level for N // 64 samples ("pre-sweep holding"), and
  only then plays epochs A, B, C ... one after another. The old code ignored
  this offset, which shifts every stimulus time by 1/64 of the sweep length
  (e.g. 62.5 ms for a 4 s sweep). This was verified on the sample files: the
  command waveform steps exactly at N // 64 + (sum of previous epoch durations).
* Epoch durations and levels can change from sweep to sweep:
      duration(sweep k) = initial duration + k * duration increment
      level(sweep k)    = initial level    + k * level increment
  (This is how paired-pulse protocols with growing intervals are built.)
* Digital outputs are defined per epoch as bit masks:
      nDigitalValue      bit i set -> output i is HIGH for the whole epoch
      nDigitalTrainValue bit i set -> output i pulses inside the epoch with the
                                      epoch's "pulse period" and "pulse width"
  Bit 0 is digital output 0 on the digitizer, bit 1 is output 1, and so on.
  The digital epochs follow the timing of the analog output chosen as the
  "digital DAC channel" in the protocol (nDigitalDACChannel).
* With "alternate digital outputs" enabled, odd sweeps (1, 3, 5 ...) use a
  second set of digital bit masks.

All durations inside the file are in SAMPLES (per channel); this module
converts them to milliseconds / seconds using the sampling rate.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Clampex epoch types (nEpochType) -> readable names.
EPOCH_TYPE_NAMES = {
    0: "off",
    1: "step",
    2: "ramp",
    3: "pulse train",
    4: "triangle train",
    5: "cosine train",
    6: "unused",
    7: "biphasic train",
}

# Clampex acquisition modes (nOperationMode) -> readable names.
ACQUISITION_MODE_NAMES = {
    1: "event-driven (variable length)",
    2: "oscilloscope (loss-free)",
    3: "gap-free",
    4: "high-speed oscilloscope",
    5: "episodic stimulation",
}

# Column names of the two output tables (kept here so empty tables have them too).
PROTOCOL_EPOCH_COLUMNS = [
    "sweep",               # sweep index (0-based)
    "analog_output",       # DAC number (0 = first analog output)
    "analog_output_name",  # e.g. "OUT 0"
    "analog_output_unit",  # e.g. "mV" (voltage clamp) or "pA" (current clamp)
    "epoch",               # -1 = pre-sweep holding, 0 = epoch A, 1 = epoch B ...
    "epoch_type",          # "step", "ramp", ...
    "start_ms",            # epoch start, from the start of the sweep
    "duration_ms",
    "level",               # output level during the epoch (unit above)
    "digital_bits_high",   # list of digital outputs held HIGH during the epoch
    "digital_bits_train",  # list of digital outputs pulsing during the epoch
    "pulse_period_ms",     # for trains: time between pulse onsets
    "pulse_width_ms",      # for trains: width of each pulse
]
STIMULUS_PULSE_COLUMNS = ["sweep", "digital_output_bit", "onset_s", "duration_s", "epoch"]


def _bits_set(bit_mask: int) -> list[int]:
    """Return the positions of all bits that are 1, e.g. 0b0101 -> [0, 2]."""
    return [bit for bit in range(16) if (int(bit_mask) >> bit) & 1]


def _clean_text(value: Any) -> str:
    """Decode bytes from the file header and strip padding / spaces."""
    if isinstance(value, bytes):
        value = value.decode("latin-1", errors="ignore")
    return str(value).strip().strip("\x00").strip()


def describe_acquisition_mode(axon_info: dict) -> str:
    """Readable acquisition mode, e.g. 'episodic stimulation'."""
    mode_number = axon_info.get("protocol", {}).get("nOperationMode")
    if mode_number is None:
        mode_number = axon_info.get("nOperationMode")
    return ACQUISITION_MODE_NAMES.get(mode_number, f"unknown ({mode_number})")


def decode_protocol(
    axon_info: dict,
    n_sweeps: int,
    n_samples_per_sweep: int,
    sampling_rate_hz: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the ``protocol_epochs`` and ``stimulus_pulses`` tables.

    Parameters
    ----------
    axon_info : dict
        The parsed ABF header (neo's ``AxonIO(...)._axon_info``).
    n_sweeps, n_samples_per_sweep : int
        Recording layout (needed because durations can change per sweep and
        because the pre-sweep holding period is n_samples_per_sweep // 64).
    sampling_rate_hz : float
        Samples per second per channel, to convert samples -> time.

    Returns
    -------
    (protocol_epochs, stimulus_pulses) : two pandas DataFrames.
    Both are empty (with the right columns) for gap-free recordings or old
    ABF1 files that do not store epoch tables.
    """
    empty_epochs = pd.DataFrame(columns=PROTOCOL_EPOCH_COLUMNS)
    empty_pulses = pd.DataFrame(columns=STIMULUS_PULSE_COLUMNS)

    epochs_per_output: dict = axon_info.get("dictEpochInfoPerDAC") or {}
    if not epochs_per_output:
        # Gap-free recordings and ABF1 files have no epoch table.
        return empty_epochs, empty_pulses

    ms_per_sample = 1000.0 / sampling_rate_hz
    protocol = axon_info.get("protocol", {})
    output_info_list = axon_info.get("listDACInfo", [])

    # Samples before epoch A starts (Clampex convention, see module docstring).
    pre_sweep_holding_samples = n_samples_per_sweep // 64

    # Digital-output settings (apply to one chosen analog output's epochs).
    digital_outputs_enabled = bool(protocol.get("nDigitalEnable", 0))
    digital_timing_output = int(protocol.get("nDigitalDACChannel", 0))
    alternate_digital_enabled = bool(protocol.get("nAlternateDigitalOutputState", 0))
    # Per-epoch digital bit masks, indexed by epoch number.
    digital_info_by_epoch = {
        entry["nEpochNum"]: entry for entry in axon_info.get("EpochInfo", [])
    }

    epoch_rows: list[dict] = []
    pulse_rows: list[dict] = []

    for output_number, epochs in sorted(epochs_per_output.items()):
        # Name / unit / holding level of this analog output, when available.
        output_info = (
            output_info_list[output_number] if output_number < len(output_info_list) else {}
        )
        output_name = _clean_text(output_info.get("DACChNames", f"OUT {output_number}"))
        output_unit = _clean_text(output_info.get("DACChUnits", ""))
        holding_level = float(output_info.get("fDACHoldingLevel", np.nan))

        # Digital pulses are generated only along the chosen output's timeline.
        this_output_drives_digital = digital_outputs_enabled and (
            output_number == digital_timing_output
        )

        for sweep in range(n_sweeps):
            # Row for the pre-sweep holding period (epoch = -1).
            epoch_rows.append(
                {
                    "sweep": sweep,
                    "analog_output": output_number,
                    "analog_output_name": output_name,
                    "analog_output_unit": output_unit,
                    "epoch": -1,
                    "epoch_type": "pre-sweep holding",
                    "start_ms": 0.0,
                    "duration_ms": pre_sweep_holding_samples * ms_per_sample,
                    "level": holding_level,
                    "digital_bits_high": [],
                    "digital_bits_train": [],
                    "pulse_period_ms": 0.0,
                    "pulse_width_ms": 0.0,
                }
            )

            # Epochs A, B, C ... are played one after the other.
            epoch_start_sample = pre_sweep_holding_samples
            use_alternate_bits = alternate_digital_enabled and (sweep % 2 == 1)

            for epoch_number in sorted(epochs):
                epoch = epochs[epoch_number]
                epoch_type = EPOCH_TYPE_NAMES.get(epoch["nEpochType"], "unknown")
                if epoch["nEpochType"] == 0:
                    # Disabled epochs take no time and produce nothing.
                    continue

                duration_samples = int(
                    epoch["lEpochInitDuration"] + sweep * epoch["lEpochDurationInc"]
                )
                level = float(epoch["fEpochInitLevel"] + sweep * epoch["fEpochLevelInc"])
                pulse_period_samples = int(epoch.get("lEpochPulsePeriod", 0))
                pulse_width_samples = int(epoch.get("lEpochPulseWidth", 0))

                # Which digital outputs are high / pulsing in this epoch?
                bits_high: list[int] = []
                bits_train: list[int] = []
                if this_output_drives_digital and epoch_number in digital_info_by_epoch:
                    digital = digital_info_by_epoch[epoch_number]
                    high_mask = digital[
                        "nAlternateDigitalValue" if use_alternate_bits else "nDigitalValue"
                    ]
                    train_mask = digital[
                        "nAlternateDigitalTrainValue"
                        if use_alternate_bits
                        else "nDigitalTrainValue"
                    ]
                    bits_high = _bits_set(high_mask)
                    bits_train = _bits_set(train_mask)

                epoch_rows.append(
                    {
                        "sweep": sweep,
                        "analog_output": output_number,
                        "analog_output_name": output_name,
                        "analog_output_unit": output_unit,
                        "epoch": epoch_number,
                        "epoch_type": epoch_type,
                        "start_ms": epoch_start_sample * ms_per_sample,
                        "duration_ms": duration_samples * ms_per_sample,
                        "level": level,
                        "digital_bits_high": bits_high,
                        "digital_bits_train": bits_train,
                        "pulse_period_ms": pulse_period_samples * ms_per_sample,
                        "pulse_width_ms": pulse_width_samples * ms_per_sample,
                    }
                )

                # ---- Convert digital settings of this epoch into pulses -------
                # (a) outputs held HIGH for the whole epoch -> one pulse each
                for bit in bits_high:
                    pulse_rows.append(
                        {
                            "sweep": sweep,
                            "digital_output_bit": bit,
                            "onset_s": epoch_start_sample / sampling_rate_hz,
                            "duration_s": duration_samples / sampling_rate_hz,
                            "epoch": epoch_number,
                        }
                    )
                # (b) outputs pulsing inside the epoch -> a train of pulses
                if bits_train:
                    if pulse_period_samples <= 0:
                        logger.warning(
                            "Epoch %s has digital train bits %s but a pulse period "
                            "of 0; no train pulses generated.",
                            epoch_number,
                            bits_train,
                        )
                    else:
                        # Pulse onsets every period, as long as the pulse fits in the epoch.
                        onsets = np.arange(
                            0, duration_samples - pulse_width_samples + 1, pulse_period_samples
                        )
                        for bit in bits_train:
                            for onset in onsets:
                                pulse_rows.append(
                                    {
                                        "sweep": sweep,
                                        "digital_output_bit": bit,
                                        "onset_s": (epoch_start_sample + onset)
                                        / sampling_rate_hz,
                                        "duration_s": pulse_width_samples / sampling_rate_hz,
                                        "epoch": epoch_number,
                                    }
                                )

                epoch_start_sample += duration_samples

    protocol_epochs = pd.DataFrame(epoch_rows, columns=PROTOCOL_EPOCH_COLUMNS)
    stimulus_pulses = _merge_back_to_back_pulses(
        pd.DataFrame(pulse_rows, columns=STIMULUS_PULSE_COLUMNS)
    )
    return protocol_epochs, stimulus_pulses


def _merge_back_to_back_pulses(pulses: pd.DataFrame) -> pd.DataFrame:
    """
    Join pulses on the same output that touch each other in time.

    If an output is high in epoch B and also in the directly following epoch C,
    the physical TTL line stays high: that is ONE pulse, not two.
    """
    if pulses.empty:
        return pulses

    merged_rows: list[dict] = []
    pulses = pulses.sort_values(["sweep", "digital_output_bit", "onset_s"])
    for (sweep, bit), group in pulses.groupby(["sweep", "digital_output_bit"], sort=False):
        current: dict | None = None
        for row in group.to_dict("records"):
            if current is not None and np.isclose(
                row["onset_s"], current["onset_s"] + current["duration_s"]
            ):
                # Starts exactly where the previous one ended -> extend it.
                current["duration_s"] += row["duration_s"]
            else:
                if current is not None:
                    merged_rows.append(current)
                current = dict(row)
        if current is not None:
            merged_rows.append(current)

    return (
        pd.DataFrame(merged_rows, columns=STIMULUS_PULSE_COLUMNS)
        .sort_values(["sweep", "onset_s", "digital_output_bit"])
        .reset_index(drop=True)
    )
