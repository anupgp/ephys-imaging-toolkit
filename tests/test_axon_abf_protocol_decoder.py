"""
Tests of the protocol decoder with small, hand-made protocols, so every rule
(pre-sweep holding, per-sweep increments, trains, merging, alternation) is
checked in isolation.
"""

import pytest

from ephys_imaging_data_loader.file_readers.axon_abf_protocol_decoder import decode_protocol

SAMPLING_RATE_HZ = 10000.0  # 0.1 ms per sample
N_SAMPLES = 6400            # -> pre-sweep holding = 100 samples = 10 ms


def make_epoch(number, duration, duration_increment=0, level=0.0, level_increment=0.0,
               period=0, width=0, epoch_type=1):
    return {
        "nEpochNum": number, "nEpochType": epoch_type,
        "fEpochInitLevel": level, "fEpochLevelInc": level_increment,
        "lEpochInitDuration": duration, "lEpochDurationInc": duration_increment,
        "lEpochPulsePeriod": period, "lEpochPulseWidth": width,
    }


def make_digital(number, high=0, train=0, alt_high=0, alt_train=0):
    return {
        "nEpochNum": number, "nDigitalValue": high, "nDigitalTrainValue": train,
        "nAlternateDigitalValue": alt_high, "nAlternateDigitalTrainValue": alt_train,
    }


def make_header(epochs, digital, alternate=False):
    return {
        "dictEpochInfoPerDAC": {0: {e["nEpochNum"]: e for e in epochs}},
        "EpochInfo": digital,
        "listDACInfo": [{"DACChNames": b"OUT 0", "DACChUnits": b"mV", "fDACHoldingLevel": -70.0}],
        "protocol": {
            "nDigitalEnable": 1, "nDigitalDACChannel": 0,
            "nAlternateDigitalOutputState": int(alternate), "nOperationMode": 5,
        },
    }


def test_single_pulse_timing_includes_holding_offset():
    header = make_header(
        [make_epoch(0, 1000), make_epoch(1, 1)],       # 100 ms baseline, then 0.1 ms pulse
        [make_digital(0), make_digital(1, high=0b1)],
    )
    epochs, pulses = decode_protocol(header, 1, N_SAMPLES, SAMPLING_RATE_HZ)
    assert len(pulses) == 1
    # 100 holding samples + 1000 baseline samples = 1100 samples = 0.11 s
    assert pulses.iloc[0]["onset_s"] == pytest.approx(0.11)
    assert pulses.iloc[0]["duration_s"] == pytest.approx(0.0001)
    assert pulses.iloc[0]["digital_output_bit"] == 0
    assert epochs.iloc[0]["epoch_type"] == "pre-sweep holding"
    assert epochs.iloc[0]["level"] == -70.0


def test_duration_and_level_increase_per_sweep():
    """Paired-pulse style: the interval epoch grows by 50 ms every sweep."""
    header = make_header(
        [make_epoch(0, 100), make_epoch(1, 1), make_epoch(2, 500, duration_increment=500, level=-80, level_increment=5),
         make_epoch(3, 1)],
        [make_digital(0), make_digital(1, high=0b10), make_digital(2), make_digital(3, high=0b10)],
    )
    epochs, pulses = decode_protocol(header, 3, N_SAMPLES, SAMPLING_RATE_HZ)
    for sweep, expected_interval_s in enumerate([0.0501, 0.1001, 0.1501]):
        onsets = pulses[pulses["sweep"] == sweep]["onset_s"].to_numpy()
        assert onsets[1] - onsets[0] == pytest.approx(expected_interval_s)
    levels = epochs[epochs["epoch"] == 2]["level"].tolist()
    assert levels == [-80.0, -75.0, -70.0]


def test_train_pulses_inside_an_epoch():
    header = make_header(
        [make_epoch(0, 2000, period=500, width=2)],      # 4 pulses, 50 ms apart
        [make_digital(0, train=0b100)],
    )
    _, pulses = decode_protocol(header, 1, N_SAMPLES, SAMPLING_RATE_HZ)
    assert pulses["digital_output_bit"].unique().tolist() == [2]
    assert pulses["onset_s"].tolist() == pytest.approx([0.01, 0.06, 0.11, 0.16])
    assert pulses["duration_s"].tolist() == pytest.approx([0.0002] * 4)


def test_back_to_back_high_epochs_are_merged():
    header = make_header(
        [make_epoch(0, 100), make_epoch(1, 100)],
        [make_digital(0, high=0b1), make_digital(1, high=0b1)],
    )
    _, pulses = decode_protocol(header, 1, N_SAMPLES, SAMPLING_RATE_HZ)
    assert len(pulses) == 1
    assert pulses.iloc[0]["duration_s"] == pytest.approx(0.02)


def test_alternate_digital_outputs_on_odd_sweeps():
    header = make_header([make_epoch(0, 10)], [make_digital(0, high=0b1, alt_high=0b10)], alternate=True)
    _, pulses = decode_protocol(header, 2, N_SAMPLES, SAMPLING_RATE_HZ)
    assert pulses[pulses["sweep"] == 0]["digital_output_bit"].tolist() == [0]
    assert pulses[pulses["sweep"] == 1]["digital_output_bit"].tolist() == [1]


def test_disabled_epochs_are_skipped():
    header = make_header(
        [make_epoch(0, 100), make_epoch(1, 999, epoch_type=0), make_epoch(2, 1)],
        [make_digital(0), make_digital(1), make_digital(2, high=0b1)],
    )
    _, pulses = decode_protocol(header, 1, N_SAMPLES, SAMPLING_RATE_HZ)
    assert pulses.iloc[0]["onset_s"] == pytest.approx((100 + 100) / SAMPLING_RATE_HZ)


def test_gap_free_file_gives_empty_tables():
    epochs, pulses = decode_protocol({"protocol": {"nOperationMode": 3}}, 1, N_SAMPLES, SAMPLING_RATE_HZ)
    assert epochs.empty and pulses.empty
    assert "onset_s" in pulses.columns
