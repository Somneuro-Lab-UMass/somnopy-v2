import mne
from mne.io import Raw


def set_up_raw(raw: Raw, rerefer: bool = False, chan_limit=None,
               montage_temp: str = "standard_1005", is_montage: bool = False, drop_chan=()) -> Raw:
    ch_drop = [
        ch for ch in raw.ch_names
        if ch.startswith('M') or 'EMG' in ch or 'EOG' in ch or 'ECG' in ch or 'chin' in ch.lower() or ch.startswith('E')
    ]
    raw.drop_channels(ch_drop)
    raw.drop_channels(drop_chan, on_missing='warn')
    if chan_limit is not None:
        raw = raw.pick_channels(chan_limit, ordered=False, on_missing='warn')
    if rerefer:
        raw.set_eeg_reference(ref_channels=['M1', 'M2'])
    if is_montage:
        montage = mne.channels.make_standard_montage(montage_temp)
        raw.set_montage(montage, on_missing='warn')
    return raw
