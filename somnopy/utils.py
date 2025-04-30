import mne
from mne.io import Raw


def set_up_raw(raw: Raw, rerefer: bool = False, chan_limit=None,
               montage_temp: str = "standard_1005", is_montage: bool = False, drop_chan=()) -> Raw:
    """
    Preprocess an MNE Raw object: drop unwanted channels, optionally pick a subset,
    re-reference, and apply a standard montage.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data to preprocess.
    rerefer : bool, default=False
        If True, re-reference the data to mastoid channels ['M1', 'M2'].
    chan_limit : iterable of str or None, default=None
        If provided, only these channel names will be retained; others are dropped.
    montage_temp : str, default="standard_1005"
        Name of the standard montage to apply when `is_montage` is True.
    is_montage : bool, default=False
        If True, apply the specified standard montage to channel locations.
    drop_chan : iterable of str, default=()
        Additional channel names to drop (on top of physiological channels).

    Returns
    -------
    raw : mne.io.Raw
        The modified Raw object with channels dropped/picked, re-referenced, and montage applied.

    Notes
    -----
    - Drops channels whose names start with 'M', 'E' or contain 'EMG', 'EOG', 'ECG' or 'chin'.
    - If `chan_limit` is set, any channels not in that list are removed.
    - Use `on_missing='warn'` to warn (not error) if specified `drop_chan` or `chan_limit` names are absent.
    - Re-referencing and montage steps modify the Raw in place.
    """
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
