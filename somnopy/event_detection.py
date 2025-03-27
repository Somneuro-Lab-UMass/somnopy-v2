from typing import Optional

import mne
import numpy as np
import scipy.signal
import math
import pandas as pd
from mne.io import Raw
from scipy.signal import hilbert


def SO_detection(raw, stage, target_stage=('N2', 'SWS'), filter_freq=None, duration=None, baseline=True,
                 filter_type='fir',
                 method='Staresina', verbose=True):
    """
        Detect Slow Oscillations (SOs) in EEG data using different methods.

        Parameters
        ----------
        raw : mne.io.Raw
            MNE Raw object containing EEG data.
        stage : list of tuples
            List describing sleep stage segments, e.g., [(stage_value, dur_sec, label), ...].
        target_stage : tuple, default=('N2', 'SWS')
            Sleep stages to analyze. Can be specified as names ('N2', 'SWS') or numerical codes.
        filter_freq : tuple, optional
            Lower and upper frequency bounds for filtering. Defaults depend on `method`.
        duration : tuple, optional
            Minimum and maximum duration of an SO. Defaults depend on `method`.
        baseline : bool, default=True
            Whether to subtract the mean from EEG data before SO detection.
        filter_type : str, default='fir'
            Filtering method, either 'fir' or 'iir'.
        method : str, default='Staresina'
            SO detection method. Available options:
            - 'Staresina'
            - 'Ng'
            - 'Massimini'
            - 'Dimulescu'
            - 'Molle'
            - 'Ngo'
        verbose : bool, default=True
            Whether to print detection results.

        Returns
        -------
        raw_copy : mne.io.Raw
            Preprocessed copy of the Raw object.
        final_SO_candidates : pd.DataFrame
            DataFrame containing detected SO events.
        SO_summary_df : pd.DataFrame
            Summary statistics of detected SOs.
        """
    method_params = {
        'Staresina': {'filter_freq': (0.16, 1.25), 'duration': (0.8, 2)},
        'Ng': {'filter_freq': (0.16, 1.25), 'duration': (0.8, 2)},
        'Massimini': {'filter_freq': (0.1, 4), 'duration': (0.8, 2)},
        'Dimulescu': {'filter_freq': (0.1, 4), 'duration': (0.8, 2)},
        'Molle': {'filter_freq': (0.1, 2), 'duration': (0.9, 2)},
        'Ngo': {'filter_freq': (None, 3.5), 'duration': (0.9, 2)}
    }

    stage_mapping = {"Wake": 0, "N1": 1, "N2": 2, "SWS": 3, "REM": 4}
    target_stage = [
        stage_mapping[stg] if isinstance(stg, str) and stg in stage_mapping else stg
        for stg in target_stage
    ]
    defaults = method_params.get(method, {'filter_freq': (0.25, 3), 'duration': (0.8, 2)})
    filter_freq = filter_freq or defaults['filter_freq']
    duration = duration or defaults['duration']

    l_freq, h_freq = filter_freq
    dur_lower, dur_upper = duration

    # Drop unnecessary channels
    drop_ch = [ch for ch in raw.ch_names if ch.startswith('M') or any(x in ch for x in ['EMG', 'EOG', 'ECG'])]
    raw_copy = raw.copy().drop_channels(drop_ch)

    # Apply filtering
    filter_params = {
        'picks': 'eeg',
        'l_freq': l_freq,
        'h_freq': h_freq,
        'phase': 'zero',
        'verbose': 'ERROR'
    }

    if filter_type == 'iir':
        filter_params.update({'method': 'iir', 'iir_params': {'order': 2, 'ftype': 'butter'}})
    else:
        filter_params.update({
            'method': 'fir',
            'fir_design': 'firwin',
            'phase': 'zero-double',
            'fir_window': 'hamming',
            'filter_length': 'auto'
        })

    raw_copy.filter(**filter_params)

    eeg_data = raw_copy.get_data(picks='eeg')
    if baseline:
        eeg_centered = eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
    else:
        eeg_centered = eeg_data

    annotations = raw_copy.annotations
    bad_epochs = [(ann['onset'], ann['onset'] + ann['duration']) for ann in annotations if
                  'Bad_epoch' in ann['description']]

    for bad_start, bad_end in bad_epochs:
        bad_start_idx = int(bad_start * raw_copy.info['sfreq'])
        bad_end_idx = int(bad_end * raw_copy.info['sfreq'])
        eeg_centered[:, bad_start_idx:bad_end_idx] = 0

    all_SO_candidates = []

    # Run separately for each target stage
    for t_stage in target_stage:
        stage_data = [s for s in stage if s[0] == t_stage]
        if not stage_data:
            continue
        durations_and_ptps = []

        # Process each sleep stage individually
        for idx, (stage_value, duration, _) in enumerate(stage):
            if stage_value != t_stage:
                continue

            start_time = np.sum([dur for _, dur, _ in stage[:idx]])
            end_time = start_time + duration

            for ch_idx in range(eeg_centered.shape[0]):
                global_start_idx = int(start_time * raw_copy.info['sfreq'])
                global_end_idx = int(end_time * raw_copy.info['sfreq'])
                signal = eeg_centered[ch_idx, global_start_idx:global_end_idx]
                zero_crossings = np.where(np.diff(np.signbit(signal)))[0]

                p2n_crossings = []
                for crossing in zero_crossings:
                    onset = start_time + (crossing / raw_copy.info['sfreq'])
                    if signal[crossing] > 0:
                        p2n_crossings.append(onset)

                for i in range(1, len(p2n_crossings)):
                    p2n_dur = p2n_crossings[i] - p2n_crossings[i - 1]

                    if dur_lower <= p2n_dur <= dur_upper:
                        start_time_global = p2n_crossings[i - 1]
                        end_time_global = p2n_crossings[i]

                        start_idx = int(start_time_global * raw_copy.info['sfreq'])
                        end_idx = int(end_time_global * raw_copy.info['sfreq'])

                        ptp_amplitude = np.ptp(eeg_centered[ch_idx, start_idx:end_idx])
                        trough_idx = np.argmin(eeg_centered[ch_idx, start_idx:end_idx])
                        trough_amplitude = eeg_centered[ch_idx, start_idx + trough_idx]

                        durations_and_ptps.append({
                            'ch_id': ch_idx + 1,
                            'ch_name': raw_copy.ch_names[ch_idx],
                            'p2n_start': start_time_global,
                            'p2n_end': end_time_global,
                            'duration': p2n_dur,
                            'ptp_amplitude': ptp_amplitude,
                            'trough_amplitude': trough_amplitude
                        })

        df = pd.DataFrame(durations_and_ptps)

        if method in ['Staresina', 'Ng']:

            if method == 'Staresina':
                top_25_pct_ptp_threshold = df['ptp_amplitude'].quantile(0.75)
                top_durations = df[df['ptp_amplitude'] >= top_25_pct_ptp_threshold]
            elif method == 'Ng':
                top_25_pct_ptp_threshold = df['ptp_amplitude'].quantile(0.7)
                top_25_pct_trough_threshold = df['trough_amplitude'].quantile(0.3)
                top_durations = df[(df['ptp_amplitude'] >= top_25_pct_ptp_threshold) & (
                        df['trough_amplitude'] <= top_25_pct_trough_threshold)]

            sfreq = raw_copy.info['sfreq']
            SO_candidate = []

            for idx, row in top_durations.iterrows():
                start_idx = int(row['p2n_start'] * sfreq)
                end_idx = int(row['p2n_end'] * sfreq)

                signal_segment = eeg_centered[row['ch_id'] - 1, start_idx:end_idx]
                trough_idx = np.argmin(signal_segment)
                trough_time = row['p2n_start'] + (trough_idx / sfreq)

                SO_candidate.append({
                    'ch_id': row['ch_id'],
                    'ch_name': row['ch_name'],
                    'p2n_start': row['p2n_start'],
                    'p2n_end': row['p2n_end'],
                    'duration': row['duration'],
                    'ptp_amplitude': row['ptp_amplitude'],
                    'trough_time': trough_time,
                    'trough_amplitude': row['trough_amplitude'],
                    'stage': t_stage  # Add stage info
                })

        elif method in ['Dimulescu', 'Massimini']:
            # Absolute method
            sfreq = raw_copy.info['sfreq']
            SO_candidate = []

            if method == 'Dimulescu':
                trough_amp_thres = -0.00004
                ptp_amp_thres = 0.00007
            elif method == 'Massimini':
                trough_amp_thres = -0.00008
                ptp_amp_thres = 0.00014

            for idx, row in df.iterrows():
                if row['trough_amplitude'] < trough_amp_thres and row['ptp_amplitude'] > ptp_amp_thres:
                    start_idx = int(row['p2n_start'] * sfreq)
                    end_idx = int(row['p2n_end'] * sfreq)

                    signal_segment = eeg_centered[row['ch_id'] - 1, start_idx:end_idx]
                    trough_idx = np.argmin(signal_segment)
                    trough_time = row['p2n_start'] + (trough_idx / sfreq)

                    SO_candidate.append({
                        'ch_id': row['ch_id'],
                        'ch_name': row['ch_name'],
                        'p2n_start': row['p2n_start'],
                        'p2n_end': row['p2n_end'],
                        'duration': row['duration'],
                        'ptp_amplitude': row['ptp_amplitude'],
                        'trough_time': trough_time,
                        'trough_amplitude': row['trough_amplitude'],
                        'stage': t_stage  # Add stage info
                    })

        elif method in ['Molle', 'Ngo']:
            # Ratio method
            sfreq = raw_copy.info['sfreq']
            SO_candidate = []

            mean_trough_amplitude = df['trough_amplitude'].mean()
            mean_ptp_amplitude = df['ptp_amplitude'].mean()

            if method == 'Molle':
                trough_ratio = 2 / 3
                ptp_ratio = 2 / 3
            elif method == 'Ngo':
                trough_ratio = 1.25
                ptp_ratio = 1.25

            for idx, row in df.iterrows():
                if row['trough_amplitude'] <= trough_ratio * mean_trough_amplitude and row[
                    'ptp_amplitude'] >= ptp_ratio * mean_ptp_amplitude:
                    start_idx = int(row['p2n_start'] * sfreq)
                    end_idx = int(row['p2n_end'] * sfreq)

                    signal_segment = eeg_centered[row['ch_id'] - 1, start_idx:end_idx]
                    trough_idx = np.argmin(signal_segment)
                    trough_time = row['p2n_start'] + (trough_idx / sfreq)

                    SO_candidate.append({
                        'ch_id': row['ch_id'],
                        'ch_name': row['ch_name'],
                        'p2n_start': row['p2n_start'],
                        'p2n_end': row['p2n_end'],
                        'duration': row['duration'],
                        'ptp_amplitude': row['ptp_amplitude'],
                        'trough_time': trough_time,
                        'trough_amplitude': row['trough_amplitude'],
                        'stage': t_stage  # Add stage info
                    })

        all_SO_candidates.append(pd.DataFrame(SO_candidate))

    final_SO_candidates = pd.concat(all_SO_candidates, ignore_index=True)

    SO_summary = []

    for t_stage in target_stage:
        stage_data = [s for s in stage if s[0] == t_stage]
        if not stage_data:
            continue
        stage_SO_candidates = final_SO_candidates[final_SO_candidates['stage'] == t_stage]

        mean_ptp_amp = stage_SO_candidates['ptp_amplitude'].mean()
        mean_dur = stage_SO_candidates['duration'].mean()
        total_dur_stage = sum([dur for stage_val, dur, _ in stage if stage_val == t_stage])
        num_segments = total_dur_stage // 30
        num_channels = eeg_centered.shape[0]
        SO_density = len(stage_SO_candidates) / (num_channels * num_segments) if num_segments > 0 else 0

        stage_mapping = {0: "Wake", 1: "N1", 2: "N2", 3: "SWS", 4: "REM"}
        if verbose:
            print(f"Detected SOs in stage {stage_mapping.get(t_stage, t_stage)}")
            print(f"Method: {method}")
            print(f"PTP amplitude: {mean_ptp_amp * 1e6:.4f}")
            print(f"Mean Duration: {mean_dur:.4f}")
            print(f"SO Density per 30s: {SO_density:.4f}")
            print(f"Total Count: {len(stage_SO_candidates)}")
            print("---------------------------------------------------------------")

        SO_summary.append({
            'stage': t_stage,
            'stage_dur': sum([d for (stg, d, _) in stage if stg == t_stage]),
            'mean_SO_amp': mean_ptp_amp,
            'mean_SO_dur': mean_dur,
            'SO_density': SO_density
        })

    SO_summary_df = pd.DataFrame(SO_summary)

    if not final_SO_candidates.empty:
        all_stage_dur = sum([d for (stg, d, _) in stage if stg in target_stage])
        num_segments_all = int(all_stage_dur // 30)
        num_channels = eeg_centered.shape[0]

        mean_SO_amp_all = final_SO_candidates['ptp_amplitude'].mean()
        mean_SO_dur_all = final_SO_candidates['duration'].mean()
        SO_density_all = len(final_SO_candidates) / (num_channels * num_segments_all) if num_segments_all > 0 else 0

        all_summary = {
            'stage': 'all',
            'stage_dur': all_stage_dur,
            'mean_SO_amp': mean_SO_amp_all,
            'mean_SO_dur': mean_SO_dur_all,
            'SO_density': SO_density_all
        }

        SO_summary_df = pd.concat([SO_summary_df, pd.DataFrame([all_summary])], ignore_index=True)

    channel_summary_list = []
    for t_stage in target_stage:
        stage_candidates = final_SO_candidates[final_SO_candidates['stage'] == t_stage]
        if stage_candidates.empty:
            continue
        total_dur_stage = sum([d for (stg, d, _) in stage if stg == t_stage])
        num_segments = total_dur_stage // 30
        for ch in stage_candidates['ch_name'].unique():
            ch_events = stage_candidates[stage_candidates['ch_name'] == ch]
            mean_amp = ch_events['ptp_amplitude'].mean()
            mean_dur = ch_events['duration'].mean()
            density = len(ch_events) / num_segments if num_segments > 0 else 0
            channel_summary_list.append({
                'stage': t_stage,
                'channel': ch,
                'stage_dur': total_dur_stage,
                'mean_SO_amp': mean_amp,
                'mean_SO_dur': mean_dur,
                'SO_density': density
            })
    channel_summary_df = pd.DataFrame(channel_summary_list)
    SO_summary_df['channel'] = 'all'
    SO_summary_df = pd.concat([SO_summary_df, channel_summary_df], ignore_index=True)

    # Add channel-level data for the aggregated 'all' stage
    if not final_SO_candidates.empty:
        channel_summary_all = []
        total_all_dur = sum([d for (stg, d, _) in stage if stg in target_stage])
        num_segments_all = total_all_dur // 30
        for ch in final_SO_candidates['ch_name'].unique():
            ch_events = final_SO_candidates[final_SO_candidates['ch_name'] == ch]
            mean_amp = ch_events['ptp_amplitude'].mean()
            mean_dur = ch_events['duration'].mean()
            density = len(ch_events) / num_segments_all if num_segments_all > 0 else 0
            channel_summary_all.append({
                'stage': 'all',
                'channel': ch,
                'stage_dur': total_all_dur,
                'mean_SO_amp': mean_amp,
                'mean_SO_dur': mean_dur,
                'SO_density': density
            })
        channel_summary_all_df = pd.DataFrame(channel_summary_all)
        SO_summary_df = pd.concat([SO_summary_df, channel_summary_all_df], ignore_index=True)

    column_order = ['stage', 'channel', 'stage_dur', 'mean_SO_amp', 'mean_SO_dur', 'SO_density']
    SO_summary_df = SO_summary_df[column_order]

    return raw_copy, final_SO_candidates, SO_summary_df


def SP_detection(raw: Raw, stage, target_stage=('N2', 'SWS'), method: str = "Hahn2020",
                 l_freq: Optional[float] = None, h_freq: Optional[float] = None, dur_lower: Optional[float] = None,
                 dur_upper: Optional[float] = None, baseline: bool = True, verbose: bool = True):
    """
    Detect sleep spindles in EEG data using five published methods:

    1) hahn2020 (default)
       - 12-16 Hz bandpass filter, Hilbert transform to obtain amplitude envelope,
         thresholded at a given percentile (default 75th),
         with a duration criterion of 0.5 to 3 seconds.

    2) martin2013
       - 11-15 Hz bandpass filter (linear phase FIR, forward and reverse for 0-phase distortion).
       - Compute RMS of the filtered signal with a 0.25 s time window.
       - Threshold at its 95th percentile (published criteria: Schabus et al., 2007).
       - A spindle is identified if at least two consecutive RMS time points
         exceed the threshold, and total duration >= 0.5 s.

    3) wamsley2012
       - Discrete sleep spindle events detected using a wavelet-based algorithm
         with an 8-parameter complex Morlet wavelet, ~10-16 Hz range.
       - Spindles are identified at each channel by a thresholding algorithm
         applied to the wavelet scale corresponding to 10-16 Hz.
       - The rectified moving average of this wavelet signal is calculated
         with a 100 ms sliding window.
       - Threshold = 4.5 times the mean signal amplitude across artifact-free epochs.
       - Duration >= 0.4 s to count as a spindle.

    4) Wendt2012
       - The raw data are bandpass filtered between 11-16 Hz.
       - The signal is then rectified.
       - Two separate lowpass filters are applied to create two envelopes
         (one ~2.25 Hz passband with offset 3 μV, the other ~1 Hz passband with offset 8 μV).
       - A time-varying threshold is formed by each envelope + offset.
       - Any rectified signal crossing either threshold boundary within 0.5-3 s
         is considered a spindle candidate. A final decision fusion is applied

    5) Ferrarelli2007
       - 12-15 Hz bandpass filter (–3 dB at 12 and 15 Hz).
       - The amplitude of the rectified filtered signal is used as the time series.
       - If the signal's amplitude exceeds an upper threshold (8 × channel average amplitude),
         it is considered a spindle candidate. Peak amplitude is the local maximum above
         that threshold. The start and end are the preceding/following points
         where amplitude drops below a lower threshold (2 × channel average amplitude).
       - Duration >= 0.5 s is typically required (AASM-based).

    Parameters
    ----------
    verbose
    raw : mne.io.Raw
        MNE Raw object containing EEG data.
    stage : list of tuples
        A list describing sleep stage segments, e.g. [(stage_value, dur_sec, label), ...].
        The function accumulates these durations in order to determine
        the time boundaries for each stage segment.
    method : str
        The spindle detection method to be used. Must be one of:
        ["hahn2020", "martin2013", "wamsley2012", "Wendt2012", "Ferrarelli2007"].
    l_freq : float
        Default low cut-off frequency for the bandpass filter
    h_freq : float
        Default high cut-off frequency for the bandpass filter
    dur_lower : float
        Minimum duration (seconds) for a valid spindle event.
    dur_upper : float
        Maximum duration (seconds) for a valid spindle event.
    baseline : bool
        Whether to subtract the mean of the entire channel signal
        from each channel prior to detection.
    target_stage : list
        List of stage values (e.g., [2, 3]) on which detection will be performed.

    Returns
    -------
    raw_copy : mne.io.Raw
        A copy of the Raw object with only EEG channels of interest
        (e.g., excluding EMG/EOG/ECG), filtered according to the chosen method.
    final_SP_candidates : pd.DataFrame
        A DataFrame containing all detected spindle events across channels
        and target stages, with columns including 'ch_id', 'ch_name',
        'start_time', 'end_time', 'peak_time', 'duration', 'amplitude', etc.
    event_summary : pd.DataFrame
         A DataFrame containing spindle metrics by stage and channel, with columns
         including 'stage', 'channel', 'mean_SP_amp', 'mean_SP_dur',
         'SP_density', 'SP_count'
    """

    raw_copy = raw.copy()
    sfreq = raw_copy.info['sfreq']
    stage_mapping = {"Wake": 0, "N1": 1, "N2": 2, "SWS": 3, "REM": 4}
    target_stage = [
        stage_mapping[stg] if isinstance(stg, str) and stg in stage_mapping else stg
        for stg in target_stage
    ]
    if method == "Hahn2020":
        # 12-16 Hz
        if l_freq is None:
            l_freq = 12
        if h_freq is None:
            h_freq = 16
        if dur_lower is None:
            dur_lower = 0.5
        if dur_upper is None:
            dur_upper = 3

    elif method == "Martin2013":
        # 11-15 Hz
        if l_freq is None:
            l_freq = 11
        if h_freq is None:
            h_freq = 15
        if dur_lower is None:
            dur_lower = 0.5
        if dur_upper is None:
            dur_upper = math.inf

    elif method == "Wamsley2012":
        # wavelet-based approach (10-16 Hz scale).
        # We will still do an initial bandpass (approx 10-16) to remove out-of-band noise
        # prior to wavelet transform if desired, but the main detection is wavelet-based.
        if l_freq is None:
            l_freq = 10
        if h_freq is None:
            h_freq = 16
        if dur_lower is None:
            dur_lower = 0.4
        if dur_upper is None:
            dur_upper = math.inf


    elif method == "Wendt2012":
        # 11-16 Hz
        if l_freq is None:
            l_freq = 11
        if h_freq is None:
            h_freq = 16
        if dur_lower is None:
            dur_lower = 0.5
        if dur_upper is None:
            dur_upper = 3


    elif method == "Ferrarelli2007":
        # 12-15 Hz
        if l_freq is None:
            l_freq = 12
        if h_freq is None:
            h_freq = 15
        if dur_lower is None:
            dur_lower = 0.5
        if dur_upper is None:
            dur_upper = math.inf

    else:
        raise RuntimeError("Chosen method not known. Please pick existing method")

    raw_copy.filter(
        picks='eeg',
        l_freq=l_freq,
        h_freq=h_freq,
        method='fir',
        fir_design='firwin',
        phase='zero-double',
        fir_window='hamming',
        filter_length='auto',
        verbose='ERROR'
    )

    eeg_data = raw_copy.get_data(picks='eeg')
    if baseline:
        eeg_data = eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
    else:
        eeg_data = eeg_data

    annotations = raw_copy.annotations
    bad_epochs = [
        (ann['onset'], ann['onset'] + ann['duration'])
        for ann in annotations
        if 'Bad_epoch' in ann['description']
    ]
    for bad_start, bad_end in bad_epochs:
        bad_start_idx = int(bad_start * sfreq)
        bad_end_idx = int(bad_end * sfreq)
        eeg_data[:, bad_start_idx:bad_end_idx] = 0

    # 8. Loop through target stages and detect spindles segment by segment
    all_stage_dfs = []
    n_channels = eeg_data.shape[0]

    for s_val in target_stage:
        # find all segments for this stage
        stage_segments = [(val, dur, label) for (val, dur, label) in stage if val == s_val]
        if not stage_segments:
            continue

        local_candidates = []

        # Accumulate time so we know start index for each segment
        accumulated_time = 0.0
        for idx, (val, duration_s, _) in enumerate(stage):
            if val != s_val:
                accumulated_time += duration_s
                continue

            segment_start_time = accumulated_time
            segment_end_time = segment_start_time + duration_s
            accumulated_time += duration_s

            seg_start_idx = int(segment_start_time * sfreq)
            seg_end_idx = int(segment_end_time * sfreq)

            for ch_idx in range(n_channels):
                signal_seg = eeg_data[ch_idx, seg_start_idx:seg_end_idx]

                # Detect spindles on this single-epoch segment
                sp_list = __detect_spindles_for_epoch(method, signal_seg, sfreq, l_freq, h_freq, dur_lower, dur_upper)
                # Convert local indices to global times
                for sp in sp_list:
                    local_candidates.append({
                        'ch_id': ch_idx + 1,
                        'ch_name': raw_copy.ch_names[ch_idx],
                        'start_time': (sp['start_idx'] + seg_start_idx) / sfreq,
                        'end_time': (sp['end_idx'] + seg_start_idx) / sfreq,
                        'peak_time': (sp['peak_idx'] + seg_start_idx) / sfreq,
                        'duration': sp['duration'],
                        'amplitude': sp['amplitude'],
                        'stage': s_val
                    })

        # Convert to DataFrame
        stage_df = pd.DataFrame(local_candidates)
        if not stage_df.empty:
            # Optional Z-normalization across spindles in this stage
            stage_df['z_amplitude'] = (
                    (stage_df['amplitude'] - stage_df['amplitude'].mean()) /
                    stage_df['amplitude'].std()
            )
            stage_df['z_duration'] = (
                    (stage_df['duration'] - stage_df['duration'].mean()) /
                    stage_df['duration'].std()
            )
        all_stage_dfs.append(stage_df)

    # 9. Concatenate results from all target stages
    if all_stage_dfs:
        final_SP_candidates = pd.concat(all_stage_dfs, ignore_index=True)
    else:
        final_SP_candidates = pd.DataFrame(
            columns=['ch_id', 'ch_name', 'start_time', 'end_time',
                     'peak_time', 'duration', 'amplitude', 'stage',
                     'z_amplitude', 'z_duration']
        )

    SP_summary = []

    for s_val in target_stage:
        stage_df = final_SP_candidates[final_SP_candidates['stage'] == s_val]
        if stage_df.empty:
            continue

        if method == "Wamsley2012":
            mean_amp = np.sqrt(stage_df['amplitude'].mean())  # Convert power to amplitude
        else:
            mean_amp = stage_df['amplitude'].mean()
        mean_dur = stage_df['duration'].mean()
        total_dur_stage = sum([d for (stg, d, _) in stage if stg == s_val])
        num_segments = int(total_dur_stage // 30)
        sp_density = len(stage_df) / (n_channels * num_segments) if num_segments > 0 else 0
        count = len(stage_df)

        stage_mapping = {0: "Wake", 1: "N1", 2: "N2", 3: "SWS", 4: "REM"}
        if verbose:
            print(f"Detected SPs in stage {stage_mapping.get(s_val, s_val)}")
            print(f"Method: {method}")
            print(f"Average Amplitude: {mean_amp * 1e6:.4f} uv")
            print(f"Average Duration: {mean_dur:.4f}s")
            print(f"Spindle Density (per 30s): {sp_density:.4f}")
            print(f"Total Spindle Count: {len(stage_df)}")
            print("---------------------------------------------------------------")

        SP_summary.append({
            'stage': s_val,
            'mean_SP_amp': mean_amp,
            'mean_SP_dur': mean_dur,
            'SP_density': sp_density,
            'SP_count': count
        })

    SP_summary_df = pd.DataFrame(SP_summary)

    if not final_SP_candidates.empty:
        all_stage_dur = sum([d for (stg, d, _) in stage if stg in target_stage])
        num_segments_all = int(all_stage_dur // 30)
        num_channels = eeg_data.shape[0]

        mean_SP_amp_all = final_SP_candidates['amplitude'].mean()
        mean_SP_dur_all = final_SP_candidates['duration'].mean()
        SP_density_all = len(final_SP_candidates) / (num_channels * num_segments_all) if num_segments_all > 0 else 0
        count = len(final_SP_candidates)
        all_summary = {
            'stage': 'all',
            'mean_SP_amp': mean_SP_amp_all,
            'mean_SP_dur': mean_SP_dur_all,
            'SP_density': SP_density_all,
            'SP_count': count
        }

        SP_summary_df = pd.concat([SP_summary_df, pd.DataFrame([all_summary])], ignore_index=True)

    channel_summary_list = []
    for s_val in target_stage:
        stage_candidates = final_SP_candidates[final_SP_candidates['stage'] == s_val]
        if stage_candidates.empty:
            continue
        total_dur_stage = sum([d for (stg, d, _) in stage if stg == s_val])
        num_segments = int(total_dur_stage // 30)
        for ch in stage_candidates['ch_name'].unique():
            ch_events = stage_candidates[stage_candidates['ch_name'] == ch]
            mean_amp = ch_events['amplitude'].mean()
            mean_dur = ch_events['duration'].mean()
            density = len(ch_events) / num_segments if num_segments > 0 else 0
            count = len(ch_events)
            channel_summary_list.append({
                'stage': s_val,
                'channel': ch,
                'mean_SP_amp': mean_amp,
                'mean_SP_dur': mean_dur,
                'SP_density': density,
                'SP_count': count
            })
    channel_summary_df = pd.DataFrame(channel_summary_list)
    SP_summary_df['channel'] = 'all'
    SP_summary_df = pd.concat([SP_summary_df, channel_summary_df], ignore_index=True)

    # Add channel-level data for the aggregated 'all' stage
    if not final_SP_candidates.empty:
        channel_summary_all = []
        total_all_dur = sum([d for (stg, d, _) in stage if stg in target_stage])
        num_segments_all = int(total_all_dur // 30)
        for ch in final_SP_candidates['ch_name'].unique():
            ch_events = final_SP_candidates[final_SP_candidates['ch_name'] == ch]
            mean_amp = ch_events['amplitude'].mean()
            mean_dur = ch_events['duration'].mean()
            density = len(ch_events) / (num_segments_all) if num_segments_all > 0 else 0
            count = len(ch_events)
            channel_summary_all.append({
                'stage': 'all',
                'channel': ch,
                'mean_SP_amp': mean_amp,
                'mean_SP_dur': mean_dur,
                'SP_density': density,
                'SP_count': count
            })
        channel_summary_all_df = pd.DataFrame(channel_summary_all)
        SP_summary_df = pd.concat([SP_summary_df, channel_summary_all_df], ignore_index=True)

    column_order = ['stage', 'channel', 'mean_SP_amp', 'mean_SP_dur', 'SP_density', 'SP_count']
    SP_summary_df = SP_summary_df[column_order]

    return raw_copy, final_SP_candidates, SP_summary_df


def detect_swa(raw: Raw, stages=None, file_name='id', l_freq=0.5, h_freq=4):
    """
    Compute slow wave activity (SWA) from an mne.Raw object, optionally filtering
    the analysis to only include data from specified sleep stages.

    Parameters:
    -----------
    raw : mne.io.Raw
        The raw EEG/MEG data.
    stages : list of str, optional
        A list of stage descriptions (e.g., ['N2', 'N3']) to include in the analysis.
        Only segments with annotations whose description matches an entry in this list
        will be used for computing SWA. If None, the entire recording is used.
    file_name : str, default='id'
        Identifier for the subject.
    l_freq : float, default=0.5
        Lower bound of the slow-wave frequency band.
    h_freq : float, default=4
        Upper bound of the slow-wave frequency band.

    Returns
    -------
    pd.DataFrame
        DataFrame containing SWA measurements per channel.
    """
    # Sampling frequency
    fs = raw.info['sfreq']
    # Channel names
    ch_names = raw.info['ch_names']
    # Get the data as a numpy array (shape: n_channels x n_times)
    data = raw.get_data()

    if stages is not None and raw.annotations is not None:
        mask = np.zeros(data.shape[1], dtype=bool)
        for onset, duration, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
            if desc in stages:
                start_sample = int(onset * fs)
                end_sample = int((onset + duration) * fs)
                # Ensure indices remain within data bounds
                start_sample = max(start_sample, 0)
                end_sample = min(end_sample, data.shape[1])
                mask[start_sample:end_sample] = True
        if not np.any(mask):
            raise ValueError(f"No segments found for the specified stages: {stages}")
    else:
        # Use the entire data if no stages are provided
        mask = np.ones(data.shape[1], dtype=bool)

    # Bandpass filter the data to the slow wave (delta) band: 0.5 - 4 Hz
    filtered_data = mne.filter.filter_data(data, sfreq=fs, l_freq=l_freq, h_freq=h_freq, verbose=False)

    # Compute the analytic signal via Hilbert transform along the time axis
    analytic_signal = scipy.signal.hilbert(filtered_data, axis=1)
    # Compute the amplitude envelope (i.e. the instantaneous amplitude)
    amplitude_envelope = np.abs(analytic_signal)

    # Average the amplitude envelope over time for each channel to get a SWA measure
    swa_values = np.mean(amplitude_envelope[:, mask], axis=1)

    # # Create and return a DataFrame with the results
    # df = pd.DataFrame({
    #     'Channel': ch_names,
    #     'Slow_Wave_Activity': swa_values
    # })
    # Create a dictionary mapping each channel name to its slow wave activity value.
    data_dict = {ch: swa for ch, swa in zip(ch_names, swa_values)}

    # Create a DataFrame with one row using the filename as the index.
    df = pd.DataFrame([data_dict])

    # Optionally, insert the filename as the first column.
    df.insert(0, 'Participant_id', file_name)

    return df


def __detect_spindles_for_epoch(method, signal_1d, sfreq, l_freq, h_freq, dur_lower, dur_upper):
    # 7. Define helper function to detect spindles from a single epoch signal
    #    according to the chosen method
    """
    Detect spindle events in one single-epoch, single-channel signal
    using the specified 'method'.
    Returns a list of dictionaries (one per spindle event).
    """

    if method == "Hahn2020":
        spindle_candidates = __hahn(sfreq, signal_1d, l_freq, h_freq, dur_lower, dur_upper)

    elif method == "Martin2013":
        spindle_candidates = __martin(sfreq, signal_1d, l_freq, h_freq, dur_lower, dur_upper)

    elif method == "Wamsley2012":
        spindle_candidates = __wamsley(sfreq, signal_1d, l_freq, h_freq, dur_lower, dur_upper)

    elif method == "Wendt2012":
        spindle_candidates = __wendt(sfreq, signal_1d, l_freq, h_freq, dur_lower, dur_upper)

    elif method == "Ferrarelli2007":
        spindle_candidates = __ferrarelli(sfreq, signal_1d, l_freq, h_freq, dur_lower, dur_upper)

    else:
        raise RuntimeError("Chosen method not known. Please pick existing method")

    return spindle_candidates


def __ferrarelli(sfreq, signal_1d, l_freq, h_freq, dur_lower, dur_upper):
    # (Ferrarelli 2007)
    # 12–15 Hz bandpass (already done), rectify signal -> amplitude time series.
    # Upper threshold = 8 × mean amplitude => potential spindle region starts there.
    # Then continues until amplitude < lower threshold = 2 × mean amplitude.
    spindle_candidates = []

    rectified = np.abs(signal_1d)
    mean_amp = np.mean(rectified)
    upper_thr = 8.0 * mean_amp
    lower_thr = 2.0 * mean_amp
    in_spindle = False
    start_idx = None
    for i, val in enumerate(rectified):
        if (not in_spindle) and (val > upper_thr):
            in_spindle = True
            start_idx = i
        elif in_spindle and (val < lower_thr):
            end_idx = i
            duration_sec = (end_idx - start_idx) / sfreq
            if duration_sec >= dur_lower:
                peak_local = np.argmax(rectified[start_idx:end_idx]) + start_idx
                peak_amp = rectified[peak_local]
                spindle_candidates.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'peak_idx': peak_local,
                    'duration': duration_sec,
                    'amplitude': peak_amp
                })
            in_spindle = False

    return spindle_candidates


def __wendt(sfreq, signal_1d, l_freq, h_freq, dur_lower, dur_upper):
    spindle_candidates = []
    rectified = np.abs(signal_1d)
    # Envelope #1 (approx passband 2.25 Hz) -> e.g. ~0.45s kernel
    win_size1 = int(0.45 * sfreq)
    envelope1 = np.convolve(rectified, np.ones(win_size1) / win_size1, mode='same')
    offset1 = 3.0  # in μV
    # Envelope #2 (approx passband 1 Hz) -> ~1s kernel
    win_size2 = int(1.0 * sfreq)
    envelope2 = np.convolve(rectified, np.ones(win_size2) / win_size2, mode='same')
    offset2 = 8.0  # in μV
    thresh_array = np.minimum(envelope1 + offset1, envelope2 + offset2)
    above = rectified > thresh_array
    start_idx = None
    for i, val in enumerate(above):
        if val and start_idx is None:
            start_idx = i
        elif (not val) and (start_idx is not None):
            end_idx = i
            duration_sec = (end_idx - start_idx) / sfreq
            if dur_lower <= duration_sec <= dur_upper:
                peak_local = np.argmax(rectified[start_idx:end_idx]) + start_idx
                peak_amp = rectified[peak_local]
                spindle_candidates.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'peak_idx': peak_local,
                    'duration': duration_sec,
                    'amplitude': peak_amp
                })
            start_idx = None

    return spindle_candidates


def __wamsley(sfreq, signal_1d, l_freq, h_freq, dur_lower, dur_upper):
    # (Wamsley 2012)
    # Discrete sleep spindle events automatically detected via a wavelet-based algorithm
    # covering ~10-16 Hz. For thresholding, the rectified moving average (100 ms window)
    # of the wavelet scale is used. A spindle is identified if the signal
    # exceeds 4.5 * mean amplitude for >= 400 ms.
    #
    # Implementation note:
    # We'll use MNE's Morlet wavelet transform for the relevant frequencies (10-16 Hz),
    # combine power or amplitude across that band, rectify, and smooth with 100 ms.
    #
    # 1. Create a short "dummy" RawArray for wavelet analysis. Or we can do TFR in a direct manner.
    # 2. We'll compute time-frequency representation, sum (or average) power in 10-16 Hz range.
    # 3. Rectify => take the absolute value (or it's already power). Then a 100 ms moving average.
    # 4. threshold = 4.5 * mean
    # 5. events must last >= 0.4 s.
    spindle_candidates = []
    # We'll shape the data into (n_epochs=1, n_channels=1, n_times) for mne tfr_morlet:
    signal_2d = signal_1d[np.newaxis, np.newaxis, :]
    # freq range
    freqs = np.arange(l_freq, h_freq + 1, 1)  # 10 to 16 Hz inclusive
    # We'll use ~8 cycles for Morlet, based on "8-parameter complex Morlet wavelet"
    n_cycles = 8
    # Create an info just for this single channel
    info_tmp = mne.create_info(ch_names=['temp'], sfreq=sfreq, ch_types=['eeg'])
    raw_tmp = mne.io.RawArray(signal_2d[0], info_tmp, verbose=False)
    # Now compute TFR with Morlet
    # We want a single-epoch structure, so create a dummy epoch
    power_data = mne.time_frequency.tfr_array_morlet(
        signal_1d[np.newaxis, np.newaxis, :],  # (1, 1, n_times)
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output='power',
        use_fft=True,
        decim=1,
        verbose=False
    )[0, 0]
    avg_power = np.mean(power_data, axis=0)
    window_samples = int(0.1 * sfreq)
    smoothed = np.convolve(avg_power, np.ones(window_samples) / window_samples, mode='same')
    thr = 4.5 * np.mean(smoothed)
    above = smoothed > thr
    start_idx = None
    for i, val in enumerate(above):
        if val and start_idx is None:
            start_idx = i
        elif (not val) and (start_idx is not None):
            end_idx = i
            duration_sec = (end_idx - start_idx) / sfreq
            if dur_lower <= duration_sec <= dur_upper:
                peak_local = np.argmax(smoothed[start_idx:end_idx]) + start_idx
                peak_amp = smoothed[peak_local]
                spindle_candidates.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'peak_idx': peak_local,
                    'duration': duration_sec,
                    'amplitude': peak_amp
                })
            start_idx = None

    return spindle_candidates


def __martin(sfreq, signal_1d, l_freq, h_freq, dur_lower, dur_upper):
    # (Martin et al. 2013, referencing Schabus et al. 2007)
    # 11-15 Hz bandpass (already done).
    # Then, compute RMS using a 0.25-s window, threshold at 95th percentile.
    spindle_candidates = []
    window_samples = int(0.25 * sfreq)
    squared_signal = signal_1d ** 2
    # moving average of squared => RMS
    moving_avg = np.convolve(
        squared_signal,
        np.ones(window_samples) / window_samples,
        mode='same'
    )
    rms_envelope = np.sqrt(moving_avg)
    amp_threshold = np.percentile(rms_envelope, 95)
    above = rms_envelope > amp_threshold
    start_idx = None
    for i, val in enumerate(above):
        if val and start_idx is None:
            start_idx = i
        elif (not val) and (start_idx is not None):
            end_idx = i
            duration_sec = (end_idx - start_idx) / sfreq
            # Martin2013 requires at least 0.5 s
            if dur_lower <= duration_sec <= dur_upper:
                peak_local = np.argmax(rms_envelope[start_idx:end_idx]) + start_idx
                peak_amp = rms_envelope[peak_local]
                spindle_candidates.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'peak_idx': peak_local,
                    'duration': duration_sec,
                    'amplitude': peak_amp
                })
            start_idx = None

    return spindle_candidates


def __hahn(sfreq, signal_1d, l_freq, h_freq, dur_lower, dur_upper):
    # (Hahn et al. 2020)
    # Hilbert transform -> extract amplitude envelope thresholded at a chosen percentile
    # default percentile=75, duration 0.5-3 s
    spindle_candidates = []
    analytic_signal = hilbert(signal_1d)
    amplitude_envelope = np.abs(analytic_signal)
    # Smooth with a 200 ms moving average
    window_samples = int(0.2 * sfreq)
    smoothed_envelope = np.convolve(
        amplitude_envelope,
        np.ones(window_samples) / window_samples,
        mode='same'
    )
    amp_threshold = np.percentile(smoothed_envelope, 75)
    above = smoothed_envelope > amp_threshold
    start_idx = None
    consecutive_count = 0  # Count consecutive points above threshold
    for i, val in enumerate(above):
        if val:
            if start_idx is None:
                start_idx = i
            consecutive_count += 1
        elif start_idx is not None:
            if consecutive_count >= 2:  # Ensure at least 2 consecutive points
                end_idx = i
                duration_sec = (end_idx - start_idx) / sfreq
                if dur_lower <= duration_sec <= dur_upper:
                    peak_local = np.argmax(smoothed_envelope[start_idx:end_idx]) + start_idx
                    peak_amp = smoothed_envelope[peak_local]
                    spindle_candidates.append({
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'peak_idx': peak_local,
                        'duration': duration_sec,
                        'amplitude': peak_amp
                    })
            # Reset for the next segment
            start_idx = None
            consecutive_count = 0

    return spindle_candidates
