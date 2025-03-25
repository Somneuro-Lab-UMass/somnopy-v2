import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import zscore, t, ttest_rel, circmean


def plot_SO(SO_candidate, raw, grp_thres=0.5, sync_trough=True, interact=True, multilayer=False):
    """
        Plot detected Slow Oscillations (SOs) across EEG channels with event grouping.

        Parameters
        ----------
        SO_candidate : pd.DataFrame
            DataFrame containing detected SO events, including 'ch_id' and 'trough_time'.
        raw : mne.io.Raw
            The EEG raw data object.
        grp_thres : float, default=0.5
            Time threshold (in seconds) for grouping SOs into the same event.
        sync_trough : bool, default=True
            Whether to synchronize signals to the trough.
        interact : bool, default=True
            If True, enables interactive plotting.
        multilayer : bool, default=False
            If True, displays SOs on separate layers.
        """

    # Pick only EEG channels from raw
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    eeg_data = raw.get_data(picks=picks)

    # Get the sampling frequency (sfreq)
    sfreq = raw.info['sfreq']

    # Create a new Info object containing only the EEG channels
    eeg_info = mne.pick_info(raw.info, picks)

    # Apply Butterworth zero-phase filter (0.5-1.25 Hz) to the EEG data
    nyquist = sfreq / 2
    low = 0.5 / nyquist
    high = 1.25 / nyquist
    b, a = butter(2, [low, high], btype='bandpass')
    eeg_data = filtfilt(b, a, eeg_data, axis=1)

    # Create a new column for grouping SOs with None values initially
    SO_candidate['grp_SO'] = None

    # Initialize a unique group counter
    group_id = 1

    # Sort the DataFrame by trough_time for easier sequential comparison
    SO_candidate = SO_candidate.sort_values(by='trough_time').reset_index(drop=True)

    # Loop through each row and assign group IDs
    for idx, row in SO_candidate.iterrows():
        if pd.notnull(SO_candidate.at[idx, 'grp_SO']):
            continue

        SO_candidate.at[idx, 'grp_SO'] = group_id
        group_trough_times = [row['trough_time']]

        while True:
            candidates = (SO_candidate['trough_time'] - np.mean(group_trough_times)).abs() <= grp_thres
            candidates = candidates & (SO_candidate['ch_id'] != row['ch_id']) & SO_candidate['grp_SO'].isnull()

            candidate_indices = SO_candidate.index[candidates].tolist()
            if not candidate_indices:
                break

            for candidate_idx in candidate_indices:
                SO_candidate.at[candidate_idx, 'grp_SO'] = group_id
                group_trough_times.append(SO_candidate.at[candidate_idx, 'trough_time'])

        group_id += 1

    grouped = SO_candidate.groupby('grp_SO')

    if interact:
        plt.ion()
    else:
        plt.ioff()

    for group_id, group in grouped:
        if len(group) <= 1:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        channel_indices = []
        all_signals = []
        avg_trough_time = group['trough_time'].mean()
        start_time = avg_trough_time - 1
        end_time = avg_trough_time + 1
        start_idx = int(start_time * sfreq)
        end_idx = int(end_time * sfreq)

        amplitude_scaling = 10 / np.sqrt(len(group))
        trough_points = []

        for _, row in group.iterrows():
            ch_idx = row['ch_id'] - 1
            trough_time = row['trough_time']
            channel_indices.append(ch_idx)

            signal = zscore(eeg_data[ch_idx, start_idx:end_idx])
            time = np.arange(start_idx, end_idx) / sfreq
            all_signals.append(signal)

            # Plot the signal in green
            ax.plot(time, signal * amplitude_scaling + (len(channel_indices) - 1) * 7, color='green', linewidth=2)

            # Mark the trough point
            trough_idx = int(trough_time * sfreq) - start_idx
            ax.plot(time[trough_idx], signal[trough_idx] * amplitude_scaling + (len(channel_indices) - 1) * 7, 'ko')
            trough_points.append(
                (time[trough_idx], signal[trough_idx] * amplitude_scaling + (len(channel_indices) - 1) * 7))

        # Connect trough points with green lines
        for i in range(len(trough_points) - 1):
            ax.plot([trough_points[i][0], trough_points[i + 1][0]],
                    [trough_points[i][1], trough_points[i + 1][1]], 'g--', linewidth=1)

        ax.axvline(avg_trough_time, color='red', linestyle='--')
        ax.set_yticks([(i - 1) * 7 for i in range(1, len(channel_indices) + 1)])
        ax.set_yticklabels([raw.info['ch_names'][ch_idx] for ch_idx in channel_indices])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channels')

        # Topomap in the upper-right corner
        ax_inset_topomap = inset_axes(ax, width="20%", height="20%", loc='upper right')
        ptp_amplitudes = []
        for ch_idx in range(eeg_data.shape[0]):
            ptp_signal = eeg_data[ch_idx, start_idx:end_idx]
            ptp_amplitude = np.ptp(ptp_signal)
            ptp_amplitudes.append(ptp_amplitude)

        mask = np.zeros(len(eeg_info['ch_names']), dtype=bool)
        mask[channel_indices] = True

        mne.viz.plot_topomap(ptp_amplitudes, eeg_info, axes=ax_inset_topomap, show=False, mask=mask,
                             mask_params=dict(markersize=8, markerfacecolor='r', alpha=0.5), outlines='head',
                             cmap='viridis')

        if len(all_signals) > 1:
            lags = []
            correlations = []
            for i in range(len(all_signals)):
                for j in range(i + 1, len(all_signals)):
                    signal_i = all_signals[i]
                    signal_j = all_signals[j]

                    max_lag_samples = int(0.2 * sfreq)
                    corr = np.correlate(signal_i, signal_j, mode='same')
                    auto_corr_i = np.correlate(signal_i, signal_i, mode='same')[len(signal_i) // 2]
                    auto_corr_j = np.correlate(signal_j, signal_j, mode='same')[len(signal_j) // 2]
                    norm_corr = corr / np.sqrt(auto_corr_i * auto_corr_j)

                    valid_corr = norm_corr[
                                 len(signal_i) // 2 - max_lag_samples: len(signal_i) // 2 + max_lag_samples + 1]
                    max_corr_idx = np.argmax(valid_corr)
                    max_corr = valid_corr[max_corr_idx]
                    lag_ms = (max_corr_idx - max_lag_samples) / sfreq * 1000

                    correlations.append(max_corr)
                    lags.append(lag_ms)

            # Time lag histogram (upper left, square)
            ax_inset_lag = inset_axes(ax, width="10%", height="15%", loc='lower left')
            ax_inset_lag.hist(lags, bins=20, color='green', alpha=0.7)
            ax_inset_lag.set_xlim(-200, 200)
            ax_inset_lag.set_xticks([-200, 0, 200])
            ax_inset_lag.set_yticks([])
            ax_inset_lag.set_title('Lag', fontsize=8)

        if interact:
            plt.pause(0.2)
        else:
            plt.show()


def plot_SP(SP_candidate, raw, grp_thres=0.5, sync_peak=False, interact=False, multilayer=False):
    """
        Plot detected Sleep Spindles (SPs) across EEG channels with event grouping.

        Parameters
        ----------
        SP_candidate : pd.DataFrame
            DataFrame containing detected SP events, including 'ch_id' and 'peak_time'.
        raw : mne.io.Raw
            The EEG raw data object.
        grp_thres : float, default=0.5
            Time threshold (in seconds) for grouping SPs into the same event.
        sync_peak : bool, default=False
            Whether to synchronize signals to the spindle peak.
        interact : bool, default=False
            If True, enables interactive plotting.
        multilayer : bool, default=False
            If True, displays SPs on separate layers.
        """
    # Pick only EEG channels from raw
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    eeg_data = raw.get_data(picks=picks)

    # Get the sampling frequency (sfreq)
    sfreq = raw.info['sfreq']

    # Create a new Info object containing only the EEG channels
    eeg_info = mne.pick_info(raw.info, picks)

    # Apply Butterworth zero-phase filter (12-16 Hz) to the EEG data
    nyquist = sfreq / 2
    low = 12 / nyquist
    high = 16 / nyquist
    b, a = butter(2, [low, high], btype='bandpass')
    eeg_data = filtfilt(b, a, eeg_data, axis=1)

    # Create a new column for grouping SPs with None values initially
    SP_candidate['grp_SP'] = None

    # Initialize a unique group counter
    group_id = 1

    # Sort the DataFrame by peak_time for easier sequential comparison
    SP_candidate = SP_candidate.sort_values(by='peak_time').reset_index(drop=True)

    # Loop through each row and assign group IDs
    for idx, row in SP_candidate.iterrows():
        if pd.notnull(SP_candidate.at[idx, 'grp_SP']):
            continue

        SP_candidate.at[idx, 'grp_SP'] = group_id
        group_peak_times = [row['peak_time']]

        while True:
            candidates = (SP_candidate['peak_time'] - np.mean(group_peak_times)).abs() <= grp_thres
            candidates = candidates & (SP_candidate['ch_id'] != row['ch_id']) & SP_candidate['grp_SP'].isnull()

            candidate_indices = SP_candidate.index[candidates].tolist()
            if not candidate_indices:
                break

            for candidate_idx in candidate_indices:
                SP_candidate.at[candidate_idx, 'grp_SP'] = group_id
                group_peak_times.append(SP_candidate.at[candidate_idx, 'peak_time'])

        group_id += 1

    grouped = SP_candidate.groupby('grp_SP')

    if interact:
        plt.ion()
    else:
        plt.ioff()

    for group_id, group in grouped:
        if len(group) <= 1:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        channel_indices = []
        all_signals = []
        avg_peak_time = group['peak_time'].mean()
        start_time = avg_peak_time - 1
        end_time = avg_peak_time + 1
        start_idx = int(start_time * sfreq)
        end_idx = int(end_time * sfreq)

        amplitude_scaling = 10 / np.sqrt(len(group))
        peak_points = []

        for _, row in group.iterrows():
            ch_idx = row['ch_id'] - 1
            peak_time = row['peak_time']
            channel_indices.append(ch_idx)

            signal = zscore(eeg_data[ch_idx, start_idx:end_idx])
            time = np.arange(start_idx, end_idx) / sfreq
            all_signals.append(signal)

            # Plot the signal in green
            ax.plot(time, signal * amplitude_scaling + (len(channel_indices) - 1) * 7, color='green', linewidth=2)

            # Mark the peak point
            peak_idx = int(peak_time * sfreq) - start_idx
            ax.plot(time[peak_idx], signal[peak_idx] * amplitude_scaling + (len(channel_indices) - 1) * 7, 'ko')
            peak_points.append((time[peak_idx], signal[peak_idx] * amplitude_scaling + (len(channel_indices) - 1) * 7))

        # Connect peak points with green lines
        for i in range(len(peak_points) - 1):
            ax.plot([peak_points[i][0], peak_points[i + 1][0]],
                    [peak_points[i][1], peak_points[i + 1][1]], 'g--', linewidth=1)

        ax.axvline(avg_peak_time, color='red', linestyle='--')
        ax.set_yticks([(i - 1) * 7 for i in range(1, len(channel_indices) + 1)])
        ax.set_yticklabels([raw.info['ch_names'][ch_idx] for ch_idx in channel_indices])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channels')

        # Topomap in the upper-right corner
        ax_inset_topomap = inset_axes(ax, width="20%", height="20%", loc='upper right')
        ptp_amplitudes = []
        for ch_idx in range(eeg_data.shape[0]):
            ptp_signal = eeg_data[ch_idx, start_idx:end_idx]
            ptp_amplitude = np.ptp(ptp_signal)
            ptp_amplitudes.append(ptp_amplitude)

        mask = np.zeros(len(eeg_info['ch_names']), dtype=bool)
        mask[channel_indices] = True

        mne.viz.plot_topomap(ptp_amplitudes, eeg_info, axes=ax_inset_topomap, show=False, mask=mask,
                             mask_params=dict(markersize=8, markerfacecolor='r', alpha=0.5), outlines='head',
                             cmap='viridis')

        if len(all_signals) > 1:
            lags = []
            correlations = []
            for i in range(len(all_signals)):
                for j in range(i + 1, len(all_signals)):
                    signal_i = all_signals[i]
                    signal_j = all_signals[j]

                    max_lag_samples = int(0.2 * sfreq)
                    corr = np.correlate(signal_i, signal_j, mode='same')
                    auto_corr_i = np.correlate(signal_i, signal_i, mode='same')[len(signal_i) // 2]
                    auto_corr_j = np.correlate(signal_j, signal_j, mode='same')[len(signal_j) // 2]
                    norm_corr = corr / np.sqrt(auto_corr_i * auto_corr_j)

                    valid_corr = norm_corr[
                                 len(signal_i) // 2 - max_lag_samples: len(signal_i) // 2 + max_lag_samples + 1]
                    max_corr_idx = np.argmax(valid_corr)
                    max_corr = valid_corr[max_corr_idx]
                    lag_ms = (max_corr_idx - max_lag_samples) / sfreq * 1000

                    correlations.append(max_corr)
                    lags.append(lag_ms)

            # Time lag histogram (upper left, square)
            ax_inset_lag = inset_axes(ax, width="10%", height="15%", loc='lower left')
            ax_inset_lag.hist(lags, bins=20, color='green', alpha=0.7)
            ax_inset_lag.set_xlim(-200, 200)
            ax_inset_lag.set_xticks([-200, 0, 200])
            ax_inset_lag.set_yticks([])
            ax_inset_lag.set_title('Lag', fontsize=8)

        if interact:
            plt.pause(0.2)
        else:
            plt.show()


def compute_pac_metrics(events, raw, window_size, eeg_data_dict, n_bins, sfreq, stage_dur):
    """
    Compute PAC metrics for a given set of events (subset for a channel and optionally a stage).
    Returns a dictionary with metrics.
    """
    peaks_so_phase = []
    valid_events_count = 0
    for _, event in events.iterrows():
        ch = event['ch_name']
        so_trough_time = event['SO_trough_time']
        sp_peak_time = event['SP_peak_time']
        trough_idx = int(so_trough_time * sfreq)
        sp_idx = int(sp_peak_time * sfreq)
        if trough_idx - window_size < 0 or trough_idx + window_size >= raw.n_times:
            continue
        local_window = eeg_data_dict[ch][trough_idx - window_size: trough_idx + window_size]
        analytic_signal = hilbert(local_window)
        phase_data = np.angle(analytic_signal)
        relative_spindle_peak_idx = sp_idx - (trough_idx - window_size)
        if 0 <= relative_spindle_peak_idx < len(phase_data):
            peaks_so_phase.append(phase_data[relative_spindle_peak_idx])
            valid_events_count += 1
    peaks_so_phase = np.array(peaks_so_phase)
    if len(peaks_so_phase) > 0:
        pp = np.angle(np.exp(1j * circmean(peaks_so_phase)))
        amp_hist, _ = np.histogram(peaks_so_phase, bins=np.linspace(-np.pi, np.pi, n_bins + 1), density=True)
        U = np.ones(n_bins) / n_bins
        P = amp_hist / np.sum(amp_hist)
        eps = 1e-10  # Small constant to prevent log(0)
        P = np.where(P == 0, eps, P)  # Replace zero probabilities
        D_KL = np.nansum(P * np.log(P / U))
        mi = D_KL / np.log(n_bins) if np.log(n_bins) > 0 else 0
        mvl = np.abs(np.mean(np.exp(1j * peaks_so_phase)))
        R = mvl
        z = len(peaks_so_phase) * R ** 2
        p_value = np.exp(-z) * (1 + z)
    else:
        pp = np.nan
        mi = np.nan
        mvl = np.nan
        z = np.nan
        p_value = np.nan
    # Compute coupling density: for a single channel, use (number of events / stage duration) * 30.
    cp_count = events.shape[0]
    cp_density = (cp_count / stage_dur) * 30 if stage_dur > 0 else 0
    return {
        'preferred_phase': pp,
        'modulation_index': mi,
        'mean_vector_length': mvl,
        'rayleigh_z': z,
        'coupling_density': cp_density
    }


def pac(raw, coupled_events, event_summary, verbose=True):
    """
    Compute Peri-Event Time Histogram (PETH) and Phase-Amplitude Coupling (PAC)
    between SO and spindle events in a single function.

    Parameters:
    - coupled_events (pd.DataFrame): DataFrame containing SO-Spindle coupling events.
    - raw (mne.io.Raw): Raw EEG data.

    Prints:
    - Preferred Phase, Modulation Index, Mean Vector Length, Rayleigh z, and p-value.

    Plots:
    - Left: PETH histogram.
    - Right: PAC polar plot.
    """
    sfreq = raw.info['sfreq']  # Sampling frequency
    window = 1.2  # Time window for PETH
    bins = 20  # Bins for PETH histogram
    window_size = int(window * sfreq)  # Convert ±1.2s window into samples
    time_diffs = coupled_events['Time_diff'].dropna().values  # Remove NaN values

    # **Compute histogram (normalized)**
    peth_bin_edges = np.linspace(-window, window, bins + 1)

    # **Extract SO waveform around each SO trough (±1.2s)**
    time_range = np.arange(-window, window, 1 / sfreq)
    avg_so_wave = np.zeros_like(time_range)

    num_valid_events = 0
    SO_waveform = []
    for _, event in coupled_events.iterrows():
        ch_name = event['ch_name']
        so_trough_time = event['SO_trough_time']

        trough_idx = int(so_trough_time * sfreq)
        start_idx = trough_idx - window_size
        end_idx = trough_idx + window_size

        if start_idx >= 0 and end_idx < raw.n_times:
            eeg_data = raw.get_data(picks=ch_name)[0]
            avg_so_wave += eeg_data[start_idx:end_idx]
            num_valid_events += 1

    if num_valid_events > 0:
        avg_so_wave /= num_valid_events  # Compute mean SO waveform

    avg_so_wave *= 1e6

    SO_waveform.append({'stage': 'all', 'so_waveform': ','.join(map(str, avg_so_wave))})

    # **PAC Computation**
    unique_channels = coupled_events['ch_name'].unique()
    eeg_data_dict = {ch: raw.get_data(picks=ch)[0] for ch in unique_channels}

    so_trough_indices = (coupled_events['SO_trough_time'] * sfreq).astype(int)
    spindle_peak_indices = (coupled_events['SP_peak_time'] * sfreq).astype(int)

    valid_mask = (so_trough_indices >= window_size) & (so_trough_indices < raw.n_times - window_size)

    peaks_so_phase_all = []

    for ch, so_idx, sp_idx in zip(coupled_events['ch_name'][valid_mask],
                                  so_trough_indices[valid_mask],
                                  spindle_peak_indices[valid_mask]):
        local_window = eeg_data_dict[ch][so_idx - window_size: so_idx + window_size]
        analytic_signal = hilbert(local_window)
        phase_data = np.angle(analytic_signal)

        relative_spindle_peak_idx = sp_idx - (so_idx - window_size)
        if 0 <= relative_spindle_peak_idx < len(phase_data):
            peaks_so_phase_all.append(phase_data[relative_spindle_peak_idx])

    peaks_so_phase_all = np.array(peaks_so_phase_all)

    if len(peaks_so_phase_all) == 0:
        print("No valid SO phases found.")
        return

    # Always calculate both all-stage data and individual-stage data
    stage_mapping = {0: "Wake", 1: "N1", 2: "N2", 3: "SWS", 4: "REM"}
    coupling_results = []

    # Compute all-stage metrics
    peaks_so_phase_all = np.array(peaks_so_phase_all)
    n_bins = 20
    pac_bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    U = np.ones(n_bins) / n_bins

    if len(peaks_so_phase_all) > 0:
        pp_all = np.angle(np.exp(1j * circmean(peaks_so_phase_all)))
        amp_hist_all, _ = np.histogram(peaks_so_phase_all, bins=np.linspace(-np.pi, np.pi, n_bins + 1), density=True)
        P_all = amp_hist_all / np.sum(amp_hist_all)
        eps = 1e-10  # Small constant to prevent log(0)
        P_all = np.where(P_all == 0, eps, P_all)  # Replace zero probabilities
        D_KL_all = np.nansum(P_all * np.log(P_all / U))
        mi_all = D_KL_all / np.log(n_bins) if np.log(n_bins) > 0 else 0
        mvl_all = np.abs(np.mean(np.exp(1j * peaks_so_phase_all)))
        R_all = np.abs(np.mean(np.exp(1j * peaks_so_phase_all)))
        z_all = len(peaks_so_phase_all) * R_all ** 2
        p_all = np.exp(-z_all) * (1 + z_all)

        cp_count_all = coupled_events.shape[0]
        total_stage_dur = event_summary.loc[event_summary['stage'] == 'all', 'stage_dur'].values[0]
        num_channels_all = coupled_events['ch_name'].nunique()
        cp_density_all = (cp_count_all / (total_stage_dur * num_channels_all)) * 30 if total_stage_dur > 0 else 0

        coupling_results.append({
            'stage': 'all',
            'preferred_phase': pp_all,
            'modulation_index': mi_all,
            'mean_vector_length': mvl_all,
            'rayleigh_z': z_all,
            'coupling_density': cp_density_all
        })

    # Compute individual-stage metrics
    for stage in coupled_events['stage'].unique():
        stage_name = stage_mapping.get(stage, f"Unknown ({stage})")
        stage_data = coupled_events[coupled_events['stage'] == stage]
        peaks_so_phase_stage = []
        avg_so_wave_stage = np.zeros_like(time_range)
        num_valid_events_stage = 0

        so_trough_indices_stage = (stage_data['SO_trough_time'] * sfreq).astype(int)
        spindle_peak_indices_stage = (stage_data['SP_peak_time'] * sfreq).astype(int)

        valid_mask_stage = (so_trough_indices_stage >= window_size) & (
                so_trough_indices_stage < raw.n_times - window_size)

        for ch, so_idx, sp_idx in zip(stage_data['ch_name'][valid_mask_stage],
                                      so_trough_indices_stage[valid_mask_stage],
                                      spindle_peak_indices_stage[valid_mask_stage]):
            local_window = eeg_data_dict[ch][so_idx - window_size: so_idx + window_size]
            analytic_signal = hilbert(local_window)
            phase_data = np.angle(analytic_signal)

            relative_spindle_peak_idx = sp_idx - (so_idx - window_size)
            if 0 <= relative_spindle_peak_idx < len(phase_data):
                peaks_so_phase_stage.append(phase_data[relative_spindle_peak_idx])

        peaks_so_phase_stage = np.array(peaks_so_phase_stage)

        if len(peaks_so_phase_stage) > 0:
            pp_stage = np.angle(np.exp(1j * circmean(peaks_so_phase_stage)))
            amp_hist_stage, _ = np.histogram(peaks_so_phase_stage, bins=np.linspace(-np.pi, np.pi, n_bins + 1),
                                             density=True)
            P_stage = amp_hist_stage / np.sum(amp_hist_stage)
            eps = 1e-10  # Small constant to prevent log(0)
            P_stage = np.where(P_stage == 0, eps, P_stage)  # Replace zero probabilities
            D_KL_stage = np.nansum(P_stage * np.log(P_stage / U))
            mi_stage = D_KL_stage / np.log(n_bins) if np.log(n_bins) > 0 else 0
            mvl_stage = np.abs(np.mean(np.exp(1j * peaks_so_phase_stage)))
            R_stage = np.abs(np.mean(np.exp(1j * peaks_so_phase_stage)))
            z_stage = len(peaks_so_phase_stage) * R_stage ** 2
            p_stage = np.exp(-z_stage) * (1 + z_stage)

            cp_count = coupled_events.groupby('stage').size()
            stage_dur = event_summary.loc[event_summary['stage'] == stage, 'stage_dur'].values[0]
            num_cp_event = cp_count.get(stage, 0)
            num_channels_stage = stage_data['ch_name'].nunique()
            cp_density_stage = (num_cp_event / (stage_dur * num_channels_stage)) * 30 if stage_dur > 0 else 0

            coupling_results.append({
                'stage': stage,
                'preferred_phase': pp_stage,
                'modulation_index': mi_stage,
                'mean_vector_length': mvl_stage,
                'rayleigh_z': z_stage,
                'coupling_density': cp_density_stage
            })

            if verbose:
                print(f"\033[1mStage {stage_name}:\033[0m")
                print(f"Preferred Phase:       {round(pp_stage, 3)} rad")
                print(f"Modulation Index:      {mi_stage:.3f}")
                print(f"Mean Vector Length:    {mvl_stage:.3f}")
                print(f"Rayleigh z:            {z_stage:.3f}")
                print(f"p-value:               {p_stage:.3f}")
                print(f"Coupling Density:      {cp_density_stage:.3f}")

        for _, event in stage_data.iterrows():
            ch_name = event['ch_name']
            so_trough_time = event['SO_trough_time']

            trough_idx = int(so_trough_time * sfreq)
            start_idx = trough_idx - window_size
            end_idx = trough_idx + window_size

            if start_idx >= 0 and end_idx < raw.n_times:
                eeg_data = raw.get_data(picks=ch_name)[0]
                avg_so_wave_stage += eeg_data[start_idx:end_idx]
                num_valid_events_stage += 1

        if num_valid_events_stage > 0:
            avg_so_wave_stage /= num_valid_events_stage
        avg_so_wave_stage *= 1e6

        SO_waveform.append({'stage': stage, 'so_waveform': ','.join(map(str, avg_so_wave_stage))})

    # Merge results into event_summary
    coupling_df = pd.DataFrame(coupling_results)
    coupling_df['channel'] = 'all'

    per_channel_results = []
    # Prepare a dictionary for stage durations from event_summary input
    stage_dur_dict = {row['stage']: row['stage_dur'] for _, row in event_summary.iterrows()}
    # Process each channel for aggregated (all stages) metrics
    for ch in coupled_events['ch_name'].unique():
        events_ch_all = coupled_events[coupled_events['ch_name'] == ch]
        if events_ch_all.empty:
            continue
        # Get stage duration for 'all'
        total_stage_dur = stage_dur_dict.get('all', np.nan)
        metrics_all = compute_pac_metrics(events_ch_all, raw, window_size, eeg_data_dict, n_bins, sfreq,
                                          total_stage_dur)
        metrics_all.update({'stage': 'all', 'channel': ch})
        per_channel_results.append(metrics_all)
        # Process each stage for this channel
        for stage in events_ch_all['stage'].unique():
            events_ch_stage = events_ch_all[events_ch_all['stage'] == stage]
            stage_dur = stage_dur_dict.get(stage, np.nan)
            metrics_stage = compute_pac_metrics(events_ch_stage, raw, window_size, eeg_data_dict, n_bins, sfreq,
                                                stage_dur)
            metrics_stage.update({'stage': stage, 'channel': ch})
            per_channel_results.append(metrics_stage)
    per_channel_coupling_df = pd.DataFrame(per_channel_results)
    coupling_all = pd.concat([coupling_df, per_channel_coupling_df], ignore_index=True)
    event_summary = pd.merge(event_summary, coupling_all, on=['stage', 'channel'], how='outer')

    if verbose:
        # **Plot PETH + PAC**
        fig = plt.figure(figsize=(8, 3.6))
        gs = plt.GridSpec(1, 2, width_ratios=[1.9, 1])

        # Left: PETH
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(time_diffs, bins=peth_bin_edges, facecolor='none', edgecolor='#4682B4', linewidth=2, alpha=1,
                 density=True, label='PETH')
        ax1.axvline(0, color='black', lw=2, linestyle='--', label='SO Trough')
        ax1.set_xlabel('Time relative to SO Trough (s)')
        ax1.set_ylabel('Density')
        ax1.set_title('Peri-Event Time Histogram (PETH)', pad=10)

        ax2 = ax1.twinx()
        ax2.plot(time_range, avg_so_wave, color='black', linewidth=2, label='Averaged SO Waveform')
        ax2.set_ylim([avg_so_wave.min() * 1.5, avg_so_wave.max() * 1.2])
        ax2.spines['right'].set_visible(False)
        ax2.set_yticklabels([])
        ax2.set_yticks([])

        # Right: PAC polar plot
        theta = (pac_bin_edges[:-1] + pac_bin_edges[1:]) / 2
        r = amp_hist_all

        ax3 = fig.add_subplot(gs[0, 1], projection='polar')
        ax3.bar(theta, r, width=(2 * np.pi / n_bins), facecolor='none', edgecolor='#4682B4', linewidth=2, alpha=1)
        ax3.plot([pp_all, pp_all], [0, max(r)], color='k', lw=2, linestyle='--', label="Preferred Phase")
        ax3.set_yticks([])
        ax3.yaxis.set_visible(False)
        ax3.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])  # Set only four tick labels
        ax3.set_xticklabels(['0 \n peak', r'$\pi/2$', r'$\pi$', r'$-\pi/2$'])  # Custom labels

        ax3.set_theta_zero_location("E")  # 'East' → Places 0 on the right
        ax3.set_theta_direction(1)  # Counterclockwise direction

        ax3.set_title('Coupling Phase Polar Plot')

        plt.tight_layout()
        plt.show()

    waveform_df = pd.DataFrame(SO_waveform)
    return event_summary, waveform_df


def pac_grp(raw, cp_event_grp, event_summary_grp, SO_waveform, stage='all', chan='all'):
    """
    Compute and plot:
    1. Peri-Event Time Histogram (PETH) with averaged SO waveform, with a permutation test
       to assess the temporal stability of the histogram.
    2. Group-level preferred phase circular plot, where the preferred phase indicator line
       extends from 0 to the mean vector length computed across subjects.

    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data.
    cp_event_grp : pd.DataFrame
        DataFrame containing SO-Spindle coupling events. Must include columns:
        'Time_diff', 'SO_trough_time', 'ch_name', 'subject', and 'stage'.
    event_summary_grp : pd.DataFrame
        DataFrame containing participant-level PAC metrics
    stage : str, optional
        Specify which stage to plot. Enter 'all' to aggregate across all stages, or specify
        a stage such as 'Wake', 'N1', 'N2', 'SWS', or 'REM' to generate a plot for that specific stage.

    Notes:
    ------
    To test the temporal stability of the PETHs, we implemented a permutation test:
      - For each subject, the temporal order of the PETH bins is randomly shuffled 1000 times.
      - The resulting surrogate histograms are averaged for each individual.
      - Paired one-tailed t-tests are performed for each bin comparing the original and surrogate histograms
      - Controled multiple comparisons by a cluster-based permutation test with 5000 permutations.
    """

    stage_mapping = {0: "Wake", 1: "N1", 2: "N2", 3: "SWS", 4: "REM"}
    sfreq = raw.info['sfreq']
    window = 1.2
    bins = 20
    window_size = int(window * sfreq)

    # Determine stages to iterate over
    if stage == 'all':
        stages = ['all']
        cp_event_grp = cp_event_grp.copy()
        event_summary_grp = event_summary_grp[event_summary_grp['stage'] == 'all']
    else:
        inv_stage_mapping = {v: k for k, v in stage_mapping.items()}
        if stage not in inv_stage_mapping:
            raise ValueError("Invalid stage input. Must be one of 'all', 'Wake', 'N1', 'N2', 'SWS', 'REM'.")
        stage_int = inv_stage_mapping[stage]
        stages = [stage_int]

    for stg in stages:
        if stg == 'all':
            stage_name = "Global"
            stage_events = cp_event_grp
            stage_pac_metrics = event_summary_grp
        else:
            stage_name = stage_mapping.get(stg, f"Unknown ({stg})")
            stage_events = cp_event_grp[cp_event_grp['stage'] == stg]
            stage_pac_metrics = event_summary_grp[event_summary_grp['stage'] == stg]

        if chan == 'all':
            stage_pac_metrics = stage_pac_metrics[stage_pac_metrics['channel'] == 'all']
        else:
            if isinstance(chan, list):
                stage_pac_metrics = stage_pac_metrics[stage_pac_metrics['channel'].isin(chan)]
            else:
                stage_pac_metrics = stage_pac_metrics[stage_pac_metrics['channel'] == chan]

            if stage_pac_metrics.empty:
                print(f"Error: Selected channel(s) {chan} not found in event_summary_grp.")
                return

        if stage_events.empty or stage_pac_metrics.empty:
            continue

        time_diffs = stage_events['Time_diff'].dropna().values
        peth_bin_edges = np.linspace(-window, window, bins + 1)
        time_range = np.arange(-window, window, 1 / sfreq)

        # Extract SO waveforms from DataFrame instead of recomputing from raw
        if stg == 'all':
            stage_waveforms = SO_waveform[SO_waveform['stage'] == 'all']['so_waveform']
        else:
            stage_waveforms = SO_waveform[SO_waveform['stage'] == stg]['so_waveform']

        # Convert waveform strings into numerical arrays and average across subjects
        if not stage_waveforms.empty:
            waveform_arrays = np.array([np.fromstring(w, sep=',') for w in stage_waveforms])
            avg_so_wave = waveform_arrays.mean(axis=0)  # Average across subjects
        else:
            avg_so_wave = np.zeros_like(time_range)  # Fallback to zeros if no data

        subjects = stage_events['subject'].unique()
        n_subjects = len(subjects)
        subject_histograms = np.zeros((n_subjects, bins))
        for i, subj in enumerate(subjects):
            subj_data = stage_events[stage_events['subject'] == subj]['Time_diff'].dropna().values
            hist, _ = np.histogram(subj_data, bins=peth_bin_edges, density=True)
            subject_histograms[i, :] = hist
        group_original = subject_histograms.mean(axis=0)

        n_permutations = 1000
        subject_surrogate_avg = np.zeros((n_subjects, bins))
        for i in range(n_subjects):
            surrogate_all = np.zeros((n_permutations, bins))
            for j in range(n_permutations):
                surrogate_all[j, :] = np.random.permutation(subject_histograms[i, :])
            subject_surrogate_avg[i, :] = surrogate_all.mean(axis=0)

        diff = subject_histograms - subject_surrogate_avg
        t_vals, p_vals = ttest_rel(subject_histograms, subject_surrogate_avg, axis=0)

        group_surrogate_avg = subject_surrogate_avg.mean(axis=0)  # Average across subjects

        alpha = 0.05
        t_thresh = t.ppf(1 - alpha, df=n_subjects - 1)

        def find_clusters_t(t_values, t_thresh):
            clusters = []
            cluster_stats = []
            current_cluster = []
            for idx in range(len(t_values)):
                if t_values[idx] > t_thresh:
                    current_cluster.append(idx)
                else:
                    if current_cluster:
                        clusters.append(current_cluster)
                        cluster_stats.append(np.sum(t_values[current_cluster]))
                        current_cluster = []
            if current_cluster:
                clusters.append(current_cluster)
                cluster_stats.append(np.sum(t_values[current_cluster]))
            return clusters, cluster_stats

        observed_clusters, observed_cluster_stats = find_clusters_t(t_vals, t_thresh)

        n_cluster_permutations = 5000
        max_cluster_stats_perm = np.zeros(n_cluster_permutations)

        for perm in range(n_cluster_permutations):
            sign_flip = np.random.choice([1, -1], size=n_subjects)
            diff_perm = diff * sign_flip[:, None]
            t_perm = np.zeros(bins)
            for b in range(bins):
                d = diff_perm[:, b]
                std_d = np.std(d, ddof=1)
                if std_d == 0:
                    t_perm[b] = 0
                else:
                    t_perm[b] = np.mean(d) / (std_d / np.sqrt(n_subjects))
            clusters_perm, cluster_stats_perm = find_clusters_t(t_perm, t_thresh)
            if cluster_stats_perm:
                max_cluster_stats_perm[perm] = np.max(cluster_stats_perm)
            else:
                max_cluster_stats_perm[perm] = 0

        significant_bins = np.zeros(bins, dtype=bool)
        for cluster, cluster_stat in zip(observed_clusters, observed_cluster_stats):
            p_corr = np.mean(max_cluster_stats_perm >= cluster_stat)
            if p_corr < 0.05:
                significant_bins[cluster] = True

        fig = plt.figure(figsize=(8, 3.6))
        gs = plt.GridSpec(1, 2, width_ratios=[1.9, 1])

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(time_diffs, bins=peth_bin_edges, facecolor='none', edgecolor='#4682B4',
                 linewidth=2, alpha=1, density=True, label='PETH')
        ax1.axvline(0, color='black', lw=2, linestyle='--', label='SO Trough')
        bin_centers = (peth_bin_edges[:-1] + peth_bin_edges[1:]) / 2
        ax1.plot(bin_centers, group_surrogate_avg, color='grey', linewidth=2, label='Surrogate Average')
        ax1.set_xlabel('Time relative to SO Trough (s)')
        ax1.set_ylabel('Density')

        ax2 = ax1.twinx()
        ax2.plot(time_range, avg_so_wave, color='black', linewidth=2, label='Averaged SO Waveform')
        ax2.set_ylim([avg_so_wave.min() * 1.5, avg_so_wave.max() * 1.2])
        ax2.spines['right'].set_visible(False)
        ax2.set_yticklabels([])
        ax2.set_yticks([])

        bin_centers = (peth_bin_edges[:-1] + peth_bin_edges[1:]) / 2
        ylim = ax1.get_ylim()
        y_offset = ylim[1] * 1.05
        for i, sig in enumerate(significant_bins):
            if sig:
                ax1.plot(bin_centers[i], y_offset, 'o', color='red')

        ax3 = fig.add_subplot(gs[0, 1], projection='polar')
        pp_data = stage_pac_metrics['preferred_phase'].dropna().values
        if len(pp_data) > 0:
            mean_phase_vector = np.mean(np.exp(1j * pp_data))
            mean_phase = np.angle(mean_phase_vector)
            mvl = np.abs(mean_phase_vector)
            ax3.plot([mean_phase, mean_phase], [0, mvl], color='k', lw=2, label="Mean Preferred Phase")
            ax3.scatter(pp_data, np.ones_like(pp_data), color='#4682B4', s=50, label='Participants', alpha=0.8)

        ax3.set_yticks([])
        ax3.yaxis.set_visible(False)
        ax3.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax3.set_xticklabels(['0 \n peak', r'$\pi/2$', r'$\pi$', r'$-\pi/2$'])
        ax3.set_theta_zero_location("E")
        ax3.set_theta_direction(1)

        plt.tight_layout()
        plt.show()


def event_lock(raw, SO_candidates, SP_candidates, event_summary, window=1.5, verbose=True):
    """
    Identify spindles occurring within ±1.5 s of each detected SO trough.

    Parameters:
    - raw (mne.io.Raw): EEG raw data.
    - SO_candidates (pd.DataFrame): DataFrame containing detected SO events.
    - SP_candidates (pd.DataFrame): DataFrame containing detected spindle events.
    - sfreq (float): Sampling frequency.
    - window (float): Time window (default: ±1.5s around SO trough).

    Returns:
    - pd.DataFrame: Coupled SO-Spindle events.
    """
    stage_mapping = {0: "Wake", 1: "N1", 2: "N2", 3: "SWS", 4: "REM"}
    merged = SO_candidates.merge(SP_candidates, on=['ch_name', 'stage'])
    filtered = merged.query("peak_time >= trough_time - @window and peak_time <= trough_time + @window").copy()
    filtered['Time_diff'] = filtered['peak_time'] - filtered['trough_time']
    filtered['Abs_Time_diff'] = filtered['Time_diff'].abs()

    total_spindles = len(SP_candidates)
    total_SOs = len(SO_candidates)
    total_couplings = len(filtered)

    cp_percent = []
    SPcSO_all = (total_couplings / total_spindles) if total_spindles > 0 else 0
    SOcSP_all = (total_couplings / total_SOs) if total_SOs > 0 else 0
    cp_percent.append({'stage': 'all', 'SPcSO': SPcSO_all, 'SOcSP': SOcSP_all})

    for stage in filtered['stage'].unique():
        total_stage_spindles = len(SP_candidates[SP_candidates['stage'] == stage])
        total_stage_SOs = len(SO_candidates[SO_candidates['stage'] == stage])
        total_stage_couplings = len(filtered[filtered['stage'] == stage])

        SPcSO_stage = (total_stage_couplings / total_stage_spindles) if total_stage_spindles > 0 else 0
        SOcSP_stage = (total_stage_couplings / total_stage_SOs) if total_stage_SOs > 0 else 0

        stage_name = stage_mapping.get(stage, f"Unknown ({stage})")
        cp_percent.append({'stage': stage, 'SPcSO': SPcSO_stage, 'SOcSP': SOcSP_stage})
        if verbose:
            print(f"Spindles coupled with SOs in stage \033[1m{stage_name}\033[0m: {SPcSO_stage * 100:.2f}%")
            print(f"SOs coupled with spindles in stage \033[1m{stage_name}\033[0m: {SOcSP_stage * 100:.2f}%")

    cp_percent_df = pd.DataFrame(cp_percent)
    cp_percent_df['channel'] = 'all'

    cp_percent_channel_list = []
    # Group the filtered events by stage and channel
    for (stage, ch), group in filtered.groupby(['stage', 'ch_name']):
        couplings = group.shape[0]
        sp_channel_count = SP_candidates[(SP_candidates['stage'] == stage) & (SP_candidates['ch_name'] == ch)].shape[0]
        so_channel_count = SO_candidates[(SO_candidates['stage'] == stage) & (SO_candidates['ch_name'] == ch)].shape[0]
        SPcSO_channel = (couplings / sp_channel_count) if sp_channel_count > 0 else 0
        SOcSP_channel = (couplings / so_channel_count) if so_channel_count > 0 else 0
        cp_percent_channel_list.append({
            'stage': stage,
            'ch_name': ch,
            'SPcSO': SPcSO_channel,
            'SOcSP': SOcSP_channel
        })
    cp_percent_channel_df = pd.DataFrame(cp_percent_channel_list)
    # Rename 'ch_name' to 'channel' for consistency
    cp_percent_channel_df.rename(columns={'ch_name': 'channel'}, inplace=True)

    # Add channel-level coupling metrics for stage 'all'
    cp_percent_all_channel_list = []
    # Use filtered coupling counts for all channels regardless of stage
    for ch, couplings in filtered['ch_name'].value_counts().items():
        sp_channel_count = SP_candidates[SP_candidates['ch_name'] == ch].shape[0]
        so_channel_count = SO_candidates[SO_candidates['ch_name'] == ch].shape[0]
        SPcSO_channel = (couplings / sp_channel_count) if sp_channel_count > 0 else 0
        SOcSP_channel = (couplings / so_channel_count) if so_channel_count > 0 else 0
        cp_percent_all_channel_list.append({
            'stage': 'all',
            'channel': ch,
            'SPcSO': SPcSO_channel,
            'SOcSP': SOcSP_channel
        })
    cp_percent_all_channel_df = pd.DataFrame(cp_percent_all_channel_list)

    cp_percent_all = pd.concat([cp_percent_df, cp_percent_channel_df, cp_percent_all_channel_df], ignore_index=True)
    print('event_summary\n', event_summary)
    print('cp_percent_all\n', cp_percent_all)
    event_summary = pd.merge(event_summary, cp_percent_all, left_on=['stage', 'channel_y'],
                             right_on=['stage', 'channel'], how='outer')

    # ** Per-Channel Calculation using the same method **
    channel_counts = filtered['ch_name'].value_counts()
    total_spindles_per_channel = SP_candidates['ch_name'].value_counts()
    total_SOs_per_channel = SO_candidates['ch_name'].value_counts()

    spindle_ratio_per_channel = (channel_counts / total_spindles_per_channel * 100).reindex(
        raw.info['ch_names']).fillna(0)
    SO_ratio_per_channel = (channel_counts / total_SOs_per_channel * 100).reindex(raw.info['ch_names']).fillna(0)

    SO_measure = pd.DataFrame({
        'ch_name': raw.info['ch_names'],
        'Spindles_Coupled_with_SOs': spindle_ratio_per_channel.values
    })

    # Generate topomap
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    SO_measure = SO_measure.set_index('ch_name').reindex([raw.ch_names[pick] for pick in picks]).reset_index()

    if verbose:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        im, _ = mne.viz.plot_topomap(SO_measure['Spindles_Coupled_with_SOs'].values, raw.info, axes=ax, show=False,
                                     cmap="Blues")

        im.set_clim(vmin=0, vmax=100)
        plt.colorbar(im, ax=ax, format="%.0f%%")
        ax.set_title('Spindles Coupled with SOs (%)')
        plt.show()

    return filtered[['ch_name', 'stage', 'trough_time', 'peak_time', 'amplitude', 'Time_diff']].rename(
        columns={'trough_time': 'SO_trough_time', 'peak_time': 'SP_peak_time',
                 'amplitude': 'SP_amplitude'}), event_summary
