import math
import os

import pandas as pd

from somnopy.polysomnography import PolySomnoGraphy


# TODO: Analyze Melissa's data for SWA and Spindles, data in infantro1/data/psg/datatobeanalyzed, n y: is in presentations/flux 2025

# TODO: Add custom method for spindle detection

# TODO: can compare with Melissa's data

# TODO: Can we look at the spindle frequency


def get_sosp(psg: PolySomnoGraphy, file_name, interest_stage=('N2', 'SWS'),
             sp_method='Hahn2020',
             so_method='Staresina', coupling=False,
             filter_freq=None, duration=None, filter_type: str = 'fir', l_freq: float = None,
             h_freq: float = None, dur_lower: float = None, dur_upper: float = None,
             baseline: bool = True, verbose: bool = True):
    """
        Detect slow oscillations (SOs) and sleep spindles (SPs) in a given EEG recording.

        Parameters
        ----------
        psg : PolySomnoGraphy
            An instance of PolySomnoGraphy containing the EEG and hypnogram data.
        file_name : str
            Name of the participant or EEG recording file.
        interest_stage : tuple, default=('N2', 'SWS')
            Sleep stages to analyze.
        sp_method : str, default='Hahn2020'
            Method for spindle detection.
        so_method : str, default='Staresina'
            Method for slow oscillation detection.
        coupling : bool, default=False
            Whether to perform phase-amplitude coupling analysis.
        filter_freq : tuple, optional
            Frequency range for SO detection.
        duration : tuple, optional
            Duration range for SO detection.
        filter_type : str, default='fir'
            Filtering method ('fir' or 'iir').
        l_freq : float, optional
            Low-frequency cutoff for spindle detection.
        h_freq : float, optional
            High-frequency cutoff for spindle detection.
        dur_lower : float, optional
            Minimum spindle duration.
        dur_upper : float, optional
            Maximum spindle duration.
        baseline : bool, default=True
            Whether to baseline-correct the EEG data.
        verbose : bool, default=True
            Whether to print detection results.

        Returns
        -------
        event_summary : pd.DataFrame
            Summary of detected SOs and SPs.
        cp_event : pd.DataFrame or None
            Coupling event data if `coupling=True`, else None.
        so_waveform : pd.DataFrame or None
            SO waveform data if `coupling=True`, else None.
        """
    _, so_candidate, so_summary = psg.detect_slow_oscillations(target_stage=interest_stage, method=so_method,
                                                               baseline=baseline, verbose=verbose,
                                                               filter_freq=filter_freq,
                                                               duration=duration, filter_type=filter_type)
    _, sp_candidate, sp_summary = psg.detect_spindles(target_stage=interest_stage, method=sp_method,
                                                      l_freq=l_freq, h_freq=h_freq, dur_lower=dur_lower,
                                                      dur_upper=dur_upper, baseline=baseline, verbose=verbose)
    event_summary = pd.merge(so_summary, sp_summary, on=['stage'], how='outer')

    cp_event = None
    so_waveform = None

    if coupling:
        event_summary, so_waveform = psg.pac(verbose=verbose, file_name=file_name)
    return event_summary, cp_event, so_waveform


def get_sosp_for_folder(raw_folder: str, stage_folder: str, interest_stage=('N2', 'SWS'),
                        sp_method='Hahn2020',
                        so_method='Staresina', coupling=True, scoring_dur=30, rerefer=False, chan_limit=None,
                        ch_drop=(), skip_header=True, skip_footer=0,
                        montage_temp="standard_1005", is_montage=True,
                        filter_freq=None, duration=None, filter_type: str = 'fir', l_freq: float = None,
                        h_freq: float = None, dur_lower: float = 0.5, dur_upper: float = math.inf,
                        baseline: bool = True, verbose: bool = True, bad_epoch=True):
    """
        Detect SOs and SPs for all EEG recordings in a folder.

        Parameters
        ----------
        raw_folder : str
            Path to the folder containing EEG recordings.
        stage_folder : str
            Path to the folder containing hypnogram scoring files.
        interest_stage : tuple, default=('N2', 'SWS')
            Sleep stages to analyze.
        sp_method : str, default='Hahn2020'
            Method for spindle detection.
        so_method : str, default='Staresina'
            Method for slow oscillation detection.
        coupling : bool, default=True
            Whether to perform phase-amplitude coupling analysis.
        scoring_dur : int, default=30
            Duration of each hypnogram epoch.
        rerefer : bool, default=False
            Whether to re-reference EEG channels.
        chan_limit : list, optional
            List of EEG channels to retain.
        ch_drop : tuple, default=()
            List of channels to exclude.
        skip_header : bool, default=True
            Whether to skip the hypnogram file header.
        skip_footer : int, default=0
            Number of footer rows to skip.
        montage_temp : str, default="standard_1005"
            Montage template for EEG channel locations.
        is_montage : bool, default=True
            Whether to apply a montage.
        filter_freq : tuple, optional
            Frequency range for SO detection.
        duration : tuple, optional
            Duration range for SO detection.
        filter_type : str, default='fir'
            Filtering method ('fir' or 'iir').
        l_freq : float, optional
            Low-frequency cutoff for spindle detection.
        h_freq : float, optional
            High-frequency cutoff for spindle detection.
        dur_lower : float, default=0.5
            Minimum spindle duration.
        dur_upper : float, default=math.inf
            Maximum spindle duration.
        baseline : bool, default=True
            Whether to baseline-correct EEG data.
        verbose : bool, default=True
            Whether to print detection results.
        bad_epoch : bool, default=True
            Whether to mark stage 6 epochs as bad.

        Returns
        -------
        event_summary_all : dict
            Dictionary containing event summaries for all recordings.
        coupling_event_all : dict
            Dictionary containing coupling event data for all recordings.
        so_waveform_all : dict
            Dictionary containing SO waveform data for all recordings.
        """
    file_paths = [os.path.join(raw_folder, f) for f in os.listdir(raw_folder) if
                  f.endswith(('.vhdr', '.edf', '.fif', '.set', '.fdt', '.bdf', '.cnt'))]
    coupling_event_all = {}
    event_summary_all = {}
    so_waveform_all = {}
    remaining_participants = []

    for file_path in file_paths:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\033[1mStart event detection for subject {file_name}.\033[0m")
        try:
            psg = None

            if os.path.isfile(os.path.join(stage_folder, f"{file_name}.mat")):
                stage_path = os.path.join(stage_folder, f"{file_name}.mat")
                psg = PolySomnoGraphy(file_path, hypnogram_path=stage_path, hypnogram_type='Hume',
                                      skip_header=skip_header, interval=scoring_dur,
                                      bad_epoch=bad_epoch, rerefer=rerefer, chan_limit=chan_limit, drop_chan=ch_drop,
                                      montage_temp=montage_temp, is_montage=is_montage)
            elif os.path.isfile(os.path.join(stage_folder, f"{file_name}.txt")):
                stage_path = os.path.join(stage_folder, f"{file_name}.txt")
                psg = PolySomnoGraphy(file_path, hypnogram_path=stage_path, hypnogram_type='RemLogic',
                                      skip_header=skip_header, interval=scoring_dur,
                                      bad_epoch=bad_epoch, rerefer=rerefer, chan_limit=chan_limit, drop_chan=ch_drop,
                                      montage_temp=montage_temp, is_montage=is_montage)
            else:
                print(f"Neither {file_name}.mat nor {file_name}.txt exists in the folder.")

            event_summary, coupling_event, so_waveform = get_sosp(psg, file_name,
                                                                  interest_stage=interest_stage, sp_method=sp_method,
                                                                  so_method=so_method, coupling=coupling,
                                                                  filter_freq=filter_freq, duration=duration,
                                                                  filter_type=filter_type, l_freq=l_freq,
                                                                  h_freq=h_freq, dur_lower=dur_lower,
                                                                  dur_upper=dur_upper,
                                                                  baseline=baseline, verbose=verbose)

            event_summary_all[file_name] = event_summary
            coupling_event_all[file_name] = coupling_event
            so_waveform_all[file_name] = so_waveform
        except:
            print(f"Participant {file_name} was skipped due to errors processing files.")
            remaining_participants.append(file_name)
    if remaining_participants != []:
        print("The following participants couldn't be completed:")
        print(remaining_participants)
    return event_summary_all, coupling_event_all, so_waveform_all
