import math
from typing import Optional, List, Tuple, Iterable, Any
import mne
import pandas as pd

from somnopy.event_detection import SP_detection, SO_detection, detect_swa
from somnopy.metrics import pac, event_lock
from somnopy.RemLogicDataLoader import RemLogicDataLoader  # Ensure this module is available.
from somnopy.HumeDataLoader import HumeDataLoader  # Ensure this module is available.


class PolySomnoGraphy:
    """
    Processing pipeline for EEG and hypnogram data to detect sleep events and compute metrics.

    This class provides methods to load EEG data, process hypnogram scoring files, segment
    sleep stages, detect slow oscillations (SO) and sleep spindles (SP), compute phase-amplitude
    coupling (PAC), and compute slow-wave activity (SWA).

    Attributes
    ----------
    raw : Optional[mne.io.BaseRaw]
        The loaded and optionally preprocessed raw EEG data.
    hypno : Optional[pd.DataFrame]
        Hypnogram DataFrame containing a 'stages' column of stage labels.
    segments : Optional[List[Tuple[int, float, float]]]
        List of tuples (stage_label, segment_length_s, valid_duration_s) per sleep epoch.
    spindles : Any
        Output from SP_detection: (raw_copy, sp_event_df, sp_summary_df).
    slow_oscillations : Any
        Output from SO_detection: (raw_copy, so_event_df, so_summary_df).
    swa : pd.DataFrame
        Slow-wave activity per channel computed by detect_swa.
    pac : Any
        Phase-amplitude coupling results after running pac().
    """
    def __init__(self,
                 eeg_path: str,
                 hypnogram_path: Optional[str] = None,
                 hypnogram_type: Optional[str] = None,
                 skip_header: bool = True,
                 interval: int = 30,
                 bad_epoch: bool = True,
                 set_up_raw: bool = True,
                 rerefer: bool = False,
                 chan_limit=None,
                 montage_temp: str = "standard_1005",
                 is_montage: bool = False,
                 drop_chan=()
                 ) -> None:
        """
        Initialize processing of EEG and hypnogram data.

        Parameters
        ----------
        eeg_path : str
            Path to the raw EEG file (supported extensions: .vhdr, .edf, .fif, .set, .bdf, .cnt).
        hypnogram_path : str or None
            Path to the hypnogram scoring file, if available.
        hypnogram_type : str or None
            Type of hypnogram file: 'RemLogic' or 'Hume'.
        skip_header : bool
            Whether to skip header lines in RemLogic files.
        interval : int
            Duration (s) of each hypnogram epoch.
        bad_epoch : bool
            Mark stage label 6 epochs as bad by annotation.
        set_up_raw : bool
            If True, apply channel dropping, re-referencing, and montage.
        rerefer : bool
            Re-reference EEG to mastoids (M1, M2) if True.
        chan_limit : list or None
            List of channel names to keep.
        montage_temp : str
            Standard montage template name.
        is_montage : bool
            If True, apply standard montage.
        drop_chan : iterable
            List of channels to drop in addition to physiological channels.
        """
        self.slow_oscillations = None
        self.spindles = None
        self.eeg_path = eeg_path
        self.hypnogram_path = hypnogram_path
        self.hypnogram_type = hypnogram_type
        self.skip_header = skip_header
        self.interval = interval
        self.bad_epoch_flag = bad_epoch

        self.raw: Optional[mne.io.BaseRaw] = None  # MNE Raw object.
        self.hypno: Optional[pd.DataFrame] = None  # Hypnogram DataFrame with column 'stages'.
        self.segments: Optional[List[Tuple[int, float, float]]] = None  # List of segments: (stage, seg_len, valid_dur).

        self.load_eeg()
        if set_up_raw:
            self.__set_up_raw(rerefer=rerefer, chan_limit=chan_limit,
                              montage_temp=montage_temp, is_montage=is_montage, drop_chan=drop_chan)
        if self.hypnogram_path:
            self.load_hypnogram()

    def load_eeg(self) -> None:
        """
        Load raw EEG data based on file extension.

        Supported formats:
        - BrainVision (.vhdr)
        - EDF (.edf)
        - FIF (.fif)
        - EEGLAB (.set)
        - BDF (.bdf)
        - CNT (.cnt)

        Raises
        ------
        ValueError
            If the file extension is unsupported or loading fails.
        """
        if self.eeg_path.endswith('.vhdr'):
            self.raw = mne.io.read_raw_brainvision(self.eeg_path, preload=True, verbose='ERROR')
        elif self.eeg_path.endswith('.edf'):
            self.raw = mne.io.read_raw_edf(self.eeg_path, preload=True, verbose='ERROR')
        elif self.eeg_path.endswith('.fif'):
            self.raw = mne.io.read_raw_fif(self.eeg_path, preload=True, verbose='ERROR')
        elif self.eeg_path.endswith('.set'):
            self.raw = mne.io.read_raw_eeglab(self.eeg_path, preload=True, verbose='ERROR')
        elif self.eeg_path.endswith('.bdf'):
            self.raw = mne.io.read_raw_bdf(self.eeg_path, preload=True, verbose='ERROR')
        elif self.eeg_path.endswith('.cnt'):
            self.raw = mne.io.read_raw_cnt(self.eeg_path, preload=True, verbose='ERROR')
        else:
            raise ValueError(f"Unsupported file format: {self.eeg_path}")

    def load_hypnogram(self) -> None:
        """
        Load and process hypnogram scoring file into a DataFrame.

        Uses RemLogicDataLoader or HumeDataLoader based on hypnogram_type.
        After loading, invokes segment_hypnogram() to generate segments.

        Raises
        ------
        ValueError
            If hypnogram_type is invalid.
        """

        if self.hypnogram_type.lower() == "remlogic":
            loader = RemLogicDataLoader(self.hypnogram_path, skip_header=self.skip_header)
            # Assume get_data() returns a DataFrame with a 'stages' column.
            self.hypno = loader.get_data()
        elif self.hypnogram_type.lower() == "hume":
            loader = HumeDataLoader(self.hypnogram_path)
            # get_data() here returns a NumPy array, so wrap it in a DataFrame.
            stages_array = loader.get_data()
            self.hypno = pd.DataFrame(stages_array, columns=['stages'])
        else:
            # This branch should not occur.
            raise ValueError("Invalid hypnogram_type provided.")
        self.segment_hypnogram()

    def segment_hypnogram(self) -> None:
        """
        Convert hypnogram stages into contiguous stage segments.

        Generates self.segments as a list of
        (stage_label, segment_length_s, valid_duration_s).
        Marks bad epochs as annotations if bad_epoch_flag is True.

        Raises
        ------
        ValueError
            If hypnogram data has not been loaded.
        """
        if self.hypno is None:
            raise ValueError("Hypnogram data not loaded.")

        # Extract stage values from the DataFrame.
        hypno = self.hypno['stages'].values.flatten()
        stage_segments: List[Tuple[int, float, float]] = []
        cur_stage: int = int(hypno[0])
        cnt: int = 0
        seg_start: int = 0

        # raw_duration = self.raw.times[-1]
        # score_duration = hypno.shape[0]*self.interval
        # if abs(raw_duration-score_duration) > 2*self.interval:
        #     raise Warning('Duration mismatch between eeg and scoring files')

        for i, stage in enumerate(hypno):
            if int(stage) == cur_stage:
                cnt += 1
            else:
                seg_len: int = cnt * self.interval
                seg_end: int = seg_start + seg_len
                valid_dur: float = self.__good_epoch_dur(seg_start, seg_end, seg_len)
                stage_segments.append((cur_stage, max(0, seg_len), valid_dur))
                cur_stage = int(stage)
                cnt = 1
                seg_start = seg_end

            if self.bad_epoch_flag and int(stage) == 6:
                onset: float = i * self.interval
                bad_annotation = mne.Annotations(onset=[onset],
                                                 duration=[self.interval],
                                                 description=['Bad_epoch'],
                                                 orig_time=None)
                self.raw.set_annotations(bad_annotation)

        seg_len = cnt * self.interval
        seg_end = min(seg_start + seg_len, self.raw.times[-1])
        seg_len = seg_end - seg_start
        valid_dur = self.__good_epoch_dur(seg_start, seg_end, seg_len)
        stage_segments.append((cur_stage, max(0, seg_len), valid_dur))

        self.segments = stage_segments

    def get_segments(self) -> List[Tuple[int, float, float]]:
        """
        Return the list of hypnogram segments.

        Returns
        -------
        List[Tuple[int, float, float]]
            Each tuple is (stage_label, segment_length_s, valid_duration_s).
        """
        if self.segments is None:
            raise ValueError("Segments have not been computed.")
        return self.segments

    def get_raw(self) -> mne.io.BaseRaw:
        """
        Return the loaded raw EEG object.

        Returns
        -------
        mne.io.BaseRaw
            The MNE Raw object containing EEG data.

        Raises
        ------
        ValueError
            If EEG data has not been loaded.
        """
        if self.raw is None:
            raise ValueError("EEG data not loaded.")
        return self.raw

    def get_hypnogram(self) -> pd.DataFrame:
        """
        Return the processed hypnogram DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with a 'stages' column of integer stage labels.

        Raises
        ------
        ValueError
            If hypnogram data has not been loaded.
        """
        if self.hypno is None:
            raise ValueError("Hypnogram data not loaded.")
        return self.hypno

    def __good_epoch_dur(self, seg_start: float, seg_end: float, seg_len: float) -> float:
        """
        Compute the valid duration of an epoch segment, excluding bad annotations.

        Parameters
        ----------
        seg_start : float
            Start time (s) of the segment.
        seg_end : float
            End time (s) of the segment.
        seg_len : float
            Nominal length (s) of the segment.

        Returns
        -------
        float
            Duration (s) excluding any bad-epoch intervals.
        """

        bad_dur = sum(max(0.0, min(anno['onset'] + anno['duration'], seg_end) - max(anno['onset'], seg_start))
                      for anno in self.raw.annotations if 'Bad_epoch' in anno['description'])

        return max(0.0, seg_len - bad_dur)

    def __set_up_raw(self,
                     rerefer: bool = False,
                     chan_limit=None,
                     montage_temp: str = "standard_1005",
                     is_montage: bool = False,
                     drop_chan=()):
        """
        Preprocess raw EEG by dropping non-EEG channels, re-referencing, and applying montage.

        Parameters
        ----------
        rerefer : bool
            If True, set EEG reference to mastoid channels ['M1','M2'].
        chan_limit : list or None
            List of channel names to retain; others are dropped.
        montage_temp : str
            Name of the standard montage to apply.
        is_montage : bool
            If True, apply a standard montage to channel locations.
        drop_chan : iterable
            Additional channel names to drop.
        """
        ch_drop = [
            ch for ch in self.raw.ch_names
            if ch.startswith('M') or 'EMG' in ch or 'EOG' in ch or 'ECG' in ch or 'chin' in ch.lower() or ch.startswith(
                'E')
        ]
        self.raw.drop_channels(ch_drop)
        self.raw.drop_channels(drop_chan, on_missing='warn')
        if chan_limit is not None:
            chan_limit = [
                ch for ch in self.raw.ch_names
                if ch in chan_limit
            ]
            self.raw = self.raw.pick_channels(chan_limit, ordered=False)
        if rerefer:
            self.raw.set_eeg_reference(ref_channels=['M1', 'M2'])
        if is_montage:
            montage = mne.channels.make_standard_montage(montage_temp)
            self.raw.set_montage(montage, on_missing='warn')

    def detect_spindles(self,
                        target_stage: Iterable = ('N2', 'SWS'),
                        method: str = "Hahn2020",
                        l_freq: float = 10,
                        h_freq: float = 16,
                        dur_lower: float = 0.5,
                        dur_upper: float = math.inf,
                        baseline: bool = True,
                        verbose: bool = True):
        
        """
        Detect sleep spindles in specified sleep stages.

        Wraps SP_detection from somnopy.event_detection.

        Parameters
        ----------
        target_stage : iterable
            Sleep stages to analyze (e.g. ['N2','SWS']).
        method : str
            Spindle detection method name. Options:
            'Hahn2020','Martin2013','Wamsley2012','Wendt2012','Ferrarelli2007'.
        l_freq, h_freq : float
            Bandpass filter limits (Hz).
        dur_lower, dur_upper : float
            Minimum and maximum spindle duration (s).
        baseline : bool
            If True, subtract mean from each channel prior to detection.
        verbose : bool
            If True, print detection summary.

        Returns
        -------
        tuple
            (raw_copy, sp_event_df, sp_summary_df)
        """

        self.spindles = SP_detection(self.raw, self.segments,
                                     target_stage=target_stage,
                                     method=method,
                                     l_freq=l_freq,
                                     h_freq=h_freq,
                                     dur_lower=dur_lower,
                                     dur_upper=dur_upper,
                                     baseline=baseline,
                                     verbose=verbose)
        return self.spindles

    def detect_slow_oscillations(self,
                                 target_stage: Iterable = ('N2', 'SWS'),
                                 filter_freq: Any = None,
                                 duration: Any = None,
                                 baseline: bool = True,
                                 filter_type: str = 'fir',
                                 method: str = 'Staresina',
                                 verbose: bool = True):
        
        """
        Detect slow oscillations (SOs) in specified sleep stages.

        Wraps SO_detection from somnopy.event_detection.

        Parameters
        ----------
        target_stage : iterable
            Sleep stages to analyze (e.g. ['N2','SWS']).
        filter_freq : tuple or None
            Bandpass filter bounds (l_freq, h_freq) or None for default.
        duration : tuple or None
            Event duration bounds (dur_lower, dur_upper) or None for default.
        baseline : bool
            If True, subtract mean before detection.
        filter_type : str
            'fir' or 'iir'.
        method : str
            SO detection method name.
        verbose : bool
            If True, print detection summary.

        Returns
        -------
        tuple
            (raw_copy, so_event_df, so_summary_df)
        """

        self.slow_oscillations = SO_detection(self.raw, self.segments,
                                              target_stage=target_stage,
                                              method=method,
                                              filter_type=filter_type,
                                              filter_freq=filter_freq,
                                              duration=duration,
                                              baseline=baseline,
                                              verbose=verbose)
        return self.slow_oscillations

    def pac(self, verbose: bool = True, file_name: str = "Participant"):
        """
        Compute event-locked PAC between SO troughs and spindles.

        Merges SO and SP summaries, finds coupled events via event_lock,
        then computes PAC metrics and optionally plots PETH+PAC.

        Parameters
        ----------
        verbose : bool
            If True, print per-stage PAC stats and show plots.
        file_name : str
            Identifier to prepend to subject column in results.

        Returns
        -------
        pandas.DataFrame
            Merged event summary including PAC metrics and coupling density.
        """

        if self.spindles is None or self.slow_oscillations is None:
            raise Warning("Attempting to run before detect_spindles or detect_slow_oscillations")
        event_summary = pd.merge(self.slow_oscillations[2], self.spindles[2], on=['stage', 'channel'], how='outer')
        cp_event, event_summary = event_lock(self.raw, self.slow_oscillations[1],
                                             self.spindles[1], event_summary, verbose=verbose)
        cp_event.insert(0, 'subject', file_name)
        self.pac = pac(self.raw, cp_event, event_summary, verbose=verbose)
        return self.pac

    def detect_swa(self, stages=None, file_name='id', l_freq=0.5, h_freq=4):
        """
        Compute slow-wave activity (SWA) per channel.

        Wraps detect_swa from somnopy.event_detection.

        Parameters
        ----------
        stages : list or None
            Sleep stages to include in SWA computation.
        file_name : str
            Participant identifier for output DataFrame.
        l_freq, h_freq : float
            Delta band limits (Hz).

        Returns
        -------
        pandas.DataFrame
            DataFrame with SWA value per channel.
        """
        self.swa = detect_swa(self.raw, stages=stages, psg=self.segments, file_name=file_name, l_freq=l_freq,
                              h_freq=h_freq)
        return self.swa