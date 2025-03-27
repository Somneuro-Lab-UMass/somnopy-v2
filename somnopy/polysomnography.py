import math
from typing import Optional, List, Tuple, Iterable, Any
import mne
import pandas as pd

from somnopy.event_detection import SP_detection, SO_detection, detect_swa
from somnopy.metrics import pac, event_lock
from somnopy.RemLogicDataLoader import RemLogicDataLoader  # Ensure this module is available.
from somnopy.HumeDataLoader import HumeDataLoader  # Ensure this module is available.


class PolySomnoGraphy:
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
    A class for processing EEG and hypnogram data for sleep spindle and slow oscillation analysis.

    This class provides methods for loading EEG data, processing hypnogram files,
    detecting slow oscillations (SOs), detecting spindles (SPs), computing phase-amplitude coupling (PAC),
    and handling various preprocessing operations.

    Parameters
    ----------
    eeg_path : str
        Path to the raw EEG file.
    hypnogram_path : Optional[str], default=None
        Path to the hypnogram scoring file.
    hypnogram_type : Optional[str], default=None
        Type of hypnogram file, either `"RemLogic"` or `"Hume"`.
    skip_header : bool, default=True
        Whether to skip header lines in hypnogram files.
    interval : int, default=30
        Duration (in seconds) of each hypnogram epoch.
    bad_epoch : bool, default=True
        Whether to mark stage 6 epochs as bad.
    set_up_raw : bool, default=True
        Whether to preprocess EEG data.
    rerefer : bool, default=False
        Whether to re-reference EEG channels.
    chan_limit : Optional[List[str]], default=None
        List of EEG channels to retain.
    montage_temp : str, default="standard_1005"
        Montage template for channel locations.
    is_montage : bool, default=False
        Whether to apply a montage.
    drop_chan : Iterable[str], default=()
        List of channels to exclude.

    Attributes
    ----------
    raw : Optional[mne.io.BaseRaw]
        The loaded EEG raw data object.
    hypno : Optional[pd.DataFrame]
        DataFrame containing the hypnogram stages.
    segments : Optional[List[Tuple[int, float, float]]]
        List of tuples containing (stage, segment length, valid duration).
    spindles : Optional[Any]
        Detected spindles.
    slow_oscillations : Optional[Any]
        Detected slow oscillations.

    Methods
    -------
    load_eeg()
        Loads EEG data from the specified file.
    load_hypnogram()
        Loads and processes the hypnogram file.
    segment_hypnogram()
        Segments the hypnogram into epochs matching EEG data.
    get_segments() -> List[Tuple[int, float, float]]
        Returns a list of hypnogram segments.
    get_raw() -> mne.io.BaseRaw
        Returns the loaded raw EEG object.
    get_hypnogram() -> pd.DataFrame
        Returns the processed hypnogram DataFrame.
    detect_spindles(...)
        Detects sleep spindles using specified parameters.
    detect_slow_oscillations(...)
        Detects slow oscillations using specified parameters.
    pac(...)
        Computes phase-amplitude coupling (PAC) between spindles and slow oscillations.
    detect_swa(...)
        Computes slow-wave activity (SWA) in specific sleep stages.
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
        """Load the raw EEG file based on its extension."""
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
        Load hypnogram data from the provided file using the specified hypnogram_type.
        hypnogram_type must be either "RemLogic" or "Hume".

        The loaded hypnogram data is stored internally as a pandas DataFrame with a 'stages' column.
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
        Segment hypnogram data into intervals corresponding to sleep stages.
        This method matches the hypnogram (self.hypno) to the raw EEG data (self.raw).

        It stores a list of tuples: (stage, segment length, valid duration) in self.segments.
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
        """Return the list of hypnogram segments."""
        if self.segments is None:
            raise ValueError("Segments have not been computed.")
        return self.segments

    def get_raw(self) -> mne.io.BaseRaw:
        """Return the MNE Raw EEG object."""
        if self.raw is None:
            raise ValueError("EEG data not loaded.")
        return self.raw

    def get_hypnogram(self) -> pd.DataFrame:
        """Return the hypnogram DataFrame with the 'stages' column."""
        if self.hypno is None:
            raise ValueError("Hypnogram data not loaded.")
        return self.hypno

    def __good_epoch_dur(self, seg_start: float, seg_end: float, seg_len: float) -> float:
        """
        Calculate the valid duration of an epoch, excluding bad segments.

        Returns
        -------
        float
            Valid duration of the epoch.
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
        if self.spindles is None or self.slow_oscillations is None:
            raise Warning("Attempting to run before detect_spindles or detect_slow_oscillations")
        event_summary = pd.merge(self.slow_oscillations[2], self.spindles[2], on=['stage', 'channel'], how='outer')
        cp_event, event_summary = event_lock(self.raw, self.slow_oscillations[1],
                                             self.spindles[1], event_summary, verbose=verbose)
        cp_event.insert(0, 'subject', file_name)
        self.pac = pac(self.raw, cp_event, event_summary, verbose=verbose)
        return self.pac

    def detect_swa(self, stages=None, file_name='id', l_freq=0.5, h_freq=4):
        self.swa = detect_swa(self.raw, stages=stages, file_name=file_name, l_freq=l_freq, h_freq=h_freq)
        return self.swa