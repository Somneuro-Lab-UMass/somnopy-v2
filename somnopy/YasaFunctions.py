"""
YasaFunctions.py

YASA/MNE sleep analysis, spectrogram plotting,
core body temperature overlays, automatic sleep staging, 
and CBT-bandpower correlations.

Notebook usage:
    from yasa_functions import YasaFunctions

    analyzer = YasaFunctions(stage_mapping=stage_mapping)
    input_data, channels, hypno, sf, hyp_window, raw = analyzer.load_yasa_from_edf(
        edf_path="subject.edf",
        hyp_path="subject.mat",
    )
"""

from __future__ import annotations
from scipy.stats import linregress

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yasa
from lspopt import spectrogram_lspopt
from scipy.io import loadmat
from scipy.stats import spearmanr
from datetime import datetime, timedelta, time


class YasaFunctions:
    """
    Class for loading EDF/hypnogram data, plotting YASA spectrograms, running
    automatic sleep staging, overlaying core body temperature, and correlating
    CBT with EEG bandpower.

    Parameters
    ----------
    stage_mapping : dict, optional
        Mapping from source hypnogram labels/codes to YASA-compatible integer
        stage codes. You can also pass this directly to methods that need it.
    """

    def __init__(self, stage_mapping: Optional[Dict[Any, int]] = None):
        if stage_mapping is None:
            stage_mapping = {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                5: 4,
                6: -1,
                7: -2,
            }

        self.recording_start = None
        self.hyp_start = None
        self.stage_mapping = stage_mapping
        self.input_data = None
        self.channels = None
        self.hypno = None
        self.sampling_rate = None
        self.hyp_window = None
        self.raw = None
        self.stages_per_quad = None
        self.avg_temp_per_quad = None
        self.cbt=None


        self.YASA_STAGE_NAMES = {
            -2: "Unscored",
            -1: "Artifact",
            0: "Wake",
            1: "N1",
            2: "N2",
            3: "N3",
            4: "REM",
        }

    def _resolve_stage_mapping(self, stage_mapping: Optional[Dict[Any, int]] = None):
        if stage_mapping is not None:
            self.stage_mapping = stage_mapping

        return self.stage_mapping

    def _check_loaded(self):
        if self.input_data is None or self.hypno is None or self.sampling_rate is None or self.hyp_window is None:
            raise ValueError(
                "No EDF/hypnogram data loaded. Run load_yasa_from_edf first."
            )

    def load_yasa_from_edf(
        self,
        edf_path: str | Path,
        hyp_path: str | Path,
        stage_mapping: Optional[Dict[Any, int]] = None,
        load_from_sleep_onset: Optional[bool] = True 
    ):
        """
        Load raw EEG data from an EDF file and load a hypnogram from either
        a .mat or .txt file, converting sleep stages to YASA-compatible codes.

        Returns
        -------
        input_data : dict
            Dictionary mapping channel name -> EEG signal array.
        channels : list
            Channel names from the EDF file.
        hypno : np.ndarray
            Hypnogram upsampled to match EEG data length.
        sampling_rate : float
            EEG sampling frequency.
        hyp_window : float
            Hypnogram window length in seconds.
        raw : mne.io.Raw
            Loaded MNE Raw object.
        """
        stage_mapping = self._resolve_stage_mapping(stage_mapping)
        edf_path = Path(edf_path)
        hyp_path = Path(hyp_path)

        #load raw data
        raw = mne.io.read_raw_edf(str(edf_path), preload=True)
        raw_data, _ = raw.get_data(return_times=True)
        channels = raw.ch_names[: len(raw_data)]
        sampling_rate = raw.info["sfreq"]

        #load hypnogram
        if hyp_path.suffix.lower() == ".mat":
            mat = loadmat(str(hyp_path))

            #store hypnogram data, window size, and sampling rate
            hyp_data = mat["stageData"]["stages"][0, 0].squeeze().astype(int)
            hyp_window = int(mat["stageData"]["win"][0][0][0][0])
            hyp_srate = int(mat["stageData"]["srate"][0][0][0][0])
            hyp_start = self._matlab_datenum_to_datetime(mat["stageData"]["recStart"][0][0][0][0])


            #convert the integer codes present in hyp_data into YASA approved integer codes
            hyp_yasa = np.array([stage_mapping[x] for x in hyp_data], dtype=int)

        elif hyp_path.suffix.lower() == ".txt":

            #start parsing the txt hypnogram line by line
            with open(hyp_path, "r", encoding="utf-8", errors="ignore") as file:
                lines = file.readlines()

            header_idx = None
            for i, line in enumerate(lines):
                if line.startswith("Sleep Stage"):
                    header_idx = i
                    break

            if header_idx is None:
                raise ValueError("Could not find table header in file.")

            #store hypnogram as a pandas object
            df = pd.read_csv(
                hyp_path,
                sep="\t",
                skiprows=header_idx,
                engine="python",
            )

            #remove spaces from column names
            df.columns = df.columns.str.strip()

            #keeping ONLY stages that are listed in the key, otherwise yasa cant parse it.
            df["Event"] = df["Event"].astype(str).str.strip().str.upper()
            df_stage = df[df["Event"].isin(stage_mapping.keys())].copy()

            if len(df_stage) == 0:
                raise ValueError("No valid sleep stage rows found.")

            hyp_yasa = df_stage["Event"].map(stage_mapping).values.astype(int)
            hyp_window = float(df_stage["Duration[s]"].astype(float).iloc[0])
            hyp_srate = 1.0 / hyp_window

        else:
            raise ValueError("hyp_path must end with .mat or .txt")

        input_data = {
            channel: data
            for channel, data in zip(channels, raw_data)
        }
        recording_start = pd.to_datetime(raw.info["meas_date"]).tz_localize(None)
        self.recording_start = recording_start

        hypno = yasa.hypno_upsample_to_data(
            hypno=hyp_yasa,
            sf_hypno=1 / hyp_window,
            data=raw_data[0],
            sf_data=sampling_rate,
        )

        '''
        hypno = self._shift_hypno_by_time(
            hypno=hypno,
            hyp_start=hyp_start,
            sampling_rate=sampling_rate,
        )
        '''
        



        #store class attributes
        self.hyp_start = hyp_start
        self.input_data = input_data
        self.channels = channels
        self.hypno = hypno
        self.sampling_rate = sampling_rate
        self.hyp_window = hyp_window
        self.raw = raw

        return input_data, channels, hypno, recording_start, sampling_rate, hyp_window, raw
    

    def _shift_hypno_by_time(
            self,
            hypno: np.ndarray,
            hyp_start: pd.Timestamp,
            sampling_rate: float,
            verbose: bool = True,
        ) -> np.ndarray:
        """
        Shift an already-upsampled hypnogram so its start time aligns with EDF start.

        Positive offset:
            hypnogram starts after EDF, so pad beginning with -2 and crop end.

        Negative offset:
            hypnogram starts before EDF, so crop beginning and pad end with -2.
        """
        if self.recording_start is None:
            raise ValueError("recording_start is missing.")

        hyp_start = self._strip_tz(hyp_start)

        offset_sec = (hyp_start - self.recording_start).total_seconds()
        offset_samples = int(round(offset_sec * sampling_rate))

        if verbose:
            print("EDF start:", self.recording_start)
            print("Hypnogram start:", hyp_start)
            print("Offset seconds:", offset_sec)
            print("Offset samples:", offset_samples)

        n_samples = len(hypno)

        if offset_samples > 0:
            # Hypnogram starts after EDF.
            # Add unscored samples at beginning, then crop back to original length.
            shifted = np.pad(
                hypno,
                (offset_samples, 0),
                mode="constant",
                constant_values=np.int16(-2),
            )
            shifted = shifted[:n_samples]

        elif offset_samples < 0:
            # Hypnogram starts before EDF.
            # Remove early hypnogram samples, then pad end.
            crop_start = abs(offset_samples)

            if crop_start >= n_samples:
                shifted = np.full(n_samples, -2, dtype=int)
            else:
                shifted = hypno[crop_start:]
                shifted = np.pad(
                    shifted,
                    (0, n_samples - len(shifted)),
                    mode="constant",
                    constant_values=np.int16(-2),
                )

        else:
            shifted = hypno

        if verbose:
            print("Final hypno samples:", len(shifted))
            print("Unique values:", np.unique(shifted, return_counts=True))

        return shifted.astype(int)


    def plot_single_electrode(
        self,
        electrode: str,
        verbose: bool = False,
        outpath: Optional[str | Path] = None,
        filename: str = "participant",
        fmin: float = 0.5,
        fmax: float = 25,
    ):
        """Plot and optionally save a YASA spectrogram for one electrode."""
        self._check_loaded()

        if electrode not in self.input_data:
            raise ValueError(
                f"Electrode '{electrode}' not found. Available channels: {list(self.input_data.keys())}"
            )

        fig = yasa.plot_spectrogram(
            self.input_data[electrode],
            hypno=self.hypno,
            sf=self.sampling_rate,
            win_sec= self.hyp_window,
            fmin=fmin,
            fmax=fmax,
        )

        if verbose:
            plt.show()

        if outpath is not None:
            outpath = Path(outpath)
            outpath.mkdir(parents=True, exist_ok=True)
            fig.savefig(outpath / f"{filename}_{electrode}_spectrogram.png")

        return fig

    def plot_all_electrodes(
        self,
        verbose: bool = False,
        outpath: Optional[str | Path] = None,
        filename: str = "participant",
        fmin: float = 0.5,
        fmax: float = 25,
    ):
        """Plot and optionally save YASA spectrograms for all electrodes."""
        self._check_loaded()
        figures = {}

        for electrode in self.input_data:
            figures[electrode] = self.plot_single_electrode(
                electrode=electrode,
                verbose=verbose,
                outpath=outpath,
                filename=filename,
                fmin=fmin,
                fmax=fmax,
            )
            print(f"Complete: {electrode}")

        return figures

    def save_spectrogram_csv(
        self,
        electrode: str,
        win_sec: Optional[float] = None,
        outpath: Optional[str | Path] = None,
        filename: str = "participant",
        fmin: float = 0.5,
        fmax: float = 25,
    ) -> pd.DataFrame:
        """
        Compute spectrogram power using the same general approach as YASA and
        optionally save it as a CSV.

        The returned dataframe has frequencies as rows, time points as columns,
        and power in dB/Hz as values.
        """
        self._check_loaded()

        if electrode not in self.input_data:
            raise ValueError(
                f"Electrode '{electrode}' not found. Available channels: {list(self.input_data.keys())}"
            )

        if win_sec is None:
            win_sec = self.hyp_window

        nperseg = int(win_sec * self.sampling_rate)
        frequencies, times, powers = spectrogram_lspopt(
            self.input_data[electrode],
            self.sampling_rate,
            nperseg=nperseg,
            noverlap=0,
        )

        power_db = 10 * np.log10(powers)

        keep = (frequencies >= fmin) & (frequencies <= fmax)
        frequencies = frequencies[keep]
        power_db = power_db[keep, :]

        print(f"frequency shape: {frequencies.shape}")
        print(f"times shape: {times.shape}")
        print(f"power decibel shape: {power_db.shape}")

        df = pd.DataFrame(
            power_db,
            index=frequencies,
            columns=times,
        )
        df.index.name = "frequency_hz"
        df.columns.name = "time_sec"

        if outpath is not None:
            outpath = Path(outpath)
            outpath.mkdir(parents=True, exist_ok=True)
            df.to_csv(outpath / f"{filename}_{electrode}_power_data.csv")

        return df

    def automatic_sleep_staging(
        self,
        edf_path: str | Path,
        eeg: str = "C4",
        eog: Optional[str] = None,
        emg: Optional[str] = None,
        outpath: Optional[str | Path] = None,
        filename: str = "participant",
    ):
        """Run YASA automatic sleep staging on one EDF file."""
        raw = mne.io.read_raw_edf(str(edf_path), preload=True)

        sls = yasa.SleepStaging(
            raw,
            eeg_name=eeg,
            eog_name=eog,
            emg_name=emg,
        )

        hyp = sls.predict()
        probs = hyp.proba
        confidence = hyp.proba.max(axis=1)

        df = hyp.hypno.to_frame(name="stage").reset_index()
        df["time_sec"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds()

        if outpath is not None:
            outpath = Path(outpath)
            outpath.mkdir(parents=True, exist_ok=True)

            df.to_csv(outpath / f"{filename}_sleep_stages.csv", index=False)

            ax = sls.plot_predict_proba()
            fig = ax.get_figure()
            fig.savefig(outpath / f"{filename}_sleep_stage_probabilities.png")

        return df, hyp, probs, confidence

    def run_sleep_staging_on_folder(
        self,
        input_folder: str | Path,
        out_folder: str | Path,
    ):
        """Run YASA automatic sleep staging on every EDF file in a folder."""
        input_folder = Path(input_folder)
        out_folder = Path(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)

        failed_files = []
        edf_files = list(input_folder.glob("*.edf"))

        print(f"Found {len(edf_files)} EDF files.\n")

        for edf_file in edf_files:
            print(f"Processing: {edf_file.name}")

            try:
                file_out = out_folder / edf_file.stem
                file_out.mkdir(exist_ok=True)

                self.automatic_sleep_staging(
                    str(edf_file),
                    outpath=str(file_out),
                    filename=edf_file.stem,
                )

                print(f"✅ Success: {edf_file.name}\n")

            except Exception as exc:
                print(f"❌ Failed: {edf_file.name}")
                print(f"   Error: {exc}\n")
                failed_files.append((edf_file.name, str(exc)))
                continue

        print("\n===== SUMMARY =====")
        print(f"Total files: {len(edf_files)}")
        print(f"Failed: {len(failed_files)}")

        if failed_files:
            print("\nFailed files:")
            for name, err in failed_files:
                print(f"- {name}: {err}")

        return failed_files

    def plot_single_electrode_with_temp(
        self,
        electrode: str,
        temp_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        outpath: Optional[str | Path] = None,
        filename: str = "participant",
        fmin: float = 0.5,
        fmax: float = 25,
        temp_color: str = "black",
        temp_linewidth: float = 2,
        temp_alpha: float = 0.9,
    ):
        """Plot a YASA spectrogram for one electrode with CBT overlaid."""
        self._check_loaded()

        if electrode not in self.input_data:
            raise ValueError(
                f"Electrode '{electrode}' not found. Available channels: {list(self.input_data.keys())}"
            )

        if temp_df is None:
            if self.cbt is None:
                raise ValueError("No temp_df was passed and self.cbt is missing.")
            temp_plot_df = self.cbt.copy()
        else:
            temp_plot_df = temp_df.copy()

        if self.recording_start is None:
            raise ValueError("recording_start is missing. Run load_yasa_from_edf first.")

        temp_plot_df["hours_from_start"] = (
            temp_plot_df["datetime"] - self.recording_start
        ).dt.total_seconds() / 3600.0


        fig = yasa.plot_spectrogram(
            self.input_data[electrode],
            hypno=self.hypno,
            sf=self.sampling_rate,
            win_sec= self.hyp_window,
            fmin=fmin,
            fmax=fmax,
        )

        if isinstance(fig, plt.Figure):
            ax_spec = fig.axes[1]
        else:
            ax_spec = fig
            fig = ax_spec.figure

        ax_temp = ax_spec.twinx()
        ax_temp.plot(
            temp_plot_df["hours_from_start"],
            temp_plot_df["temp"],
            color=temp_color,
            linewidth=temp_linewidth,
            alpha=temp_alpha,
        )

        temp_min = temp_plot_df["temp"].min()
        ax_temp.set_ylim(temp_min - 0.1, temp_min + 1.0)

        ax_temp.set_ylabel("Core Body Temp")
        ax_temp.grid(False)

        handles1, labels1 = ax_spec.get_legend_handles_labels()
        handles2, labels2 = ax_temp.get_legend_handles_labels()

        if handles2:
            ax_temp.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

        if verbose:
            plt.show()

        if outpath is not None:
            outpath = Path(outpath)
            outpath.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                outpath / f"{filename}_{electrode}_spectrogram_temp.png",
                bbox_inches="tight",
                dpi=300,
            )

        return fig, ax_temp, temp_plot_df

    def load_cbt(self, filepath: str | Path) -> pd.DataFrame:
        """Load a core body temperature Excel file and standardize column names."""
        df = pd.read_excel(filepath).rename(
            {
                "Date(mm/dd/yyyy)": "date",
                "Hour": "hour",
                "Temperature": "temp",
            },
            axis=1,
        )

        prepared_df = self._prepare_temp_df(df)

        self.cbt = prepared_df
        return prepared_df
    

    @staticmethod
    def _matlab_datenum_to_datetime(matlab_datenum) -> pd.Timestamp:
        """Convert MATLAB datenum to pandas Timestamp."""
        matlab_datenum = float(matlab_datenum)

        py_datetime = (
            datetime.fromordinal(int(matlab_datenum))
            + timedelta(days=matlab_datenum % 1)
            - timedelta(days=366)
        )

        return pd.Timestamp(py_datetime)

    @staticmethod
    def _strip_tz(ts):
        """Convert a timezone-aware timestamp to timezone-naive."""
        ts = pd.to_datetime(ts)
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_localize(None)
        return ts

    @staticmethod
    def _prepare_temp_df(temp_df: pd.DataFrame) -> pd.DataFrame:
        """Parse and clean a CBT dataframe with date, hour, and temp columns."""
        temp_plot_df = temp_df.copy()

        required_cols = {"date", "hour", "temp"}
        missing = required_cols - set(temp_plot_df.columns)

        if missing:
            raise ValueError(f"temp_df is missing required columns: {missing}")

        temp_plot_df["datetime"] = pd.to_datetime(
            temp_plot_df["date"].astype(str) + " " + temp_plot_df["hour"].astype(str),
            errors="coerce",
        )
        temp_plot_df["temp"] = pd.to_numeric(temp_plot_df["temp"], errors="coerce")
        temp_plot_df = temp_plot_df.dropna(subset=["datetime", "temp"]).sort_values("datetime").copy()

        if temp_plot_df.empty:
            raise ValueError("No valid temperature rows remain after parsing.")

        if hasattr(temp_plot_df["datetime"].dt, "tz"):
            try:
                temp_plot_df["datetime"] = temp_plot_df["datetime"].dt.tz_localize(None)
            except TypeError:
                pass

        return temp_plot_df

    @staticmethod
    def _get_common_file_triplets(
        mat_folder: str | Path,
        edf_folder: str | Path,
        cbt_folder: str | Path,
    ):
        """Find files with matching stems across MAT, EDF, and XLSX folders."""
        mat_folder = Path(mat_folder)
        edf_folder = Path(edf_folder)
        cbt_folder = Path(cbt_folder)

        mat_files = {Path(path).stem: path for path in mat_folder.glob("*.mat")}
        edf_files = {Path(path).stem: path for path in edf_folder.glob("*.edf")}
        xlsx_files = {Path(path).stem: path for path in cbt_folder.glob("*.xlsx")}

        common = sorted(set(mat_files) & set(edf_files) & set(xlsx_files))

        triplets = [
            {
                "stem": stem,
                "mat": mat_files[stem],
                "edf": edf_files[stem],
                "xlsx": xlsx_files[stem],
            }
            for stem in common
        ]

        return triplets

    def batch_plot_spectrograms_with_temp(
        self,
        mat_folder: str | Path,
        edf_folder: str | Path,
        cbt_folder: str | Path,
        out_folder: str | Path,
        electrode: str,
        stage_mapping: Optional[Dict[Any, int]] = None,
        fmin: float = 0.5,
        fmax: float = 25,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Generate one spectrogram+CBT overlay per matched subject."""
        stage_mapping = self._resolve_stage_mapping(stage_mapping)
        out_folder = Path(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)

        triplets = self._get_common_file_triplets(mat_folder, edf_folder, cbt_folder)
        results = []

        print(f"Found {len(triplets)} matched .mat/.edf/.xlsx filename sets.")

        for item in triplets:
            stem = item["stem"]
            print(f"\nProcessing plot for: {stem}")

            try:
                self.load_yasa_from_edf(
                    edf_path=item["edf"],
                    hyp_path=item["mat"],
                    stage_mapping=stage_mapping,
                )

                if electrode not in self.input_data:
                    raise ValueError(
                        f"Electrode '{electrode}' not found. Available channels: {list(self.input_data.keys())}"
                    )

                temp_df = self.load_cbt(item["xlsx"])

                file_out = out_folder / stem
                file_out.mkdir(parents=True, exist_ok=True)

                fig, ax_temp, temp_plot_df = self.plot_single_electrode_with_temp(
                    electrode=electrode,
                    temp_df=temp_df,
                    verbose=verbose,
                    outpath=file_out,
                    filename=stem,
                    fmin=fmin,
                    fmax=fmax
                )

                plt.close(fig)

                results.append(
                    {
                        "stem": stem,
                        "edf_file": str(item["edf"]),
                        "mat_file": str(item["mat"]),
                        "xlsx_file": str(item["xlsx"]),
                        "electrode": electrode,
                        "status": "success",
                        "error": None,
                    }
                )
                print(f"✅ Saved plot for {stem}")

            except Exception as exc:
                results.append(
                    {
                        "stem": stem,
                        "edf_file": str(item["edf"]),
                        "mat_file": str(item["mat"]),
                        "xlsx_file": str(item["xlsx"]),
                        "electrode": electrode,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                print(f"❌ Failed for {stem}: {exc}")

        summary_df = pd.DataFrame(results)
        summary_df.to_csv(out_folder / "batch_plot_summary.csv", index=False)
        return summary_df

    def batch_correlate_cbt_with_bandpower(
        self,
        mat_folder: str | Path,
        edf_folder: str | Path,
        cbt_folder: str | Path,
        out_folder: str | Path,
        electrode: str,
        stage_mapping: Optional[Dict[Any, int]] = None,
        band: Tuple[float, float] = (0.5, 4.0),
        merge_tolerance_sec: int = 900,
        save_merged_timeseries: bool = True,
    ):
        """
        Batch correlation between CBT and EEG activity in a specified frequency band.

        Returns
        -------
        results_df : pd.DataFrame
            Per-subject correlation results.
        merged_dict : dict
            Per-subject merged time series used in the correlations.
        group_summary : pd.DataFrame
            Group-level Fisher-z summary when valid correlations exist.
        """
        stage_mapping = self._resolve_stage_mapping(stage_mapping)
        out_folder = Path(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)

        triplets = self._get_common_file_triplets(mat_folder, edf_folder, cbt_folder)
        results = []
        merged_dict = {}

        print(f"Found {len(triplets)} matched .mat/.edf/.xlsx filename sets.")

        for item in triplets:
            stem = item["stem"]
            print(f"\nProcessing correlation for: {stem}")

            try:
                self.load_yasa_from_edf(
                    edf_path=item["edf"],
                    hyp_path=item["mat"],
                    stage_mapping=stage_mapping,
                )

                if electrode not in self.input_data:
                    raise ValueError(
                        f"Electrode '{electrode}' not found. Available channels: {list(self.input_data.keys())}"
                    )

                temp_df = self.load_cbt(item["xlsx"])
                temp_df = self._prepare_temp_df(temp_df)

                if self.recording_start is None:
                    raise ValueError("recording_start is missing. Run load_yasa_from_edf first.")

                recording_start = self.recording_start

                band_df = self.save_spectrogram_csv(
                    electrode=electrode,
                    win_sec=self.hyp_window,
                    outpath=None,
                    filename=stem,
                    fmin=band[0],
                    fmax=band[1],
                )

                band_power = band_df.mean(axis=0)
                eeg_df = pd.DataFrame(
                    {
                        "time_sec": pd.to_numeric(band_power.index, errors="coerce"),
                        "band_power_db": pd.to_numeric(band_power.values, errors="coerce"),
                    }
                ).dropna()

                eeg_df["datetime"] = recording_start + pd.to_timedelta(eeg_df["time_sec"], unit="s")
                eeg_df = eeg_df.sort_values("datetime").copy()

                merged = pd.merge_asof(
                    eeg_df,
                    temp_df[["datetime", "temp"]].sort_values("datetime"),
                    on="datetime",
                    direction="nearest",
                    tolerance=pd.Timedelta(seconds=merge_tolerance_sec),
                ).dropna(subset=["temp", "band_power_db"])

                if len(merged) < 3:
                    raise ValueError(
                        f"Only {len(merged)} matched time points after alignment; need at least 3."
                    )

                print(f"Number of merged timestamps: {len(merged)}")

                rho, pval = spearmanr(merged["temp"], merged["band_power_db"])

                results.append(
                    {
                        "stem": stem,
                        "edf_file": str(item["edf"]),
                        "mat_file": str(item["mat"]),
                        "xlsx_file": str(item["xlsx"]),
                        "electrode": electrode,
                        "band_low_hz": band[0],
                        "band_high_hz": band[1],
                        "n_timepoints": len(merged),
                        "spearman_rho": rho,
                        "p_value": pval,
                        "status": "success",
                        "error": None,
                    }
                )

                if save_merged_timeseries:
                    subj_out = out_folder / stem
                    subj_out.mkdir(parents=True, exist_ok=True)
                    merged.to_csv(
                        subj_out / f"{stem}_{electrode}_{band[0]}-{band[1]}Hz_cbt_bandpower_timeseries.csv",
                        index=False,
                    )

                merged_dict[stem] = merged
                print(f"✅ Success for {stem}: rho={rho:.3f}, p={pval:.4g}, n={len(merged)}")

            except Exception as exc:
                results.append(
                    {
                        "stem": stem,
                        "edf_file": str(item["edf"]),
                        "mat_file": str(item["mat"]),
                        "xlsx_file": str(item["xlsx"]),
                        "electrode": electrode,
                        "band_low_hz": band[0],
                        "band_high_hz": band[1],
                        "n_timepoints": np.nan,
                        "spearman_rho": np.nan,
                        "p_value": np.nan,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                print(f"❌ Failed for {stem}: {exc}")

        results_df = pd.DataFrame(results)
        results_df.to_csv(
            out_folder / f"cbt_bandpower_correlation_{electrode}_{band[0]}-{band[1]}Hz_summary.csv",
            index=False,
        )

        valid = results_df[
            (results_df["status"] == "success")
            & (results_df["spearman_rho"].notna())
            & (results_df["spearman_rho"].abs() < 1)
        ].copy()

        if len(valid) > 0:
            valid["fisher_z"] = np.arctanh(valid["spearman_rho"])
            mean_z = valid["fisher_z"].mean()
            combined_r = np.tanh(mean_z)

            group_summary = pd.DataFrame(
                [
                    {
                        "n_subjects": len(valid),
                        "electrode": electrode,
                        "band_low_hz": band[0],
                        "band_high_hz": band[1],
                        "mean_spearman_rho": valid["spearman_rho"].mean(),
                        "median_spearman_rho": valid["spearman_rho"].median(),
                        "fisher_z_mean_rho": combined_r,
                    }
                ]
            )
            group_summary.to_csv(
                out_folder / f"cbt_bandpower_correlation_{electrode}_{band[0]}-{band[1]}Hz_group_summary.csv",
                index=False,
            )
        else:
            group_summary = pd.DataFrame()

        return results_df, merged_dict, group_summary
    
    

#QUADRANT ANALYSIS

# SLEEP START AND STOP HELPER FUNCTIONS

    @staticmethod
    def _load_sleep_windows(sleep_window_path: str | Path) -> pd.DataFrame:
        """
        Load Excel file with columns:
            PID, sleep onset, sleep offset

        PID should match the EDF/MAT stem, e.g. FEMS_01_S1.
        """
        sleep_windows = pd.read_excel(sleep_window_path).copy()
        sleep_windows.columns = sleep_windows.columns.str.strip()

        required_cols = {"PID", "sleep onset", "sleep offset"}
        missing = required_cols - set(sleep_windows.columns)

        if missing:
            raise ValueError(f"sleep_window_path is missing columns: {missing}")

        sleep_windows["PID"] = sleep_windows["PID"].astype(str).str.strip()
        sleep_windows = sleep_windows.dropna(subset=["PID", "sleep onset", "sleep offset"])

        return sleep_windows.set_index("PID")


    @staticmethod
    def _time_value_to_datetime_on_recording_date(value, recording_start: pd.Timestamp) -> pd.Timestamp:
        """
        Convert an Excel time value into a datetime on the EDF recording date.

        Handles:
            23:38:37
            11:57:30 PM
            Excel datetime/time objects
            Excel float time fractions
        """
        if pd.isna(value):
            return pd.NaT

        recording_start = pd.to_datetime(recording_start)

        if getattr(recording_start, "tzinfo", None) is not None:
            recording_start = recording_start.tz_localize(None)

        base_date = recording_start.date()

        if isinstance(value, pd.Timestamp):
            return pd.Timestamp.combine(base_date, value.time())

        if isinstance(value, datetime):
            return pd.Timestamp.combine(base_date, value.time())

        if isinstance(value, time):
            return pd.Timestamp.combine(base_date, value)

        if isinstance(value, (int, float, np.integer, np.floating)):
            # Excel time-only values are often fractions of a day.
            total_seconds = int(round((float(value) % 1) * 24 * 3600))
            total_seconds = total_seconds % (24 * 3600)
            return pd.Timestamp(base_date) + pd.to_timedelta(total_seconds, unit="s")

        parsed = pd.to_datetime(str(value), errors="coerce", format="mixed")

        if pd.isna(parsed):
            return pd.NaT

        return pd.Timestamp.combine(base_date, parsed.time())


    def _get_sleep_window_seconds(
        self,
        sleep_windows: pd.DataFrame,
        participant_id: str,
    ) -> dict:
        """
        Return sleep onset/offset as seconds from EDF recording start.
        """
        if self.recording_start is None:
            raise ValueError("recording_start is missing. Run load_yasa_from_edf first.")

        participant_id = str(participant_id).strip()

        if participant_id not in sleep_windows.index:
            raise ValueError(
                f"No sleep onset/offset row found for {participant_id}. "
                "PID in the Excel file must match the EDF/MAT file stem."
            )

        row = sleep_windows.loc[participant_id]

        recording_start = self._strip_tz(self.recording_start)

        onset_datetime = self._time_value_to_datetime_on_recording_date(
            row["sleep onset"],
            recording_start,
        )

        offset_datetime = self._time_value_to_datetime_on_recording_date(
            row["sleep offset"],
            recording_start,
        )

        if pd.isna(onset_datetime) or pd.isna(offset_datetime):
            raise ValueError(f"Could not parse sleep onset/offset for {participant_id}.")

        # If sleep onset is after midnight but recording started the prior evening.
        if onset_datetime < recording_start:
            onset_datetime += pd.Timedelta(days=1)

        # If offset is clock-time earlier than onset, it occurred the next day.
        if offset_datetime <= onset_datetime:
            offset_datetime += pd.Timedelta(days=1)

        sleep_start_sec = (onset_datetime - recording_start).total_seconds()
        sleep_end_sec = (offset_datetime - recording_start).total_seconds()

        recording_seconds = len(np.asarray(self.hypno)) / float(self.sampling_rate)

        # Clip to actual EDF/hypnogram bounds.
        sleep_start_sec = max(0.0, sleep_start_sec)
        sleep_end_sec = min(recording_seconds, sleep_end_sec)

        if sleep_end_sec <= sleep_start_sec:
            raise ValueError(
                f"Invalid sleep window for {participant_id}: "
                f"start={sleep_start_sec}, end={sleep_end_sec}"
            )

        return {
            "sleep_onset_datetime": onset_datetime,
            "sleep_offset_datetime": offset_datetime,
            "sleep_start_sec": sleep_start_sec,
            "sleep_end_sec": sleep_end_sec,
        }




    def hours_per_stage_per_quadrant(
        self,
        participant_id: Optional[str] = None,
        phase: Optional[str] = None,
        stage_codes: Optional[list[int]] = None,
        quadrant_labels: Optional[list[str]] = None,
        sleep_start_sec: Optional[float] = None,
        sleep_end_sec: Optional[float] = None,
        return_df: bool = False,
    ):
        """
        Calculate how much of each sleep stage occurs in each night quadrant
        for the currently loaded recording.

        Parameters
        ----------
        participant_id : str, optional
            Participant/file stem for the loaded recording.

        phase : str, optional
            Phase label, such as "follicular" or "luteal".

        stage_codes : list[int], optional
            YASA stage codes to include. Default is [1, 2, 3, 4].

        quadrant_labels : list[str], optional
            Labels for the four equal recording chunks.

        return_df : bool
            If True, return the long-form dataframe. If False, return the
            original nested dictionary format for backwards compatibility.

        Returns
        -------
        stages_per_quad : dict
            Nested dictionary of hours per stage per quadrant.

        df : pd.DataFrame
            Returned only when return_df=True. Long-form table with hours
            and percent columns.
        """
        self._check_loaded()

        if stage_codes is None:
            stage_codes = [0, 1, 2, 3, 4]

        if quadrant_labels is None:
            quadrant_labels = ["q1", "q2", "q3", "q4"]

        if len(quadrant_labels) != 4:
            raise ValueError("quadrant_labels must contain exactly four labels.")

        ####CROP HYPNOGRAM####

        hypno_full = np.asarray(self.hypno).astype(int)

        recording_seconds = len(hypno_full) / float(self.sampling_rate)

        if sleep_start_sec is None:
            sleep_start_sec = 0.0

        if sleep_end_sec is None:
            sleep_end_sec = recording_seconds

        sleep_start_sample = int(round(sleep_start_sec * float(self.sampling_rate)))
        sleep_end_sample = int(round(sleep_end_sec * float(self.sampling_rate)))

        sleep_start_sample = max(0, min(sleep_start_sample, len(hypno_full)))
        sleep_end_sample = max(0, min(sleep_end_sample, len(hypno_full)))

        if sleep_end_sample <= sleep_start_sample:
            raise ValueError(
                f"Invalid sleep crop for {participant_id}: "
                f"start_sample={sleep_start_sample}, end_sample={sleep_end_sample}"
            )

        hypno = hypno_full[sleep_start_sample:sleep_end_sample]
        quads = np.array_split(hypno, 4)

        sleep_start_sec = sleep_start_sample / float(self.sampling_rate)
        sleep_end_sec = sleep_end_sample / float(self.sampling_rate)
        quadrant_edges = np.linspace(sleep_start_sec, sleep_end_sec, 5)

        ###############

        stages_per_quad = {}
        rows = []

        for quadrant_num, (quad, data) in enumerate(zip(quadrant_labels, quads), start=1):
            stages_per_quad[quad] = {}

            for stage_code in stage_codes:
                stage_code = int(stage_code)
                stage_name = self.YASA_STAGE_NAMES.get(
                    stage_code,
                    f"Unknown code ({stage_code})",
                )

                n_samples = int(np.sum(data == stage_code))
                seconds = n_samples / float(self.sampling_rate)
                hours = seconds / 3600.0

                stages_per_quad[quad][stage_name] = hours

                rows.append(
                    {
                        "phase": phase,
                        "participant": participant_id,
                        "quadrant": quad,
                        "quadrant_num": quadrant_num,
                        "stage_code": stage_code,
                        "stage_name": stage_name,
                        "n_samples": n_samples,
                        "seconds": seconds,
                        "minutes": seconds / 60.0,
                        "hours": hours,

                        #CROPPING PARAMS
                        "sleep_start_seconds": sleep_start_sec,
                        "sleep_end_seconds": sleep_end_sec,
                        "quadrant_start_seconds": float(quadrant_edges[quadrant_num - 1]),
                        "quadrant_end_seconds": float(quadrant_edges[quadrant_num]),
                        "quadrant_start_hours": float(quadrant_edges[quadrant_num - 1]) / 3600.0,
                        "quadrant_end_hours": float(quadrant_edges[quadrant_num]) / 3600.0,
                    }
                )

        df = pd.DataFrame(rows)

        denominator = df.groupby("stage_code")["hours"].transform("sum")

        if isinstance(denominator, pd.Series):
            denominator_values = denominator.to_numpy()
        else:
            denominator_values = np.repeat(float(denominator), len(df))

        df["percent"] = np.where(
            denominator_values > 0,
            (df["hours"].to_numpy() / denominator_values) * 100.0,
            0.0,
        )

        self.stages_per_quad = stages_per_quad
        self.stages_per_quad_df = df

        if return_df:
            return stages_per_quad, df

        return stages_per_quad


    def average_temperature_per_quadrant(
        self,
        temp_path: str | Path,
        participant_id: Optional[str] = None,
        phase: Optional[str] = None,
        quadrant_labels: Optional[list[str]] = None,
        sleep_start_sec: Optional[float] = None,
        sleep_end_sec: Optional[float] = None,
        return_df: bool = True,
    ):
        """
        Calculate average CBT temperature in each recording quadrant
        for the currently loaded participant.
        """
        self._check_loaded()

        if self.recording_start is None:
            raise ValueError("recording_start is missing. Run load_yasa_from_edf first.")

        if quadrant_labels is None:
            quadrant_labels = ["q1", "q2", "q3", "q4"]

        if len(quadrant_labels) != 4:
            raise ValueError("quadrant_labels must contain exactly four labels.")

        temp_df = self.load_cbt(temp_path).copy()

        recording_start = self._strip_tz(self.recording_start)

        temp_df["seconds_from_start"] = (
            temp_df["datetime"] - recording_start
        ).dt.total_seconds()


        ####HYPNOGRAM CROPPING#####
        recording_seconds = len(np.asarray(self.hypno)) / float(self.sampling_rate)

        if sleep_start_sec is None:
            sleep_start_sec = 0.0

        if sleep_end_sec is None:
            sleep_end_sec = recording_seconds

        sleep_start_sec = max(0.0, float(sleep_start_sec))
        sleep_end_sec = min(recording_seconds, float(sleep_end_sec))

        if sleep_end_sec <= sleep_start_sec:
            raise ValueError(
                f"Invalid sleep crop for {participant_id}: "
                f"start={sleep_start_sec}, end={sleep_end_sec}"
            )

        quadrant_edges = np.linspace(sleep_start_sec, sleep_end_sec, 5)


        ########################

        rows = []

        for quadrant_num, quadrant in enumerate(quadrant_labels, start=1):
            start_seconds = float(quadrant_edges[quadrant_num - 1])
            end_seconds = float(quadrant_edges[quadrant_num])

            if quadrant_num < 4:
                mask = (
                    (temp_df["seconds_from_start"] >= start_seconds)
                    & (temp_df["seconds_from_start"] < end_seconds)
                )
            else:
                mask = (
                    (temp_df["seconds_from_start"] >= start_seconds)
                    & (temp_df["seconds_from_start"] <= end_seconds)
                )

            quadrant_temp = temp_df.loc[mask, "temp"].dropna()

            rows.append(
                {
                    "phase": phase,
                    "participant": participant_id,
                    "quadrant": quadrant,
                    "quadrant_num": quadrant_num,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "start_hours": start_seconds / 3600.0,
                    "end_hours": end_seconds / 3600.0,
                    "n_temperature_samples": int(len(quadrant_temp)),
                    "average_temperature": (
                        float(quadrant_temp.mean())
                        if len(quadrant_temp) > 0
                        else np.nan
                    ),
                }
            )

        df = pd.DataFrame(rows)

        self.avg_temp_per_quad = df
        return df

    def plot_stage_hours_by_quadrant(
        self,
        phase_folders: Optional[Dict[str, str | Path]] = None,
        follicular_folder: Optional[str | Path] = None,
        luteal_folder: Optional[str | Path] = None,
        save_dir: Optional[str | Path] = None,
        stage_mapping: Optional[Dict[Any, int]] = None,
        stage_codes: Optional[list[int]] = None,
        sleep_window_path: Optional[str | Path] = None,
        make_participant_plots: bool = True,
        show: bool = False,
    ):
        """
        Batch analyze sleep-stage quadrant percentages for follicular vs luteal
        folders and generate line plots.

        Each phase folder is expected to contain matching .edf and .mat files.
        The .xlsx CBT files can be present in the same folder, but they are not
        needed for this sleep-stage quadrant analysis. Set require_xlsx=True if
        you want to process only stems that have EDF, MAT, and XLSX files.

        This method:
        1. Loads each participant using load_yasa_from_edf().
        2. Splits that participant's hypnogram into four equal night quadrants.
        3. Calculates hours and percentages per stage per quadrant.
        4. Sums stage hours across participants within each phase.
        5. Recomputes phase-level percentages from the summed hours.
        6. Saves group-level line plots and optional per-participant line plots.

        Parameters
        ----------
        phase_folders : dict, optional
            Example:
            {
                "follicular": r"path/to/follicular",
                "luteal": r"path/to/luteal",
            }

        follicular_folder, luteal_folder : str or Path, optional
            Convenience arguments if you do not want to pass phase_folders.

        save_dir : str or Path, optional
            Output folder for CSVs and PNG figures.

        stage_mapping : dict, optional
            Mapping passed to load_yasa_from_edf().

        stage_codes : list[int], optional
            YASA stage codes to include. Default is [0, 1, 2, 3, 4].

        percent_denominator : str
            Default is "stage_total", so each stage sums to 100% across Q1-Q4
            within each participant or phase.

        require_xlsx : bool
            If True, only process participants with matching EDF, MAT, and XLSX
            stems. If False, only EDF and MAT are required.

        make_participant_plots : bool
            Whether to save one line graph per participant.

        show : bool
            Whether to display plots.

        Returns
        -------
        results : dict
            {
                "participant_df": long-form per-participant table,
                "group_df": phase-level table after summing across participants,
                "status_df": processing status table,
                "group_figs": group-level matplotlib figures,
                "participant_figs": per-participant matplotlib figures,
            }
        """
        stage_mapping = self._resolve_stage_mapping(stage_mapping)

        if stage_codes is None:
            stage_codes = [0, 1, 2, 3, 4]

        sleep_windows = None
        if sleep_window_path is not None:
            sleep_windows = self._load_sleep_windows(sleep_window_path)

        if phase_folders is None:
            phase_folders = {}

            if follicular_folder is not None:
                phase_folders["follicular"] = follicular_folder

            if luteal_folder is not None:
                phase_folders["luteal"] = luteal_folder

        if not phase_folders:
            raise ValueError(
                "Pass either phase_folders or follicular_folder/luteal_folder."
            )

        phase_folders = {
            str(phase): Path(folder)
            for phase, folder in phase_folders.items()
        }

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            group_plot_dir = save_dir / "group_plots"
            participant_plot_dir = save_dir / "participant_plots"
            group_plot_dir.mkdir(parents=True, exist_ok=True)
            participant_plot_dir.mkdir(parents=True, exist_ok=True)
        else:
            group_plot_dir = None
            participant_plot_dir = None

        def _safe_name(value):
            value = str(value)
            return "".join(
                char if char.isalnum() or char in ("-", "_") else "_"
                for char in value
            )

        def _phase_file_sets(folder):
            files = [path for path in folder.iterdir() if path.is_file()]

            edf_files = {
                path.stem: path
                for path in files
                if path.suffix.lower() == ".edf"
            }
            mat_files = {
                path.stem: path
                for path in files
                if path.suffix.lower() == ".mat"
            }
            xlsx_files = {
                path.stem: path
                for path in files
                if path.suffix.lower() == ".xlsx"
            }

            common = set(edf_files) & set(mat_files)

            return [
                {
                    "stem": stem,
                    "edf": edf_files[stem],
                    "mat": mat_files[stem],
                    "xlsx": xlsx_files.get(stem),
                }
                for stem in sorted(common)
            ]

        participant_dfs = []
        participant_temp_dfs = []
        status_rows = []
        participant_nadir_rate_rows = []
        participant_nadir_windows = {}

        for phase, folder in phase_folders.items():
            if not folder.exists():
                raise FileNotFoundError(f"Folder does not exist: {folder}")

            file_sets = _phase_file_sets(folder)

            print(f"\n{phase}: found {len(file_sets)} matched file sets.")

            for item in file_sets:
                stem = item["stem"]
                print(f"Processing {phase}: {stem}")

                try:
                    self.load_yasa_from_edf(
                        edf_path=item["edf"],
                        hyp_path=item["mat"],
                        stage_mapping=stage_mapping,
                    )

                    sleep_start_sec = None
                    sleep_end_sec = None
                    sleep_window_info = {}

                    if sleep_windows is not None:
                        sleep_window_info = self._get_sleep_window_seconds(
                            sleep_windows=sleep_windows,
                            participant_id=stem,
                        )

                        sleep_start_sec = sleep_window_info["sleep_start_sec"]
                        sleep_end_sec = sleep_window_info["sleep_end_sec"]

                    _, participant_df = self.hours_per_stage_per_quadrant(
                        participant_id=stem,
                        phase=phase,
                        stage_codes=stage_codes,
                        sleep_start_sec=sleep_start_sec,
                        sleep_end_sec=sleep_end_sec,
                        return_df=True,
                    )

                    participant_dfs.append(participant_df)
                    if item["xlsx"] is not None:
                        participant_temp_df = self.average_temperature_per_quadrant(
                            temp_path=item["xlsx"],
                            participant_id=stem,
                            phase=phase,
                            sleep_start_sec=sleep_start_sec,
                            sleep_end_sec=sleep_end_sec,
                            return_df=True,
                        )

                        participant_temp_dfs.append(participant_temp_df)
                    else:
                        print(f"⚠️ No matching temperature .xlsx file for {stem}")

                    status_rows.append(
                        {
                            "phase": phase,
                            "participant": stem,
                            "edf_file": str(item["edf"]),
                            "mat_file": str(item["mat"]),
                            "status": "success",
                            "error": None,

                            #HYPNOGRAM CROPPING PARAMS
                            "sleep_onset_datetime": sleep_window_info.get("sleep_onset_datetime"),
                            "sleep_offset_datetime": sleep_window_info.get("sleep_offset_datetime"),
                            "sleep_start_sec": sleep_start_sec,
                            "sleep_end_sec": sleep_end_sec,

                        }
                    )

                    print(f"✅ Success: {stem}")

                except Exception as exc:
                    status_rows.append(
                        {
                            "phase": phase,
                            "participant": stem,
                            "edf_file": str(item["edf"]),
                            "mat_file": str(item["mat"]),
                            "status": "failed",
                            "error": str(exc),
                        }
                    )

                    print(f"❌ Failed: {stem}: {exc}")

        status_df = pd.DataFrame(status_rows)

        if not participant_dfs:
            if save_dir is not None:
                status_df.to_csv(save_dir / "quadrant_processing_status.csv", index=False)

            raise ValueError("No participants were successfully processed.")

        participant_df = pd.concat(participant_dfs, ignore_index=True)

        if participant_temp_dfs:
            participant_temp_df = pd.concat(
                participant_temp_dfs,
                ignore_index=True,
            )
        else:
            participant_temp_df = pd.DataFrame()

        group_cols = [
            "phase",
            "quadrant",
            "quadrant_num",
            "stage_code",
            "stage_name",
        ]

        group_df = (
            participant_df
            .groupby(group_cols, as_index=False)
            .agg(
                n_samples=("n_samples", "sum"),
                seconds=("seconds", "sum"),
                minutes=("minutes", "sum"),
                hours=("hours", "sum"),
                n_participants=("participant", "nunique"),
            )
        )

        denominator = group_df.groupby(["phase", "stage_code"])["hours"].transform("sum")

        group_df["percent"] = np.where(
            denominator.to_numpy() > 0,
            (group_df["hours"].to_numpy() / denominator.to_numpy()) * 100.0,
            0.0,
        )

        if save_dir is not None:
            participant_df.to_csv(
                save_dir / "participant_quadrant_stage_percent.csv",
                index=False,
            )
            group_df.to_csv(
                save_dir / "phase_quadrant_stage_percent.csv",
                index=False,
            )
            status_df.to_csv(
                save_dir / "quadrant_processing_status.csv",
                index=False,
            )

            if not participant_temp_df.empty:
                participant_temp_df.to_excel(
                    save_dir / "participant_quadrant_temperature.xlsx",
                    index=False,
                )

        y_label = "% of total time in this stage"

        quadrants = ["q1", "q2", "q3", "q4"]
        x = np.arange(1, 5)

        group_figs = {}

        for stage_code in stage_codes:
            stage_code = int(stage_code)
            stage_name = self.YASA_STAGE_NAMES.get(
                stage_code,
                f"Unknown code ({stage_code})",
            )

            fig, ax = plt.subplots(figsize=(8, 5))

            for phase in phase_folders:
                plot_df = group_df[
                    (group_df["phase"] == phase)
                    & (group_df["stage_code"] == stage_code)
                ]

                y = []
                for quadrant_num in x:
                    value = plot_df.loc[
                        plot_df["quadrant_num"] == quadrant_num,
                        "percent",
                    ]

                    if len(value) == 0:
                        y.append(np.nan)
                    else:
                        y.append(float(value.iloc[0]))

                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2,
                    label=phase,
                )

            ax.set_title(
                f"{stage_name}: Sleep Stage Distribution Across Night Quadrants"
            )
            ax.set_xlabel("Night Quadrant")
            ax.set_ylabel(y_label)
            ax.set_xticks(x)
            ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)

            fig.tight_layout()

            if group_plot_dir is not None:
                fig.savefig(
                    group_plot_dir / f"group_{_safe_name(stage_name)}_quadrant_percent.png",
                    dpi=300,
                    bbox_inches="tight",
                )

            if not show:
                plt.close(fig)

            group_figs[stage_name] = fig

        participant_figs = {}

        if make_participant_plots:
            for (phase, participant), plot_df in participant_df.groupby(["phase", "participant"]):
                fig, ax = plt.subplots(figsize=(8, 5))

                for stage_code in stage_codes:
                    stage_code = int(stage_code)
                    stage_name = self.YASA_STAGE_NAMES.get(
                        stage_code,
                        f"Unknown code ({stage_code})",
                    )

                    stage_df = plot_df[plot_df["stage_code"] == stage_code]

                    y = []
                    for quadrant_num in x:
                        value = stage_df.loc[
                            stage_df["quadrant_num"] == quadrant_num,
                            "percent",
                        ]

                        if len(value) == 0:
                            y.append(np.nan)
                        else:
                            y.append(float(value.iloc[0]))

                    ax.plot(
                        x,
                        y,
                        marker="o",
                        linewidth=2,
                        label=stage_name,
                    )

                ax.set_title(
                    f"{participant} ({phase}): Sleep Stage Distribution Across Night Quadrants"
                )
                ax.set_xlabel("Night Quadrant")
                ax.set_ylabel(y_label)
                ax.set_xticks(x)
                ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
                ax.set_ylim(bottom=0)
                ax.grid(alpha=0.25)
                ax.legend(frameon=False)

                fig.tight_layout()

                if participant_plot_dir is not None:
                    phase_dir = participant_plot_dir / _safe_name(phase)
                    phase_dir.mkdir(parents=True, exist_ok=True)

                    fig.savefig(
                        phase_dir / f"{_safe_name(participant)}_quadrant_stage_percent.png",
                        dpi=300,
                        bbox_inches="tight",
                    )

                if not show:
                    plt.close(fig)

                participant_figs[(phase, participant)] = fig

        results = {
            "participant_df": participant_df,
            "group_df": group_df,
            "participant_temp_df": participant_temp_df,
            "status_df": status_df,
            "group_figs": group_figs,
            "participant_figs": participant_figs,
        }

        return results  
    


### SLOPE ANALYSIS ###

    def _base_cbt_stage_participant_row(self, item: dict) -> dict:
        return {
            "stem": item["stem"],
            "edf_file": str(item["edf"]),
            "mat_file": str(item["mat"]),
            "xlsx_file": str(item["xlsx"]),
        }


    def _empty_cbt_stage_metrics(self, stage) -> dict:
        return {
            "recording_start": np.nan,
            "recording_end": np.nan,

            "hyp_is_sample_level": np.nan,
            "hyp_step_sec": np.nan,
            "hyp_duration_sec": np.nan,

            "sleep_onset_datetime": np.nan,
            "sleep_onset_sec_from_recording_start": np.nan,
            "sleep_onset_min_from_recording_start": np.nan,
            "sleep_onset_hr_from_recording_start": np.nan,

            f"{stage}_onset_datetime": np.nan,
            f"{stage}_onset_sec_from_recording_start": np.nan,
            f"{stage}_onset_min_from_recording_start": np.nan,
            f"{stage}_onset_hr_from_recording_start": np.nan,

            f"minutes_sleep_onset_to_{stage}": np.nan,
            f"hours_sleep_onset_to_{stage}": np.nan,

            f"first_{stage}_bout_end_index_exclusive": np.nan,
            f"first_{stage}_bout_duration_sec": np.nan,
            f"first_{stage}_bout_duration_min": np.nan,
            f"first_{stage}_bout_duration_hr": np.nan,

            "n_temp_points_slope_window": np.nan,
            "cbt_at_or_after_sleep_onset": np.nan,
            f"cbt_at_or_before_{stage}_onset": np.nan,
            f"cbt_change_sleep_to_{stage}": np.nan,

            "cbt_slope_degC_per_hour": np.nan,
            "cbt_slope_degC_per_min": np.nan,

            "within_subject_cbt_slope_p_value": np.nan,
            "within_subject_r_value": np.nan,
            "within_subject_stderr": np.nan,
            "within_subject_intercept": np.nan,

            "n_temp_points_nadir_window": np.nan,
            "cbt_nadir_datetime": np.nan,
            "cbt_nadir_temp": np.nan,
            "minutes_sleep_onset_to_cbt_nadir": np.nan,
            "hours_sleep_onset_to_cbt_nadir": np.nan,
            "cbt_at_or_after_sleep_onset_for_nadir": np.nan,
            "cbt_change_sleep_onset_to_nadir": np.nan,
            "cbt_rate_sleep_onset_to_nadir_degC_per_hour": np.nan,
            "cbt_rate_sleep_onset_to_nadir_degC_per_min": np.nan,
        }


    def _failed_cbt_stage_participant_row(self, item: dict, error: Exception | str, stage_name: str) -> dict:
        row = self._base_cbt_stage_participant_row(item)
        row.update(self._empty_cbt_stage_metrics(stage_name))
        row["status"] = "failed"
        row["error"] = str(error)
        return row


    def _infer_hypno_timing(self, hyp_data: np.ndarray) -> dict:
        """
        Determine whether hypnogram is sample-level or epoch-level.

        Returns
        -------
        dict with:
            step_sec : seconds represented by one hypnogram element
            duration_sec : total hypnogram duration
            hyp_is_sample_level : bool
        """

        hyp_data = np.asarray(hyp_data)

        sampling_rate = getattr(self, "sampling_rate", None)
        hyp_window = float(getattr(self, "hyp_window", 30.0))

        if sampling_rate is not None:
            sampling_rate = float(sampling_rate)

            possible_data_lengths = (
                [len(ch_data) for ch_data in self.input_data.values()]
                if hasattr(self, "input_data") and self.input_data
                else []
            )

            hyp_is_sample_level = (
                len(possible_data_lengths) > 0
                and any(
                    abs(len(hyp_data) - n) <= sampling_rate
                    for n in possible_data_lengths
                )
            )

            if hyp_is_sample_level:
                step_sec = 1.0 / sampling_rate
                return {
                    "step_sec": step_sec,
                    "duration_sec": len(hyp_data) * step_sec,
                    "hyp_is_sample_level": True,
                }

        step_sec = hyp_window

        return {
            "step_sec": step_sec,
            "duration_sec": len(hyp_data) * step_sec,
            "hyp_is_sample_level": False,
        }


    def _hyp_index_to_seconds(self, hyp_index: int, hyp_timing: dict) -> float:
        return float(hyp_index) * float(hyp_timing["step_sec"])


    def _find_sleep_and_first_stage_indices(
        self,
        hyp_data: np.ndarray,
        sleep_codes: list[int],
        stage_name: str,
        stage_code: int,
    ) -> tuple[int, int]:
        hyp_data = np.asarray(hyp_data)

        sleep_indices = np.flatnonzero(np.isin(hyp_data, sleep_codes))

        if len(sleep_indices) == 0:
            raise ValueError(
                f"No sleep onset found. None of the sleep codes "
                f"{sleep_codes} appear in hypnogram."
            )

        first_sleep_index = int(sleep_indices[0])

        stage_indices_after_sleep = np.flatnonzero(
            hyp_data[first_sleep_index:] == stage_code
        )

        if len(stage_indices_after_sleep) == 0:
            raise ValueError(
                f"No {stage_name} onset found after sleep onset. {stage_name} code used: {stage_code}."
            )

        first_stage_index = int(first_sleep_index + stage_indices_after_sleep[0])

        return first_sleep_index, first_stage_index


    def _find_continuous_stage_bout_from_index(
        self,
        hyp_data: np.ndarray,
        start_index: int,
        stage_code: int,
        step_sec: float,
    ) -> dict:
        """
        Finds the continuous bout of `stage_code` beginning at `start_index`.

        end_index_exclusive is the first index after the bout.
        """

        hyp_data = np.asarray(hyp_data)

        if hyp_data[start_index] != stage_code:
            raise ValueError(
                f"Expected stage code {stage_code} at start_index={start_index}, "
                f"but found {hyp_data[start_index]}."
            )

        tail = hyp_data[start_index:]
        first_non_stage = np.flatnonzero(tail != stage_code)

        if len(first_non_stage) == 0:
            end_index_exclusive = len(hyp_data)
        else:
            end_index_exclusive = start_index + int(first_non_stage[0])

        duration_sec = (end_index_exclusive - start_index) * step_sec

        return {
            "end_index_exclusive": end_index_exclusive,
            "duration_sec": duration_sec,
            "duration_min": duration_sec / 60,
            "duration_hr": duration_sec / 3600,
        }


    def _calculate_cbt_slope_between_times(
        self,
        temp_df: pd.DataFrame,
        start_datetime,
        end_datetime,
        min_temp_points: int = 3,
    ) -> tuple[pd.DataFrame, object]:
        slope_df = temp_df[
            (temp_df["datetime"] >= start_datetime)
            & (temp_df["datetime"] <= end_datetime)
        ].copy()

        slope_df = slope_df.dropna(subset=["datetime", "temp"])
        slope_df = slope_df.sort_values("datetime").copy()

        if len(slope_df) < min_temp_points:
            raise ValueError(
                f"Only {len(slope_df)} CBT points between sleep onset and N3 onset; "
                f"need at least {min_temp_points}."
            )

        slope_df["hours_from_sleep_onset"] = (
            slope_df["datetime"] - start_datetime
        ).dt.total_seconds() / 3600

        slope_df["minutes_from_sleep_onset"] = (
            slope_df["datetime"] - start_datetime
        ).dt.total_seconds() / 60

        if slope_df["hours_from_sleep_onset"].nunique() < 2:
            raise ValueError(
                "CBT slope cannot be calculated because all temperature points "
                "have the same timestamp relative to sleep onset."
            )

        reg = linregress(
            slope_df["hours_from_sleep_onset"],
            slope_df["temp"],
        )

        return slope_df, reg


    def _calculate_time_to_cbt_nadir(
        self,
        temp_df: pd.DataFrame,
        sleep_onset_datetime,
        recording_end_datetime,
        min_temp_points: int = 1,
    ) -> tuple[dict, pd.DataFrame]:
        """
        Time to nadir = time from sleep onset to the lowest CBT value
        after sleep onset within the recording window.
        """

        nadir_df = temp_df[
            (temp_df["datetime"] >= sleep_onset_datetime)
            & (temp_df["datetime"] <= recording_end_datetime)
        ].copy()

        nadir_df = nadir_df.dropna(subset=["datetime", "temp"])
        nadir_df = nadir_df.sort_values("datetime").copy()

        if len(nadir_df) < min_temp_points:
            raise ValueError(
                f"Only {len(nadir_df)} CBT points after sleep onset in recording window; "
                f"need at least {min_temp_points} to calculate CBT nadir."
            )

        nadir_df["hours_from_sleep_onset"] = (
            nadir_df["datetime"] - sleep_onset_datetime
        ).dt.total_seconds() / 3600

        nadir_df["minutes_from_sleep_onset"] = (
            nadir_df["datetime"] - sleep_onset_datetime
        ).dt.total_seconds() / 60

        nadir_idx = nadir_df["temp"].idxmin()
        nadir_row = nadir_df.loc[nadir_idx]

        nadir_datetime = nadir_row["datetime"]
        nadir_temp = nadir_row["temp"]

        time_to_nadir_sec = (
            nadir_datetime - sleep_onset_datetime
        ).total_seconds()

        nadir_info = {
            "n_temp_points_nadir_window": len(nadir_df),
            "cbt_nadir_datetime": nadir_datetime,
            "cbt_nadir_temp": nadir_temp,
            "minutes_sleep_onset_to_cbt_nadir": time_to_nadir_sec / 60,
            "hours_sleep_onset_to_cbt_nadir": time_to_nadir_sec / 3600,
        }

        return nadir_info, nadir_df


    def _extract_cbt_stage_features_for_participant(
        self,
        item: dict,
        stage_mapping: dict,
        sleep_codes: list[int],
        stage_code: int,
        stage_name: str,
        min_temp_points: int,
    ) -> tuple[dict, dict]:
        stem = item["stem"]


        self.load_yasa_from_edf(
            edf_path=item["edf"],
            hyp_path=item["mat"],
            stage_mapping=stage_mapping,
        )

        if self.recording_start is None:
            raise ValueError("recording_start is missing. Run load_yasa_from_edf first.")

        if self.hypno is None:
            raise ValueError("hypnogram is missing. Run load_yasa_from_edf first.")

        recording_start = self.recording_start

        temp_df = self.load_cbt(item["xlsx"])
        temp_df = self._prepare_temp_df(temp_df)
        temp_df = temp_df.sort_values("datetime").copy()

        hyp_data = np.asarray(self.hypno)

        hyp_timing = self._infer_hypno_timing(hyp_data)
        step_sec = hyp_timing["step_sec"]

        recording_end_datetime = recording_start + pd.to_timedelta(
            hyp_timing["duration_sec"],
            unit="s",
        )

        first_sleep_index, first_stage_index = self._find_sleep_and_first_stage_indices(
            hyp_data=hyp_data,
            sleep_codes=sleep_codes,
            stage_name=stage_name,
            stage_code=stage_code,
        )

        sleep_onset_sec = self._hyp_index_to_seconds(first_sleep_index, hyp_timing)
        stage_onset_sec = self._hyp_index_to_seconds(first_stage_index, hyp_timing)

        sleep_onset_datetime = recording_start + pd.to_timedelta(
            sleep_onset_sec,
            unit="s",
        )

        stage_onset_datetime = recording_start + pd.to_timedelta(
            stage_onset_sec,
            unit="s",
        )

        slope_error = None

        first_stage_bout = self._find_continuous_stage_bout_from_index(
            hyp_data=hyp_data,
            start_index=first_stage_index,
            stage_code=stage_code,
            step_sec=step_sec,
        )

        try:
            if stage_onset_datetime <= sleep_onset_datetime:
                raise ValueError(
                    f"{stage_name} occurs at sleep onset, so there is no pre-stage "
                    "window for CBT slope calculation."
                )

            slope_df, within_subject_reg = self._calculate_cbt_slope_between_times(
                temp_df=temp_df,
                start_datetime=sleep_onset_datetime,
                end_datetime=stage_onset_datetime,
                min_temp_points=min_temp_points,
            )

            n_temp_points_slope_window = len(slope_df)
            cbt_at_or_after_sleep_onset = slope_df["temp"].iloc[0]
            cbt_at_or_before_stage_onset = slope_df["temp"].iloc[-1]
            cbt_change_sleep_to_stage = slope_df["temp"].iloc[-1] - slope_df["temp"].iloc[0]

            cbt_slope_degC_per_hour = within_subject_reg.slope
            cbt_slope_degC_per_min = within_subject_reg.slope / 60
            within_subject_cbt_slope_p_value = within_subject_reg.pvalue
            within_subject_r_value = within_subject_reg.rvalue
            within_subject_stderr = within_subject_reg.stderr
            within_subject_intercept = within_subject_reg.intercept

        except Exception as exc:
            slope_error = str(exc)

            slope_df = pd.DataFrame()

            n_temp_points_slope_window = 0
            cbt_at_or_after_sleep_onset = np.nan
            cbt_at_or_before_stage_onset = np.nan
            cbt_change_sleep_to_stage = np.nan

            cbt_slope_degC_per_hour = np.nan
            cbt_slope_degC_per_min = np.nan
            within_subject_cbt_slope_p_value = np.nan
            within_subject_r_value = np.nan
            within_subject_stderr = np.nan
            within_subject_intercept = np.nan

        nadir_info, nadir_df = self._calculate_time_to_cbt_nadir(
            temp_df=temp_df,
            sleep_onset_datetime=sleep_onset_datetime,
            recording_end_datetime=recording_end_datetime,
            min_temp_points=1,
        )

        cbt_at_or_after_sleep_onset_for_nadir = float(nadir_df["temp"].iloc[0])
        cbt_nadir_temp = float(nadir_info["cbt_nadir_temp"])
        hours_sleep_onset_to_cbt_nadir = float(
            nadir_info["hours_sleep_onset_to_cbt_nadir"]
        )

        cbt_change_sleep_onset_to_nadir = (
            cbt_nadir_temp - cbt_at_or_after_sleep_onset_for_nadir
        )

        if hours_sleep_onset_to_cbt_nadir > 0:
            cbt_rate_sleep_onset_to_nadir_degC_per_hour = (
                cbt_change_sleep_onset_to_nadir
                / hours_sleep_onset_to_cbt_nadir
            )
        else:
            cbt_rate_sleep_onset_to_nadir_degC_per_hour = np.nan

        cbt_rate_sleep_onset_to_nadir_degC_per_min = (
            cbt_rate_sleep_onset_to_nadir_degC_per_hour / 60.0
            if pd.notna(cbt_rate_sleep_onset_to_nadir_degC_per_hour)
            else np.nan
        )

        row = self._base_cbt_stage_participant_row(item)

        row.update(
            {
                "recording_start": recording_start,
                "recording_end": recording_end_datetime,

                "hyp_is_sample_level": hyp_timing["hyp_is_sample_level"],
                "hyp_step_sec": hyp_timing["step_sec"],
                "hyp_duration_sec": hyp_timing["duration_sec"],

                "sleep_onset_datetime": sleep_onset_datetime,
                "sleep_onset_sec_from_recording_start": sleep_onset_sec,
                "sleep_onset_min_from_recording_start": sleep_onset_sec / 60,
                "sleep_onset_hr_from_recording_start": sleep_onset_sec / 3600,

                f"{stage_name}_onset_datetime": stage_onset_datetime,
                f"{stage_name}_onset_sec_from_recording_start": stage_onset_sec,
                f"{stage_name}_onset_min_from_recording_start": stage_onset_sec / 60,
                f"{stage_name}_onset_hr_from_recording_start": stage_onset_sec / 3600,

                f"minutes_sleep_onset_to_{stage_name}": (
                    stage_onset_sec - sleep_onset_sec
                ) / 60,

                f"hours_sleep_onset_to_{stage_name}": (
                    stage_onset_sec - sleep_onset_sec
                ) / 3600,

                f"first_{stage_name}_bout_end_index_exclusive": (
                    first_stage_bout["end_index_exclusive"]
                ),

                f"first_{stage_name}_bout_duration_sec": first_stage_bout["duration_sec"],
                f"first_{stage_name}_bout_duration_min": first_stage_bout["duration_min"],
                f"first_{stage_name}_bout_duration_hr": first_stage_bout["duration_hr"],

                "n_temp_points_slope_window": n_temp_points_slope_window,

                "cbt_at_or_after_sleep_onset": cbt_at_or_after_sleep_onset,
                f"cbt_at_or_before_{stage_name}_onset": cbt_at_or_before_stage_onset,
                f"cbt_change_sleep_to_{stage_name}": cbt_change_sleep_to_stage,

                "cbt_slope_degC_per_hour": cbt_slope_degC_per_hour,
                "cbt_slope_degC_per_min": cbt_slope_degC_per_min,

                "within_subject_cbt_slope_p_value": within_subject_cbt_slope_p_value,
                "within_subject_r_value": within_subject_r_value,
                "within_subject_stderr": within_subject_stderr,
                "within_subject_intercept": within_subject_intercept,

                "slope_status": "success" if slope_error is None else "skipped",
                "slope_error": slope_error,
            }
        )

        row.update(nadir_info)

        row.update(
            {
                "cbt_at_or_after_sleep_onset_for_nadir": cbt_at_or_after_sleep_onset_for_nadir,
                "cbt_change_sleep_onset_to_nadir": cbt_change_sleep_onset_to_nadir,
                "cbt_rate_sleep_onset_to_nadir_degC_per_hour": cbt_rate_sleep_onset_to_nadir_degC_per_hour,
                "cbt_rate_sleep_onset_to_nadir_degC_per_min": cbt_rate_sleep_onset_to_nadir_degC_per_min,
            }
        )

        row["status"] = "success"
        row["error"] = None

        participant_data = {
            "slope_window": slope_df,
            "nadir_search_window": nadir_df,
        }

        return row, participant_data


    def _format_p_value(self, p_value: float) -> str:
        if pd.isna(p_value):
            return "NA"

        if p_value < 0.001:
            return f"{p_value:.2e}"

        return f"{p_value:.3f}"

    def _add_sleep_stage_quadrant_predictors_to_row(
        self,
        row: dict,
        participant_df: pd.DataFrame,
    ) -> dict:
        """
        Add predictors to a participant row:

            percent_N3_q1
            percent_N3_q2
            percent_REM_q4
            total_N3_hours
            total_REM_minutes
        """
        row = row.copy()

        for _, stage_row in participant_df.iterrows():
            stage_name = str(stage_row["stage_name"]).replace(" ", "_")
            quadrant = str(stage_row["quadrant"])

            row[f"percent_{stage_name}_{quadrant}"] = stage_row["percent"]
            row[f"hours_{stage_name}_{quadrant}"] = stage_row["hours"]

        total_stage_df = (
            participant_df
            .groupby("stage_name", as_index=False)
            .agg(
                total_hours=("hours", "sum"),
                total_minutes=("minutes", "sum"),
            )
        )

        for _, stage_row in total_stage_df.iterrows():
            stage_name = str(stage_row["stage_name"]).replace(" ", "_")

            row[f"total_{stage_name}_hours"] = stage_row["total_hours"]
            row[f"total_{stage_name}_minutes"] = stage_row["total_minutes"]

        return row


    def _fit_group_regression(
        self,
        results_df: pd.DataFrame,
        predictor: str,
        outcome: str,
        model_name: str,
        min_subjects: int = 3,
    ) -> dict:
        valid = results_df[
            (results_df["status"] == "success")
            & results_df[predictor].notna()
            & results_df[outcome].notna()
        ].copy()

        valid[predictor] = pd.to_numeric(valid[predictor], errors="coerce")
        valid[outcome] = pd.to_numeric(valid[outcome], errors="coerce")

        valid = valid[
            np.isfinite(valid[predictor])
            & np.isfinite(valid[outcome])
        ].copy()

        if len(valid) < min_subjects:
            return {
                "model_name": model_name,
                "n_subjects": len(valid),
                "predictor": predictor,
                "outcome": outcome,
                "beta_slope": np.nan,
                "intercept": np.nan,
                "r_value": np.nan,
                "r_squared": np.nan,
                "p_value": np.nan,
                "stderr": np.nan,
                "status": "failed",
                "error": f"Need at least {min_subjects} valid participants.",
            }

        if valid[predictor].nunique() <= 1:
            return {
                "model_name": model_name,
                "n_subjects": len(valid),
                "predictor": predictor,
                "outcome": outcome,
                "beta_slope": np.nan,
                "intercept": np.nan,
                "r_value": np.nan,
                "r_squared": np.nan,
                "p_value": np.nan,
                "stderr": np.nan,
                "status": "failed",
                "error": f"Predictor {predictor} has only one unique value.",
            }

        reg = linregress(
            valid[predictor],
            valid[outcome],
        )

        return {
            "model_name": model_name,
            "n_subjects": len(valid),
            "predictor": predictor,
            "outcome": outcome,
            "beta_slope": reg.slope,
            "intercept": reg.intercept,
            "r_value": reg.rvalue,
            "r_squared": reg.rvalue ** 2,
            "p_value": reg.pvalue,
            "stderr": reg.stderr,
            "status": "success",
            "error": None,
        }


    def _plot_group_regression(
        self,
        results_df: pd.DataFrame,
        regression_result: dict,
        predictor: str,
        outcome: str,
        out_path: str | Path,
        title: str,
        xlabel: str,
        ylabel: str,
        label_points: bool = True,
    ):
        valid = results_df[
            (results_df["status"] == "success")
            & results_df[predictor].notna()
            & results_df[outcome].notna()
        ].copy()

        valid[predictor] = pd.to_numeric(valid[predictor], errors="coerce")
        valid[outcome] = pd.to_numeric(valid[outcome], errors="coerce")

        valid = valid[
            np.isfinite(valid[predictor])
            & np.isfinite(valid[outcome])
        ].copy()

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 5))

        if len(valid) > 0:
            ax.scatter(
                valid[predictor],
                valid[outcome],
                alpha=0.85,
            )

            if label_points and "stem" in valid.columns:
                for _, row in valid.iterrows():
                    ax.annotate(
                        str(row["stem"]),
                        xy=(row[predictor], row[outcome]),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                    )

        if regression_result["status"] == "success":
            x_min = valid[predictor].min()
            x_max = valid[predictor].max()

            x_line = np.linspace(x_min, x_max, 100)
            y_line = (
                regression_result["intercept"]
                + regression_result["beta_slope"] * x_line
            )

            ax.plot(x_line, y_line)

            annotation = (
                f"slope = {regression_result['beta_slope']:.4g}\n"
                f"p = {self._format_p_value(regression_result['p_value'])}\n"
                f"R² = {regression_result['r_squared']:.3f}\n"
                f"n = {regression_result['n_subjects']}"
            )

        else:
            annotation = (
                "Regression failed\n"
                f"n = {regression_result['n_subjects']}\n"
                f"{regression_result['error']}"
            )

        ax.text(
            0.05,
            0.95,
            annotation,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", alpha=0.15),
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

        return out_path


    def cbt_slope_sleep_onset_to_stage(
        self,
        mat_folder: str | Path,
        edf_folder: str | Path,
        cbt_folder: str | Path,
        out_folder: str | Path,
        stage_mapping: Optional[Dict[Any, int]] = None,
        sleep_codes: Optional[list[int]] = None,
        stage_name: str = 'N3',
        stage_names: Optional[Dict[Any, int]] = None,
        min_temp_points: int = 3,
        save_window_timeseries: bool = True,
        stage_codes_for_predictors: Optional[list[int]] = None,

    ):
        """
        Estimate participant-level CBT/STAGE features, then run group-level regressions.

        Participant-level features
        --------------------------
        1. Sleep onset time.
        2. First stage onset time.
        3. Time from sleep onset to first stage.
        4. First stage bout duration.
        5. CBT slope from sleep onset to first stage onset.
        6. Time from sleep onset to CBT nadir.

        Group-level regressions
        -----------------------
        1. CBT slope ~ stage onset latency.
        2. CBT slope ~ first stage bout duration.
        3. Time to CBT nadir ~ stage onset latency.

        Returns
        -------
        results_df : pd.DataFrame
            Participant-level CBT/stage feature summary.

        data : dict
            Per-participant time series:
                data[stem]["slope_window"]
                data[stem]["nadir_search_window"]

        group_regression_df : pd.DataFrame
            One row per group-level regression.
        """

        if sleep_codes is None:
            sleep_codes = [1, 2, 3, 4]

        if stage_names is None:
            stage_names = {"N1": 1, "N2": 2, "N3": 3, "REM": 4}
        
        if stage_codes_for_predictors is None:
            stage_codes_for_predictors = [1, 2, 3, 4]

        if stage_name not in stage_names:
            raise ValueError(
                f"stage_name must be one of {list(stage_names.keys())}. "
                f"Got: {stage_name}"
            )

        stage_mapping = self._resolve_stage_mapping(stage_mapping)
        stage_code = stage_names[stage_name]

        out_folder = Path(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)

        plot_folder = out_folder / "group_regression_plots"
        plot_folder.mkdir(parents=True, exist_ok=True)

        triplets = self._get_common_file_triplets(
            mat_folder,
            edf_folder,
            cbt_folder,
        )

        results = []
        data = {}

        print(f"Found {len(triplets)} matched .mat/.edf/.xlsx filename sets.")

        for item in triplets:
            stem = item["stem"]
            print(f"\nExtracting CBT/stage features for: {stem}")

            try:
                row, participant_data = self._extract_cbt_stage_features_for_participant(
                    item=item,
                    stage_mapping=stage_mapping,
                    sleep_codes=sleep_codes,
                    stage_code=stage_code,
                    stage_name=stage_name,
                    min_temp_points=min_temp_points,
                )

                _, participant_quadrant_df = self.hours_per_stage_per_quadrant(
                    participant_id=stem,
                    phase=None,
                    stage_codes=stage_codes_for_predictors,
                    sleep_start_sec=row["sleep_onset_sec_from_recording_start"],
                    sleep_end_sec=row["hyp_duration_sec"],
                    return_df=True,
                )

                row = self._add_sleep_stage_quadrant_predictors_to_row(
                    row=row,
                    participant_df=participant_quadrant_df,
                )

                results.append(row)

                participant_data["stage_quadrant_predictors"] = participant_quadrant_df
                data[stem] = participant_data

                if save_window_timeseries:
                    subj_out = out_folder / stem
                    subj_out.mkdir(parents=True, exist_ok=True)

                    participant_data["slope_window"].to_csv(
                        subj_out / f"{stem}_cbt_sleep_onset_to_{stage_name}_timeseries.csv",
                        index=False,
                    )

                    participant_data["nadir_search_window"].to_csv(
                        subj_out / f"{stem}_cbt_sleep_onset_to_nadir_search_window.csv",
                        index=False,
                    )

                    participant_data["stage_quadrant_predictors"].to_csv(
                        subj_out / f"{stem}_stage_quadrant_predictors.csv",
                        index=False,
                    )

                print(
                    f"✅ Success for {stem}: "
                    f"sleep onset={row['sleep_onset_min_from_recording_start']:.2f} min, "
                    f"{stage_name} onset={row[f'{stage_name}_onset_min_from_recording_start']:.2f} min, "
                    f"time to {stage_name}={row[f'minutes_sleep_onset_to_{stage_name}']:.2f} min, "
                    f"first {stage_name} bout={row[f'first_{stage_name}_bout_duration_min']:.2f} min, "
                    f"time to nadir={row['minutes_sleep_onset_to_cbt_nadir']:.2f} min, "
                    f"CBT slope={row['cbt_slope_degC_per_hour']:.4f} °C/hr"
                )

            except Exception as exc:
                results.append(
                    self._failed_cbt_stage_participant_row(
                        item=item,
                        error=exc,
                        stage_name=stage_name
                    )
                )

                print(f"❌ Failed for {stem}: {exc}")

        results_df = pd.DataFrame(results)

        regression_specs = [
            {
                f"model_name": f"cbt_slope_vs_{stage_name}_onset_latency",
                f"predictor": f"cbt_slope_degC_per_hour",
                f"outcome": f"minutes_sleep_onset_to_{stage_name}",
                f"title": f"CBT slope vs {stage_name} onset latency",
                f"xlabel": f"CBT slope from sleep onset to {stage_name} (°C/hour)",
                f"ylabel": f"Minutes from sleep onset to first {stage_name}",
                f"plot_file": f"cbt_slope_vs_{stage_name}_onset_latency.png",
            },
            {
                f"model_name": f"cbt_slope_vs_first_{stage_name}_bout_duration",
                f"predictor": f"cbt_slope_degC_per_hour",
                f"outcome": f"first_{stage_name}_bout_duration_min",
                f"title": f"CBT slope vs first {stage_name} bout duration",
                f"xlabel": f"CBT slope from sleep onset to {stage_name} (°C/hour)",
                f"ylabel": f"First {stage_name} bout duration (minutes)",
                f"plot_file": f"cbt_slope_vs_first_{stage_name}_bout_duration.png",
            },
            {
                f"model_name": f"time_to_nadir_vs_{stage_name}_onset_latency",
                f"predictor": f"minutes_sleep_onset_to_{stage_name}",
                f"outcome": f"minutes_sleep_onset_to_cbt_nadir",
                f"title": f"Time to CBT nadir vs {stage_name} onset latency",
                f"xlabel": f"Minutes from sleep onset to first {stage_name}",
                f"ylabel": f"Minutes from sleep onset to CBT nadir",
                f"plot_file": f"time_to_cbt_nadir_vs_{stage_name}_onset_latency.png",
            },
        ]

        stage_predictor_cols = [
            col
            for col in results_df.columns
            if (
                col.startswith("percent_")
                or col.startswith("total_")
            )
            and pd.api.types.is_numeric_dtype(results_df[col])
        ]

        for predictor in stage_predictor_cols:
            regression_specs.append(
                {
                    "model_name": f"cbt_nadir_rate_vs_{predictor}",
                    "predictor": predictor,
                    "outcome": "cbt_rate_sleep_onset_to_nadir_degC_per_hour",
                    "title": f"CBT nadir rate vs {predictor}",
                    "xlabel": predictor,
                    "ylabel": "CBT rate from sleep onset to nadir (°C/hour)",
                    "plot_file": f"cbt_nadir_rate_vs_{predictor}.png",
                }
            )

        regression_rows = []

        for spec in regression_specs:
            reg_result = self._fit_group_regression(
                results_df=results_df,
                predictor=spec["predictor"],
                outcome=spec["outcome"],
                model_name=spec["model_name"],
            )

            regression_rows.append(reg_result)

            plot_path = self._plot_group_regression(
                results_df=results_df,
                regression_result=reg_result,
                predictor=spec["predictor"],
                outcome=spec["outcome"],
                out_path=plot_folder / spec["plot_file"],
                title=spec["title"],
                xlabel=spec["xlabel"],
                ylabel=spec["ylabel"],
            )

            if reg_result["status"] == "success":
                print(
                    f"\nGroup-level regression: {spec['model_name']}"
                    f"\n  {spec['outcome']} ~ {spec['predictor']}"
                    f"\n  beta={reg_result['beta_slope']:.6f}"
                    f"\n  r={reg_result['r_value']:.4f}"
                    f"\n  R²={reg_result['r_squared']:.4f}"
                    f"\n  p={reg_result['p_value']:.6g}"
                    f"\n  n={reg_result['n_subjects']}"
                    f"\n  plot={plot_path}"
                )
            else:
                print(
                    f"\nGroup-level regression failed: {spec['model_name']}"
                    f"\n  Reason: {reg_result['error']}"
                    f"\n  plot={plot_path}"
                )

        group_regression_df = pd.DataFrame(regression_rows)

        participant_csv_path = (
            out_folder / f"cbt_{stage_name}_nadir_participant_summary.csv"
        )

        group_csv_path = (
            out_folder / f"cbt_{stage_name}_nadir_group_regression_summary.csv"
        )

        xlsx_path = (
            out_folder / f"cbt_{stage_name}_nadir_analysis.xlsx"
        )

        results_df.to_csv(participant_csv_path, index=False)
        group_regression_df.to_csv(group_csv_path, index=False)

        with pd.ExcelWriter(xlsx_path) as writer:
            results_df.to_excel(
                writer,
                sheet_name="participant_summary",
                index=False,
            )

            group_regression_df.to_excel(
                writer,
                sheet_name="group_regressions",
                index=False,
            )

        print(f"\nSaved participant summary CSV to: {participant_csv_path}")
        print(f"Saved group regression CSV to: {group_csv_path}")
        print(f"Saved Excel workbook to: {xlsx_path}")
        print(f"Saved regression plots to: {plot_folder}")

        return results_df, data, group_regression_df