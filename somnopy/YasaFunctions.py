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
from datetime import datetime, timedelta


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

        temp_plot_df = self.cbt

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
    
    
    def hours_per_stage_per_quadrant(
        self,
        participant_id: Optional[str] = None,
        phase: Optional[str] = None,
        stage_codes: Optional[list[int]] = None,
        quadrant_labels: Optional[list[str]] = None,
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


        hypno = np.asarray(self.hypno).astype(int)
        quads = np.array_split(hypno, 4)

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

    def plot_stage_hours_by_quadrant(
        self,
        phase_folders: Optional[Dict[str, str | Path]] = None,
        follicular_folder: Optional[str | Path] = None,
        luteal_folder: Optional[str | Path] = None,
        save_dir: Optional[str | Path] = None,
        stage_mapping: Optional[Dict[Any, int]] = None,
        stage_codes: Optional[list[int]] = None,
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
            YASA stage codes to include. Default is [1, 2, 3, 4].
            Use [0, 1, 2, 3, 4] if you also want Wake.

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
        status_rows = []

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

                    _, participant_df = self.hours_per_stage_per_quadrant(
                        participant_id=stem,
                        phase=phase,
                        stage_codes=stage_codes,
                        return_df=True,
                    )

                    participant_dfs.append(participant_df)

                    status_rows.append(
                        {
                            "phase": phase,
                            "participant": stem,
                            "edf_file": str(item["edf"]),
                            "mat_file": str(item["mat"]),
                            "status": "success",
                            "error": None,
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
            "status_df": status_df,
            "group_figs": group_figs,
            "participant_figs": participant_figs,
        }

        return results

            