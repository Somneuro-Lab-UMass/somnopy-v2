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

        self.stage_mapping = stage_mapping
        self.input_data = None
        self.channels = None
        self.hypno = None
        self.sampling_rate = None
        self.hyp_window = None
        self.raw = None

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

        hypno = yasa.hypno_upsample_to_data(
            hypno=hyp_yasa,
            sf_hypno=1 / hyp_window,
            data=raw_data[0],
            sf_data=sampling_rate,
        )

        recording_start = pd.to_datetime(raw.info["meas_date"]).tz_localize(None)


        #store class attributes
        self.recording_start = recording_start
        self.input_data = input_data
        self.channels = channels
        self.hypno = hypno
        self.sampling_rate = sampling_rate
        self.hyp_window = hyp_window
        self.raw = raw

        return input_data, channels, hypno, recording_start, sampling_rate, hyp_window, raw

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
        temp_df: pd.DataFrame,
        verbose: bool = False,
        outpath: Optional[str | Path] = None,
        filename: str = "participant",
        fmin: float = 0.5,
        fmax: float = 25,
        recording_start: Optional[str | pd.Timestamp] = None,
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

        temp_plot_df = self._prepare_temp_df(temp_df)

        if recording_start is None:
            recording_start = temp_plot_df["datetime"].iloc[0]
        else:
            recording_start = pd.to_datetime(recording_start)

        temp_plot_df["hours_from_start"] = (
            temp_plot_df["datetime"] - recording_start
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

    @staticmethod
    def load_cbt(filepath: str | Path) -> pd.DataFrame:
        """Load a core body temperature Excel file and standardize column names."""
        df = pd.read_excel(filepath).rename(
            {
                "Date(mm/dd/yyyy)": "date",
                "Hour": "hour",
                "Temperature": "temp",
            },
            axis=1,
        )
        return df

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

                recording_start = self.raw.info.get("meas_date", None)
                if recording_start is not None:
                    recording_start = self._strip_tz(recording_start)

                file_out = out_folder / stem
                file_out.mkdir(parents=True, exist_ok=True)

                fig, ax_temp, temp_plot_df = self.plot_single_electrode_with_temp(
                    electrode=electrode,
                    temp_df=temp_df,
                    verbose=verbose,
                    outpath=file_out,
                    filename=stem,
                    fmin=fmin,
                    fmax=fmax,
                    recording_start=recording_start,
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

                recording_start = self.raw.info.get("meas_date", None)
                if recording_start is None:
                    raise ValueError("EDF recording start time (raw.info['meas_date']) is missing.")
                recording_start = self._strip_tz(recording_start)

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
