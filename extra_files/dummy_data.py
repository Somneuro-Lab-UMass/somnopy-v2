from pathlib import Path
import re
from datetime import datetime, timedelta, time
import pandas as pd
import mne


# =========================
# CONFIG
# =========================
ROOT_DIR = Path(r"Y:\SNL\MBHD\somnopy_data")   # folder of folders containing EDFs
EXCEL_PATH = Path(r"Y:\SNL\MBHD\Sleep Scoring Log.csv")# set to a specific .xlsx path if you want
OUTPUT_SAME_FOLDER = True                 # save next to each EDF
DEFAULT_LOCATION = "EEG-F4"               # used if EDF channel label is unavailable
EPOCH_SEC = 30.0


# =========================
# HELPERS
# =========================
def find_excel_file(root_dir: Path) -> Path:
    """
    Find the first Excel file in the root folder or its immediate subfolders.
    """
    excel_exts = {".xlsx", ".xls", ".xlsm"}
    
    # first check root
    for p in root_dir.iterdir():
        if p.is_file() and p.suffix.lower() in excel_exts:
            return p

    # then check first-level subfolders
    for sub in root_dir.iterdir():
        if sub.is_dir():
            for p in sub.iterdir():
                if p.is_file() and p.suffix.lower() in excel_exts:
                    return p

    raise FileNotFoundError("No Excel file found in root folder or first-level subfolders.")


def normalize_id(val) -> str:
    """
    Normalize IDs so '0101', 101, and '101.0' can match.
    """
    if pd.isna(val):
        return ""
    s = str(val).strip()

    # remove trailing .0 from Excel numeric-looking IDs
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]

    # keep digits only for matching
    digits = re.sub(r"\D", "", s)
    return digits if digits else s


def extract_id_from_filename(filename: str) -> str:
    """
    Extract the most likely participant ID from an EDF filename.
    Example:
        EmoCuing_0101.edf -> 0101
    """
    stem = Path(filename).stem

    # take the last digit group in the filename
    matches = re.findall(r"\d+", stem)
    if not matches:
        return ""

    return matches[-1]


def parse_excel_time(val) -> time:
    """
    Parse sleep start/end values from Excel.
    Supports:
    - Excel datetime/time cells
    - strings like '10:35 PM', '22:35', '22:35:00'
    """
    if pd.isna(val):
        return None

    if isinstance(val, datetime):
        return val.time()

    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime().time()

    if isinstance(val, time):
        return val

    s = str(val).strip()

    fmts = [
        "%I:%M:%S %p",
        "%I:%M %p",
        "%H:%M:%S",
        "%H:%M",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            continue

    raise ValueError(f"Could not parse time value: {val}")


def combine_with_recording_date(recording_start: datetime, t: time) -> datetime:
    """
    Combine a time-of-day with the EDF recording date.
    """
    return datetime.combine(recording_start.date(), t)


def resolve_sleep_interval(recording_start: datetime, recording_end: datetime,
                           sleep_start_t: time, sleep_end_t: time):
    """
    Convert sleep start/end times-of-day into datetimes aligned to the EDF recording.
    Handles overnight sleep intervals.
    """
    sleep_start_dt = combine_with_recording_date(recording_start, sleep_start_t)
    sleep_end_dt = combine_with_recording_date(recording_start, sleep_end_t)

    # If end is earlier than start, assume it crossed midnight
    if sleep_end_dt <= sleep_start_dt:
        sleep_end_dt += timedelta(days=1)

    # If recording itself crossed midnight and sleep times seem to belong to next day
    if sleep_start_dt < recording_start - timedelta(hours=12):
        sleep_start_dt += timedelta(days=1)
        sleep_end_dt += timedelta(days=1)

    return sleep_start_dt, sleep_end_dt


def format_remlogic_time(dt: datetime) -> str:
    """
    Format like: 12:30:05 PM.000
    """
    ms = int(dt.microsecond / 1000)
    return dt.strftime("%I:%M:%S %p") + f".{ms:03d}"


def choose_location(raw) -> str:
    """
    Try to use the first channel label from the EDF, otherwise default.
    """
    try:
        if raw.ch_names:
            return raw.ch_names[0]
    except Exception:
        pass
    return DEFAULT_LOCATION


def add_event_row(rows, stage, event_time, event_name, duration, location):
    rows.append({
        "Sleep Stage": stage,
        "Time [hh:mm:ss.xxx]": format_remlogic_time(event_time),
        "Event": event_name,
        "Duration[s]": f"{duration:.2f}",
        "Location": location
    })


def build_hypnogram_rows(recording_start: datetime,
                         recording_end: datetime,
                         sleep_start_dt: datetime,
                         sleep_end_dt: datetime,
                         location: str):
    rows = []

    # Event rows
    add_event_row(rows, "W", recording_start, "Analysis Start", 0.00, location)
    add_event_row(rows, "W", sleep_start_dt, "Lights Off", 0.00, location)

    # 30-second epoch rows
    t = recording_start
    while t < recording_end:
        stage = "N1" if (sleep_start_dt <= t < sleep_end_dt) else "W"
        add_event_row(rows, stage, t, stage, EPOCH_SEC, location)
        t += timedelta(seconds=EPOCH_SEC)

    add_event_row(rows, "W", sleep_end_dt, "Lights On", 0.00, location)
    add_event_row(rows, "W", recording_end, "Analysis Stop", 0.00, location)

    # sort by actual datetime reconstructed from formatted rows is annoying,
    # so instead rebuild with stored datetimes if needed. Since we append in order,
    # only Lights On can be out of order if outside recording; keep it only if within range.
    cleaned_rows = []

    for row in rows:
        cleaned_rows.append(row)

    return cleaned_rows


def load_sleep_table(excel_path: Path) -> pd.DataFrame:
    df = pd.read_csv(excel_path)

    # clean column names (Excel CSVs sometimes add spaces)
    df.columns = [c.strip() for c in df.columns]

    required = ["ID #", "Sleep Start Time", "Sleep End Time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.copy()
    df["ID_norm"] = df["ID #"].apply(normalize_id)
    df["Sleep Start Parsed"] = df["Sleep Start Time"].apply(parse_excel_time)
    df["Sleep End Parsed"] = df["Sleep End Time"].apply(parse_excel_time)

    return df


def match_row_for_edf(df: pd.DataFrame, edf_path: Path):
    file_id = normalize_id(extract_id_from_filename(edf_path.name))
    if not file_id:
        return None

    # exact match first
    matches = df[df["ID_norm"] == file_id]
    if len(matches) == 1:
        return matches.iloc[0]

    # fallback: compare after stripping leading zeros
    file_id_nozero = file_id.lstrip("0") or "0"
    matches = df[df["ID_norm"].apply(lambda x: (x.lstrip("0") or "0") == file_id_nozero)]
    if len(matches) == 1:
        return matches.iloc[0]

    return None


def process_one_edf(edf_path: Path, sleep_df: pd.DataFrame):
    row = match_row_for_edf(sleep_df, edf_path)
    if row is None:
        print(f"[SKIP] No matching ID found for {edf_path.name}")
        return

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

        meas_date = raw.info["meas_date"]
        if meas_date is None:
            raise ValueError("EDF has no measurement start date/time in header.")

        # convert to naive datetime if timezone-aware
        if hasattr(meas_date, "tzinfo") and meas_date.tzinfo is not None:
            recording_start = meas_date.replace(tzinfo=None)
        else:
            recording_start = meas_date

        duration_sec = raw.n_times / raw.info["sfreq"]
        recording_end = recording_start + timedelta(seconds=duration_sec)

        sleep_start_t = row["Sleep Start Parsed"]
        sleep_end_t = row["Sleep End Parsed"]

        sleep_start_dt, sleep_end_dt = resolve_sleep_interval(
            recording_start, recording_end, sleep_start_t, sleep_end_t
        )

        location = choose_location(raw)

        rows = build_hypnogram_rows(
            recording_start=recording_start,
            recording_end=recording_end,
            sleep_start_dt=sleep_start_dt,
            sleep_end_dt=sleep_end_dt,
            location=location,
        )

        out_name = edf_path.stem + ".txt"
        out_path = edf_path.with_name(out_name) if OUTPUT_SAME_FOLDER else ROOT_DIR / out_name

        out_df = pd.DataFrame(rows, columns=[
            "Sleep Stage",
            "Time [hh:mm:ss.xxx]",
            "Event",
            "Duration[s]",
            "Location"
        ])

        out_df.to_csv(out_path, sep="\t", index=False)
        print(f"[OK] Wrote {out_path}")

    except Exception as e:
        print(f"[ERROR] Could not process {edf_path.name}: {e}")


def main():
    root_dir = Path(ROOT_DIR)

    excel_path = Path(EXCEL_PATH) if EXCEL_PATH else find_excel_file(root_dir)
    print(f"Using Excel file: {excel_path}")

    sleep_df = load_sleep_table(excel_path)

    edf_files = list(root_dir.rglob("*.edf"))
    if not edf_files:
        print("No EDF files found.")
        return

    print(f"Found {len(edf_files)} EDF files.")

    for edf_path in edf_files:
        process_one_edf(edf_path, sleep_df)


if __name__ == "__main__":
    main()