import numpy as np
import pandas as pd


class RemLogicDataLoader:
    """
    Loader for REMLogic hypnogram text files.

    Reads a REMLogic-generated text file with sleep stage annotations, parses
    each row to extract sleep stage, timestamp, event description, duration,
    and location, and provides the stages sequence as a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the REMLogic text file to parse.
    skip_header : bool, default=False
        If True, finds the last line containing 'Sleep Stage' and skips all
        lines up to and including that header. Otherwise, no header skipping.

    Attributes
    ----------
    filepath : str
        Input file path.
    skip_header : bool
        Header skipping flag.
    df : pandas.DataFrame
        DataFrame of parsed rows with columns:
          - 'sleep_stage' : int (0-5 or -1 for unknown)
          - 'time'        : float, seconds since midnight
          - 'event'       : str, event description text
          - 'duration'    : float, event duration (s)
          - 'location'    : str, event location label
    """
    def __init__(self, filepath: str, skip_header: bool = False) -> None:
        """
        Initialize loader and parse the file immediately.

        Parameters
        ----------
        filepath : str
            Path to the REMLogic text file.
        skip_header : bool, default=False
            If True, skip header lines up to the last 'Sleep Stage' occurrence.
        """
        self.filepath: str = filepath
        self.skip_header: bool = skip_header
        self.df: pd.DataFrame = pd.DataFrame()  # Stores parsed data
        self.load_file()

    def parse_sleep_stage(self, stage_str: str) -> int:
        """
        Convert a sleep-stage token to an integer label (0-5).

        Parameters
        ----------
        stage_str : str
            Sleep stage string (e.g., 'N2', 'REM', 'W', or numeric text).

        Returns
        -------
        int
            Stage code: 0=Wake, 1=N1, 2=N2, 3=N3, 4=N4, 5=REM, -1=unknown.
        """
        stage_str = stage_str.strip()
        try:
            return int(stage_str)
        except ValueError:
            mapping = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 5, "R": 5,
                       "SLEEP-REM": 5, "SLEEP-RM": 5, "SLEEP-S0": 0, "SLEEP-S1": 1,
                       "SLEEP-S2": 2, "SLEEP-S3": 3, "SLEEP-S4": 4}
            return mapping.get(stage_str.upper(), -1)

    def find_time_index(self, tokens: list) -> int:
        """
        Identify the index of the time column in a list of tokens.

        Scans for the first token containing ':' and at least two ':'-separated parts.

        Parameters
        ----------
        tokens : list of str
            Split tokens from a line.

        Returns
        -------
        int
            Index of time token, or -1 if not found.
        """
        for i, token in enumerate(tokens):
            if ":" in token and len(token.split(":")) >= 2:
                return i  # Time column index
        return -1  # Return -1 if time is not found (unlikely case)

    def parse_time(self, tokens: list, time_index: int) -> (float, int):
        """
        Parse a time token (and optional AM/PM) to seconds since midnight.

        Parameters
        ----------
        tokens : list of str
            Tokens containing time and possibly an AM/PM indicator.
        time_index : int
            Index where the time string appears.

        Returns
        -------
        total_seconds : float
            Time converted to seconds from midnight.
        next_index : int
            Token index immediately after the time (and AM/PM) tokens.
        """
        time_str = tokens[time_index]
        token_index_after_time = time_index + 1
        extra_sec = 0

        if len(tokens) > time_index + 1 and (
                tokens[time_index + 1].upper().startswith("AM") or tokens[time_index + 1].upper().startswith("PM")):
            indicator_token = tokens[time_index + 1]
            token_index_after_time = time_index + 2

            if '.' in indicator_token:
                ampm, _ = indicator_token.split('.', 1)
            else:
                ampm = indicator_token

            try:
                parts = time_str.split(':')
                hour, minute, second = int(parts[0]), int(parts[1]), int(parts[2])
            except Exception:
                return np.nan, token_index_after_time

            if ampm.upper() == "PM" and hour < 12:
                hour += 12
            total_seconds = hour * 3600 + minute * 60 + second + extra_sec
            return total_seconds, token_index_after_time
        else:
            try:
                parts = time_str.split(':')
                hour, minute, second = int(parts[0]), int(parts[1]), int(parts[2])
            except Exception:
                return np.nan, time_index + 1
            total_seconds = hour * 3600 + minute * 60 + second
            return total_seconds, time_index + 1

    def load_file(self) -> None:
        """
        Read and parse the REMLogic text file.

        Steps:
          1. Read all lines, skip headers if requested.
          2. For each nonblank line:
             a. Split into tokens and ensure at least four tokens.
             b. Find and isolate the time column, remove position tokens.
             c. Extract sleep stage token, parse it to int.
             d. Parse time to seconds, extract duration and location.
             e. Concatenate remaining tokens as event text.
             f. Only keep rows where sleep-stage token appears in event text.
          3. Store valid rows in `self.df` DataFrame.
        """
        with open(self.filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # If skip_header is True, find the last "Sleep Stage" line and skip it
        if self.skip_header:
            last_header_index = -1
            for idx, line in enumerate(lines):
                if "sleep stage" in line.lower():
                    last_header_index = idx
            if last_header_index != -1:
                lines = lines[last_header_index + 1:]

        parsed_rows = []

        for line in lines:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            tokens = line.split()
            if len(tokens) < 4:
                print(f"Skipping malformed line: {line}")
                continue

            # Detect time column index
            time_index = self.find_time_index(tokens)
            if time_index == -1:
                print(f"Skipping line (no valid time found): {line}")
                continue

            # Remove everything between Sleep Stage and Time (i.e., Position Column)
            tokens = [tokens[0]] + tokens[time_index:]

            # Sleep stage is always first
            sleep_stage_token = tokens[0]

            # Parse time
            time_sec, time_token_index = self.parse_time(tokens, 1)

            # Extract duration and location (last two tokens)
            try:
                duration = float(tokens[-2])
            except ValueError:
                print(f"Error parsing duration, defaulting to NaN: {line}")
                duration = np.nan
            location = tokens[-1]

            # Extract event text
            if len(tokens) - 2 > time_token_index:
                event = " ".join(tokens[time_token_index:-2])
            else:
                event = ""

            # Ensure Sleep Stage appears in Event text (bad row removal)
            if sleep_stage_token.upper() not in event.upper():
                continue

            # Convert Sleep Stage
            sleep_stage = self.parse_sleep_stage(sleep_stage_token)

            # Store parsed row
            parsed_rows.append({
                'sleep_stage': sleep_stage,
                'time': time_sec,
                'event': event,
                'duration': duration,
                'location': location
            })

        # ts = parsed_rows[-1]['time']+parsed_rows[-1]['duration']
        # #Add end value
        # # parsed_rows.append({
        # #     'sleep_stage': 0,
        # #     'time': ts,
        # #     'event': 0,
        # #     'duration': 0,
        # #     'location': 'end'
        # # })

        # Store parsed data as a DataFrame
        self.df = pd.DataFrame(parsed_rows)

    def get_data(self) -> pd.DataFrame:
        """
        Return hypnogram stages as a DataFrame with a 'stages' column.

        Remaps stage codes: 5→4, 7→0 for compatibility.

        Returns
        -------
        pandas.DataFrame
            DataFrame with single column 'stages'.
        """
        df_stages = self.df[['sleep_stage']].copy()
        df_stages.rename(columns={'sleep_stage': 'stages'}, inplace=True)
        df_stages.loc[df_stages['stages'] == 5, 'stages'] = 4
        df_stages.loc[df_stages['stages'] == 7, 'stages'] = 0
        return df_stages


# =============================================================================
# Example usage:
# =============================================================================
if __name__ == '__main__':
    loader = RemLogicDataLoader(
        r"C:\Users\roger\PycharmProjects\PSG-Party\data\Scored files to be analyzed\ITNS_102_W2_PM1_SD.txt",
        skip_header=True
    )
    data_df = loader.get_data()
    print(data_df)
    print(loader.df)
