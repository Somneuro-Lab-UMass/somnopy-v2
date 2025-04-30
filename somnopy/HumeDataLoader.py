import numpy as np
import pandas as pd
from scipy.io import loadmat


class HumeDataLoader:
    """
    Loader for hypnogram stage data stored in MATLAB (.mat) files.

    This class reads a .mat file containing a nested 'stageData' structure,
    extracts and concatenates sleep stage annotations, applies remapping
    rules, and exposes the result as a pandas DataFrame or NumPy array.

    Parameters
    ----------
    filepath : str
        Path to the .mat file containing the hypnogram. The file must
        include a 'stageData' field with a nested 'stages' array.

    Attributes
    ----------
    filepath : str
        The path provided at initialization.
    df : pandas.DataFrame or None
        DataFrame of processed stage values under the column 'stages'.
    """
    def __init__(self, filepath):
        """
        Initialize the loader and immediately load the file.

        Parameters
        ----------
        filepath : str
            Path to the MAT file containing hypnogram data.
        """
        self.filepath = filepath
        self.df = None  # This will hold a pandas DataFrame.
        self.load_file()

    def load_file(self):
        """
        Load and process hypnogram stages from the MAT file.

        The method:
          1. Loads the .mat contents.
          2. Extracts the nested 'stageData' → 'stages' array.
          3. Concatenates all subarrays into one flat 1D array.
          4. Remaps:
             - All values == 5 → 4
             - All values == 7 → 0
          5. Forces the last element to 0.
          6. Casts the entire series to integer.
          7. Stores the result in `self.df` with a single column 'stages'.

        Raises
        ------
        KeyError
            If 'stageData' or 'stages' are missing in the .mat file.
        """
        # Load the MAT file.
        scoring = loadmat(self.filepath)
        # Extract stages from the nested structure.
        stages = scoring['stageData']['stages']
        # Concatenate nested arrays into a 1D array.
        stages = np.concatenate([np.concatenate(stage) for stage in stages])

        # Create a DataFrame from the stages.
        stages_df = pd.DataFrame(stages, columns=['stages'])

        # Apply the remapping:
        # Replace stages equal to 5 with '4' and stages equal to 7 with '0'.
        stages[stages == 5] = '4'
        stages[stages == 7] = '0'
        # Set the last stage to 0.
        stages_df.iloc[-1] = 0

        # Update the DataFrame column and cast to integer.
        stages_df['stages'] = stages_df['stages'].astype(int)
        self.df = stages_df

    def get_data(self):
        """
        Retrieve the processed hypnogram stages as a NumPy array.

        Returns
        -------
        numpy.ndarray
            Flattened array of integer stage values, length == number of epochs.
        """
        return self.df['stages'].values.flatten()
