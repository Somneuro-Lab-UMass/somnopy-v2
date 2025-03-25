import numpy as np
import pandas as pd
from scipy.io import loadmat


class HumeDataLoader:
    def __init__(self, filepath):
        """
        Parameters:
            filepath (str): Path to the MAT file containing hypnogram data.
        """
        self.filepath = filepath
        self.df = None  # This will hold a pandas DataFrame.
        self.load_file()

    def load_file(self):
        """
        Loads the hypnogram data from a MAT file.

        The file is expected to have a field 'stageData' containing 'stages'.
        The stages are concatenated from nested arrays, then processed:
          - Any stage value equal to 5 is replaced with '4'.
          - Any stage value equal to 7 is replaced with '0'.
          - The last stage is set to 0.
          - All values are cast to int.

        The processed stages are stored in a pandas DataFrame.
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
        Returns:
            numpy.ndarray: Flattened array of hypnogram stage values.
        """
        return self.df['stages'].values.flatten()
