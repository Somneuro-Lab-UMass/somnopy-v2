import pytest

from somnopy.preprocessing import load_hypnogram_data


@pytest.fixture
def dummy_data():
    # Setup your dummy EEG data and sampling frequency.
    return {
        'eeg': [0.1, 0.5, 0.3, 0.7],  # Replace with realistic dummy data.
        'sf': 100
    }

def test_detect_spindles(dummy_data):
    result = load_hypnogram_data(dummy_data['eeg'], sf=dummy_data['sf'])
    assert result is not None