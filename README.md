# Sleep Spindle-Slow Oscillation Coupling Analysis

## Overview
This package provides tools for analyzing sleep EEG data, focusing on **Sleep Spindles (SPs) and Slow Oscillations (SOs)** and their **coupling interactions**. The package includes functionalities for:
- **EEG Preprocessing**
- **Slow Oscillation (SO) Detection**
- **Sleep Spindle (SP) Detection**
- **Phase-Amplitude Coupling (PAC) Analysis**
- **Peri-Event Time Histogram (PETH) Analysis**
- **Data Visualization for SOs and SPs**

## Features
- **Multi-method SO and SP Detection**: Choose from various published detection methods.
- **Coupling Analysis**: Computes phase-amplitude coupling between SOs and SPs.
- **Batch Processing**: Analyze multiple EEG recordings efficiently.
- **Interactive and Static Plotting**: Visualize SO and SP events using topomaps and time-series plots.

## Installation
Ensure you have Python installed. Then, install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

### 1️⃣ Load EEG & Hypnogram Data
```python
from polysomnography import PolySomnoGraphy

psg = PolySomnoGraphy(
    eeg_path="subject1.edf",
    hypnogram_path="subject1.txt",
    hypnogram_type="RemLogic"
)

# Access raw EEG data
raw = psg.get_raw()
# Access hypnogram data
hypnogram = psg.get_hypnogram()
```

### 2️⃣ Detect Sleep Spindles & Slow Oscillations
```python
so_results = psg.detect_slow_oscillations(method="Staresina")
spindle_results = psg.detect_spindles(method="Hahn2020")
```

### 3️⃣ Compute Phase-Amplitude Coupling (PAC)
```python
pac_results = psg.pac()
```

### 4️⃣ Batch Processing for Multiple EEG Files
```python
from somno import get_sosp_for_folder

event_summary, coupling_events, so_waveforms = get_sosp_for_folder(
    raw_folder="EEG_data",
    stage_folder="Hypnogram_data"
)
```

### 5️⃣ Visualizations
```python
from metrics import plot_SO, plot_SP

# Plot Slow Oscillation Events
plot_SO(so_results, raw)

# Plot Spindle Events
plot_SP(spindle_results, raw)
```

## Supported File Formats
- **EEG Files:** `.edf`, `.vhdr`, `.set`, `.fif`, `.bdf`, `.cnt`
- **Hypnogram Files:**
  - **RemLogic** (`.txt`)
  - **Hume** (`.mat`)

## License
MIT License

## Contributors
- **Roger Balcells Sanchez**
- **Thea Ng**
- **Atif Abedeen**
- **Lindsey Mooney**

## Acknowledgments
This package integrates various methods from published research on sleep spindles and slow oscillations.

## Issues & Support
For bug reports and feature requests, please open an issue on GitHub.

---
📌 **Want to contribute?** Feel free to submit a pull request! 🚀

