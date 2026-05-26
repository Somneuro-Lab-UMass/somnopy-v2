# Sleep Spindle-Slow Oscillation Coupling Analysis

# NEW README

# Introduction

This Git branch is a revision and reimplementation of Somnopy 0.0.9 by Ryan Zaid. Main changes include integration of swa into spso functions, spectrogram functionality, and jupyter notebook usability. The goal of this revision is to make somnopy more easily accessible and usable by everyone in the lab, including those without computer science backgrounds. Simplification of code and setup is therefore a core part of this branch.

# Setup

NOTE: setup only needs to be performed once per computer, if pipeline has already been set up on your computer, skip to Usage section.

Before beginning setup, ensure that the full version of anaconda is installed on your computer (not miniconda). Anaconda is a package managing system and is essential for ensuring proper version control within package dependencies. Without the use of anaconda, installation of somnopy may cause errors in both other python scripts and the spindle coupling notebook provided. For anaconda installation instructions, visit this link: https://www.anaconda.com/docs/getting-started/anaconda/install/overview

To download the latest version of the Somnopy package and pipeline, you also need to install git. This will allow for easy access to the latest changes in the somnopy pipeline. For git installation instructions, visit this link: https://git-scm.com/install/windows


1. To begin setup, type in "anaconda prompt" into the windows search tool at the bottom of the screen. Select the Anaconda Prompt application to open it.
2. Ensure git is working properly: type "git" into the terminal. You should see a message that begins with "usage: git" pop up. 
3. Navigate to the folder you would like to set up your somnopy project in. To view the contents of your current folder, type "dir". To navigate to a subfolder within your current folder, type "cd name_of_folder". To backtrack to your previous folder, type "cd ..". To create a new folder, type "mkdir name_of_folder". 
4. Type "git clone -b ryan --single-branch https://github.com/Somneuro-Lab-UMass/somnopy-v2". This will download the somnopy package and pipeline to the folder you are currently in.
5. Type "cd somnopy-v2". This will navigate you to the projects root directory.
6. Type "conda env create -f environment.yml". This will install the necessary dependencies for somnopy and setup somnopy itself.
7. Done!


# Usage

Now that the Somnopy package has been set up, let's look at how to run it. 

1. First, open Visual Studio Code by typing Visual Studio Code into the windows search bar at the bottom of the screen.
2. At the top of VSCode, click File, and then click Open Folder.
3. Open somnopy-v2, where it was stored when cloning the github repository.
4. On the left hand side of the screen, hover over the top icon that looks like a notebook page. It should say "Explorer". Click this icon. This should show you all of the different files contained in the somnopy repo.
5. Click and open spso_pipeline.ipynb. 
6. Click the button at the top right that says Select Kernel. If this button says somnopy, skip this step. A drop down menu will appear. Select Python Environments and then somnopy.
7. Run each cell within the notebook by clicking the play button to the left of the cells. 
8. Enjoy!!!





# OLD README

## Overview
This package provides tools for analyzing sleep EEG data, focusing on **Sleep Spindles (SPs) and Slow Oscillations (SOs)** and their **coupling interactions**. The package includes functionalities for:
- **EEG Preprocessing**
- **Slow Oscillation (SO) Detection**
- **Sleep Spindle (SP) Detection**
- **Slow Wave Activity (SWA) Quantification**
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

#toml not requirements.txt
#no need for install, just run the first cell in SPSO_Coupling_Notebook.ipynb
#code is not up to date. This is 0.0.9, we are on v 0.0.10

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
- **Ryan Zaid**

## Acknowledgments
This package integrates various methods from published research on sleep spindles and slow oscillations.

## Issues & Support
For bug reports and feature requests, please open an issue on GitHub.

---
📌 **Want to contribute?** Feel free to submit a pull request! 🚀

