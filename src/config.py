import os

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Ensure results folder exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================================
# DATASET PATH
# ==========================================================
# We assume your file is named 'cloud_data.txt'.
# If you named it differently, change 'cloud_data.txt' below.
DATASET_PATH = os.path.join(DATA_DIR, 'cloud_dataset.csv')

# System Simulation Constants
BASE_POWER_WATTS = 100.0
CPU_POWER_COEFF = 50.0
IDLE_THRESHOLD = 0.2
HIGH_THRESHOLD = 0.8
