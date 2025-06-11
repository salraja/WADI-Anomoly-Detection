# WADI Anomaly Detection Bundle

This repository contains everything you need to run the LSTM Autoencoder baseline on the WADI dataset. Download and unzip the `Research.zip` archive to get both the training script and the data in one go.

## What’s inside `Research.zip`

- `train_lstm.py`  
- `WADI_14days.csv`  
- `WADI_attackdata.csv`  
- `attack_description.xlsx`  
- `requirements.txt`  

## Getting started

1. Clone this repository:
   ```bash
   git clone https://github.com/salraja/WADI-Anomoly-Detection.git
   cd WADI-Anomoly-Detection
Unzip the bundle:


unzip Research.zip -d wadi_bundle
cd wadi_bundle
(Optional) Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate
Install the required packages:

pip install -r requirements.txt
How to run
Inside the wadi_bundle folder, run:


python train_lstm.py --epochs 10
The --epochs flag controls how many training iterations to run (default: 5).

When you execute the script, it will:

Load and merge the two CSV files

Parse attack intervals from the Excel sheet

Fill missing values, scale features, and slice into 100-second windows

Build a balanced test set (50% normal, 50% attack)

Train the LSTM Autoencoder and print the ROC-AUC

List the top 5 anomalous windows with per-feature error and permutation-importance scores

Example output

Loading WADI_14days.csv …
Loading WADI_attackdata.csv …
Total rows loaded: 1,382,402
Label distribution:
  1 (normal): 1,375,670
 -1 (attack):   6,732

Epoch 1/5   loss = 0.7245
…
Epoch 5/5   loss = 0.6063

Test ROC-AUC: 0.7270

Top 5 anomalous windows:
Window #64  MSE=2194.40
  • 1_P_006_STATUS    err=15.12  ΔMSE=120.57
  • 2_FIT_001_PV      err=10.99   ΔMSE= 85.43
  • …
License
This project (code + data bundle) is released under the MIT License. See LICENSE for details.

Citation
If you use this work in your research, please cite:

@article{shahid2025wadi,
  author       = {Salman Shahid},
  title        = {XAI for Intrusion Detection in Water Distribution Systems: WADI LSTM-AE Baseline},
  journal      = {IEEE Transactions on Information Forensics and Security},
  year         = {2025},
  doi          = {10.1109/TIFS.2025.xxxxxx},
  url          = {https://github.com/salraja/WADI-Anomoly-Detection}
}
