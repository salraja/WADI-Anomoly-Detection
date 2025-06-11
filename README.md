# WADI Anomaly Detection Bundle

This repository contains everything you need to run the LSTM Autoencoder baseline on the WADI dataset. Rather than unpacking multiple files, simply download and unzip the `Research.zip` archive, which includes both the training script and the data.

## What’s inside `Research.zip`

- `train_lstm.py`  
- `WADI_14days.csv`  
- `WADI_attackdata.csv`  
- `attack_description.xlsx`  
- `requirements.txt`  

## Getting started

1. Clone this repository:
   ```bash
   git clone https://github.com/salraja/WADI-Anomaly-Research-Zip.git
   cd WADI-Anomaly-Research-Zip
Unzip the bundle:

unzip Research.zip -d wadi_bundle
cd wadi_bundle
(Optional) Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate
Install the required packages:

pip install -r requirements.txt
How to run
From inside the wadi_bundle folder, launch the training and evaluation with:

python train_lstm.py --epochs 10
You can change the number of epochs by adjusting the --epochs flag (default is 5).

When you run the script it will:

Load and combine the two CSV files
Read the attack windows from the Excel file
Fill missing values, scale features, and split data into 100-second windows
Create a balanced test set (50% normal, 50% attack)
Train the LSTM Autoencoder and report the ROC-AUC score
Print out the top 5 most anomalous windows, showing both per-feature error and permutation-importance scores

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
This project (both code and data bundle) is released under the MIT License. See the LICENSE file for details.

If you use this work
Please cite:

@article{shahid2025wadi,
  author       = {Salman Shahid},
  title        = {XAI for Intrusion Detection in Water Distribution Systems: WADI LSTM-AE Baseline},
  journal      = {IEEE Transactions on Information Forensics and Security},
  year         = {2025},
  doi          = {10.1109/TIFS.2025.xxxxxx},
  url          = {https://github.com/salraja/WADI-Anomaly-Research-Zip}
}
