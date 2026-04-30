# BCI EEG ABAT Integrated Project

This is the clean integrated version of the project with one set of six phases and ABAT included in the final pipeline.

## Datasets

- Public dataset: `/Users/saisharmilakalva/Desktop/Public_Dataset`
- Lab dataset: `/Users/saisharmilakalva/Desktop/Lab_Dataset`

## Six Phases

1. `phase1_data_preparation.py`
   Raw data inspection and visualization.

2. `phase2_preprocessing.py`
   Filtering and preprocessing of EEG recordings.

3. `phase3_feature_extraction.py`
   Feature extraction and signal representation building.

4. `phase4_model_training.py`
   Baseline model training on compact EEG windows.

5. `phase5_abat_automation.py`
   ABAT-style adversarial augmentation and robust training.

6. `phase6_reporting_visualization.py`
   Final baseline-versus-ABAT comparison and reporting outputs.

## Example Commands

Run from `/Users/saisharmilakalva/Desktop/BCI_EEG_ABAT_Integrated_Project`

```bash
python3 code/phase1_data_preparation.py
python3 code/phase2_preprocessing.py
python3 code/phase3_feature_extraction.py
python3 code/phase4_model_training.py --dataset lab
python3 code/phase5_abat_automation.py --dataset lab
python3 code/phase6_reporting_visualization.py --dataset lab
```

## Why This Folder Exists

The earlier project was split between a baseline branch and an experimental ABAT branch. This folder merges the work into one cleaner structure so your final project reads as a single six-phase system.
