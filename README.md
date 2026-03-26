## **EEG & Physiological Signal Emotion and Stress Recognition**

A end-to-end machine learning research project for emotion recognition from EEG signals (DEAP dataset) and stress detection from multimodal physiological signals (WESAD dataset). The project is structured as four sequential Jupyter notebooks covering preprocessing, feature engineering, model training, and evaluation.

---

## **Notebooks**

### **NB1Final.ipynb — DEAP EEG Preprocessing Pipeline**
Ingests raw DEAP dataset files (32 subjects, 40 trials each) and produces a clean, reusable feature matrix for downstream tasks. Key steps include baseline removal, artifact detection, frequency-band power extraction (delta, theta, alpha, beta, gamma), and statistical feature computation across EEG and peripheral channels. Outputs train/test splits and subject-indexed feature arrays saved for use by NB2.

---

### **NB2EmoRec.ipynb — DEAP 4-Class Emotion Recognition**
Loads preprocessed DEAP features and classifies emotional states into four quadrants (HVHA, HVLA, LVHA, LVLA) based on continuous valence/arousal ratings. Trains and tunes SVM, Random Forest, XGBoost, and LightGBM baselines using Optuna hyperparameter optimization (40 trials each), then combines them in a stacking ensemble. Addresses class imbalance with SMOTE and evaluates using macro F1-score and full classification reports.

---

### **NB3wesaddpfin.ipynb — WESAD Stress Recognition Preprocessing**
A standalone preprocessing pipeline for the WESAD multimodal physiological dataset. Loads chest-worn sensor signals (ECG, EDA, respiration, temperature) from real or synthetic subjects, applies signal quality assessment, amplitude clipping, z-score spike removal, and NaN interpolation for artifact removal. Segments signals into windows and extracts time-domain and frequency-domain features, saving outputs for direct import into NB4.

---

### **NB4WESADmodfinal.ipynb — WESAD Stress Recognition Modeling**
Companion training notebook to NB3. Implements a stacking ensemble (Random Forest, XGBoost, LightGBM, SVM-RBF with Platt calibration) and a lightweight 1D-CNN, trained under strict Leave-One-Subject-Out (LOSO) cross-validation to prevent biometric data leakage — following the benchmark protocol from Schmidt et al. (2018). Hyperparameters are tuned with Optuna on the inner CV loop, and SMOTE is applied inside the pipeline to prevent contamination of validation folds.

---

## **Tech Stack**

- **Python:** NumPy, pandas, SciPy, scikit-learn  
- **ML Models:** SVM, Random Forest, XGBoost, LightGBM, stacking ensembles, 1D-CNN  
- **Hyperparameter Tuning:** Optuna  
- **Imbalance Handling:** imbalanced-learn (SMOTE)  
- **Explainability:** SHAP  
- **Datasets:** DEAP · WESAD  
- **Environment:** Google Colab / Kaggle  
