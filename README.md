# ❤️ CardioRisk-NN

**Heart disease risk prediction with a neural network and an interactive Gradio dashboard.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Gradio](https://img.shields.io/badge/UI-Gradio-red)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/cardiorisk-nn/blob/main/heart_disease_prediction.ipynb)

> ⚠️ **Disclaimer:** This project is for educational purposes only. It is not a medical device and must not be used for diagnosis or clinical decisions.

---

## Overview

CardioRisk-NN trains a small feed-forward neural network to estimate the probability that a patient has heart disease from 13 clinical attributes. The trained model is wrapped in a multi-tab web dashboard so results can be explored without writing code.

## Features

- **End-to-end pipeline:** loading, one-hot encoding, stratified split, scaling, training, and evaluation
- **Regularized neural network:** dropout layers and early stopping against overfitting
- **Interactive dashboard (Gradio):**
  - **Single Patient:** input form, colour-coded risk gauge, probability bars, and a *"What drives this prediction?"* chart
  - **Batch Prediction:** upload a CSV, get predictions for every row, and download the results
  - **Model Insights:** training curves, confusion matrix, and test accuracy
  - **About:** a description of each input feature
- **Session history:** keeps a log of the patients checked in the current session

## Dataset

The [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (Cleveland subset): **303 patients, 13 features, 1 binary target** (`1` = heart disease, `0` = no disease).

| Type | Features |
|---|---|
| Numeric | `age`, `trestbps`, `chol`, `thalach`, `oldpeak` |
| Categorical | `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal` |

## Methodology

1. **Preprocessing:** categorical columns are one-hot encoded (13 → 30 input features), and numeric features are standardized with `StandardScaler`. The scaler is fit on the training set only, which avoids data leakage.
2. **Split:** 80% train / 20% test, stratified, `random_state=42`.
3. **Model:**

   | Layer | Output | Notes |
   |---|---|---|
   | Input | 30 | One-hot + scaled features |
   | Dense + ReLU | 32 | Dropout 0.3 |
   | Dense + ReLU | 16 | Dropout 0.3 |
   | Dense + Sigmoid | 1 | Probability of heart disease |

   1,537 trainable parameters in total.
4. **Training:** Adam optimizer, binary cross-entropy loss, batch size 16, up to 200 epochs with early stopping (patience 15, best weights restored). In the reported run, training stopped at epoch 62.

## Results

Evaluated on the held-out test set (61 patients):

| Metric | Value |
|---|---|
| **Accuracy** | **81.97%** |
| Precision (disease) | 0.81 |
| Recall (disease) | 0.88 |
| F1-score (disease) | 0.84 |

**Confusion matrix**

|  | Predicted: No disease | Predicted: Disease |
|---|---|---|
| **Actual: No disease** | 21 | 7 |
| **Actual: Disease** | 4 | 29 |

> The test set is small (61 samples), so these figures carry considerable uncertainty. Cross-validation would give a more reliable estimate.

## Screenshots

| Single Patient | Model Insights |
|---|---|
| ![Single patient](screenshots/single_patient.png) | ![Model insights](screenshots/model_insights.png) |

## Getting Started

### Option 1: Google Colab (recommended)

1. Click the **Open in Colab** badge above.
2. Upload `data/heart.csv` to your Colab session or Google Drive, and update the `path` variable in the data-loading cell.
3. Run all cells in order. The last cell launches the dashboard and prints a public link.

### Option 2: Run locally

```bash
git clone https://github.com/YOUR-USERNAME/cardiorisk-nn.git
cd cardiorisk-nn
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook heart_disease_prediction.ipynb
```

When running locally, remove the Google Drive mount cell and set `path = 'data/heart.csv'`.

## Project Structure

```
cardiorisk-nn/
├── heart_disease_prediction.ipynb   # Training, evaluation, and dashboard
├── data/
│   └── heart.csv                    # UCI Heart Disease (Cleveland) dataset
├── screenshots/                     # Dashboard images used in this README
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Batch Prediction Format

Upload a CSV with these columns (the `target` column is optional, and if present the dashboard reports accuracy):

`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`

## Limitations and Future Work

- Small dataset (303 patients) from a single source, so the model may not generalize to other populations.
- Only a single train/test split is reported; k-fold cross-validation would be more robust.
- Possible extensions: compare with Logistic Regression, Random Forest, and XGBoost baselines; add SHAP-based explanations; calibrate probabilities; deploy on Hugging Face Spaces.

## Tech Stack

Python · TensorFlow/Keras · scikit-learn · pandas · NumPy · Matplotlib · Gradio

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

- Dataset: Janosi, A., Steinbrunn, W., Pfisterer, M., Detrano, R. *Heart Disease*. UCI Machine Learning Repository.

## Author

**Your Name**
[GitHub](https://github.com/YOUR-USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR-PROFILE) · your.email@example.com
