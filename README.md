# Traffic Sign Classification with CNN

A traffic-sign image classification project built around a CNN workflow. The notebook covers dataset setup, preprocessing, training, evaluation, and saving a trained model.

## Project files

- `traffic_sign_detection.ipynb` — end-to-end training and evaluation notebook.
- `traffic_sign_classifier.keras` and `traffic_sign_classifier.joblib` — saved model artifacts.
- `class_names.npy` — class labels used by the saved model.
- `bench*.py` and `verify_plain_cnn.py` — CNN and transfer-learning experiments.

## Setup

Use Python 3.10 or later. Install the dependencies with:

```powershell
py -m pip install numpy matplotlib seaborn opencv-python scikit-learn tensorflow
```

Alternatively, run `python install_deps.py` or `install.bat` on Windows.

Open `traffic_sign_detection.ipynb` and run its cells in order. The notebook downloads and extracts the GTSRB training dataset when it is not already available. The dataset and downloaded archive are intentionally excluded from this repository because they are large; the notebook can fetch them when needed.

The standalone training and benchmark scripts expect class folders under `dataset/`.
