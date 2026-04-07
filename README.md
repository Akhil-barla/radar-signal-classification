# Radar Signal Classification Benchmark

This project demonstrates a deep learning benchmark for classifying radar-like sequential time-series data. It evaluates three different neural network architectures on the UCI HAR Dataset to compare how effectively they capture temporal patterns.

## Project Structure
- `radar_classification.py`: The core machine learning pipeline. It downloads the dataset, preprocesses the data, builds three models (FNN, CNN-1D, LSTM), trains them, and outputs performance metrics.
- `radar_dashboard.html`: A static, interactive web dashboard to visualize the training histories, confusion matrices, and model comparisons.
- `outputs/`: The directory where execution results (Loss curve charts, Confusion matrix images, and `radar_results.json`) are saved.

## Environment Setup

Ensure you have Python installed (Python 3.8+ recommended). 
Install the required dependencies using pip:

```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
```

## How to Run the Project

### 1. Training the Models
To run the deep learning pipeline from scratch:
Open your terminal (PowerShell or Command Prompt) and run:

```bash
py radar_classification.py
```
*(Note: Use `python` instead of `py` if you are on Mac/Linux or depending on your specific Python installation).*

**What this does:**
1. Downloads the official UCI HAR Dataset.
2. Preprocesses the 561-feature vectors into a 51-timestep sequence format.
3. Compiles and trains a Feedforward Neural Network (FNN), a 1D Convolutional Neural Network (CNN), and a Long Short-Term Memory network (LSTM) for 5 epochs each.
4. Generates performance plots and saves data to `./outputs/radar_results.json`.

### 2. Viewing the Dashboard
To explore the results visually:
Simply **double-click** the `radar_dashboard.html` file to open it in your default web browser.

The dashboard includes:
- Animated performance counters.
- Loss and accuracy curves.
- Normalised confusion matrices.
- "Why LSTM?" architectural reasoning and parameter efficiency charts.
- The actual Python code implementations of each model.
