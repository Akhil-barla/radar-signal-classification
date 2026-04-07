# Radar Signal Classification Benchmark

An end-to-end deep learning benchmark treating human activity data as continuous, radar-like temporal sequences. This project evaluates various Neural Network architectures to determine the best method for capturing temporal patterns and micro-Doppler signatures.

---

## 📌 Dataset
**Name:** [UCI Human Activity Recognition (HAR) Using Smartphones Data Set](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

We leverage the 561-feature vector from the UCI HAR dataset. To simulate radar returns (such as PRF bursts tracking a target over a dwell period), the baseline flat feature array is refactored into a sequential matrix structure: `(Samples, 51 Timesteps, 11 Channels)`.

---

## 🛠 Tech Stack
- **Language:** Python 3.8+
- **Deep Learning Framework:** TensorFlow 2.x / Keras
- **Data Manipulation:** NumPy, Pandas, Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Dashboard Frontend:** HTML5, CSS3 (Custom Variables/Animations), Vanilla Javascript


## 🚀 Installation & Setup

1. **Clone the Repository (If applicable):**
   ```bash
   git clone <YOUR_REPO_URL>
   cd files
   ```

2. **Install Dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
   ```

## 🧠 How We Implemented & How It Works

### Pipeline Flow:
1. **Automated Download:** The `radar_classification.py` script automatically fetches the official UCI ZIP archive if it isn't present locally.
2. **Preprocessing:** It extracts the dataset in memory, normalizes the inputs utilizing `StandardScaler`, and reshapes the standard 1D features into simulated 2D time-series matrices to create temporal significance.
3. **Model Engineering:** We implemented three distinct neural network pipelines:
    - **FNN (Feedforward Neural Network - Baseline)**: Evaluates the raw 561 features independently. Crucially lacks temporal/sequential awareness.
    - **CNN-1D (Convolutional Network)**: Sweeps a 1D kernel across the 51 timesteps to extract local radar-pulse-like patterns.
    - **LSTM (RNN)**: Deploys gated memory cells to track the feature evolution across the entire track dwell timeline.
4. **Evaluation:** Models are independently trained and tested. Loss curves and normalized Confusion Matrices are pushed to the `./outputs/` directory.
5. **JSON Injection:** An external script updates `radar_dashboard.html` dynamically by injecting the output JSON metrics directly into the static HTML for immediate viewing.


## 📊 How To Run

### 1. Execute the Training Pipeline
Run the script manually to begin data downloading, preprocessing, and model training:
```bash
py radar_classification.py
```
*(If you are on Linux/macOS, use `python radar_classification.py` or `python3`)*

### 2. Update the Dashboard
After training is finished to generate new local results, sync the new data to the HTML visualizer:
```bash
py update_dashboard.py
```

### 3. View the Results
Double-click `radar_dashboard.html` to open it in your browser. You will see:
- Real-time testing accuracy and total parameter comparisons.
- SVG Line charts mapping Epoch training performance.
- Interactive tab sections explaining the algorithmic superiority behind sequence models (LSTM) for capturing Doppler evolution versus flat feature mapping (FNN).


## 📚 Reference Papers
The dataset used in this benchmark originates from the following scholarly research:
> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. *"A Public Domain Dataset for Human Activity Recognition Using Smartphones."* 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.
