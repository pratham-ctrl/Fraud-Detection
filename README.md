<div align="center">
  <h1>🛡️ Fraud Detection System</h1>
  <p>
    <b>Secure. Fast. Intelligent.</b><br>
    A machine learning solution to identify fraudulent transactions in real-time using Random Forest.
  </p>
  
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit Learn">
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">

  <br><br>
</div>

## 📖 Overview

This project detects **fraudulent vs. legitimate transactions** using a supervised machine learning approach. The goal is to help financial systems reduce risks by analyzing transaction behavior and identifying suspicious activity early.

The model is built to be:
* 🔹 **Fast:** Optimized for quick inference.
* 🔹 **Reliable:** Handles complex transaction patterns.
* 🔹 **Developer-friendly:** Modular code structure.

---

## 🧩 Architecture

The system follows a standard ML pipeline:

```mermaid
graph LR
    A[Raw Data] --> B(Preprocessing)
    B --> C(Feature Engineering)
    C --> D{Random Forest Model}
    D -->|Legitimate| E[Approved]
    D -->|Fraud| F[Flagged]
````

### 🧠 Why Random Forest?

We chose **Random Forest** for this specific task because:

1.  **Non-linear Relationships:** It captures complex patterns in financial data better than linear models.
2.  **Overfitting Control:** Ensemble methods reduce the risk of memorizing the training data.
3.  **Feature Importance:** It provides insights into *why* a transaction was flagged (e.g., unusually high amount).

-----

## 🛠 Tech Stack

| Domain | Tools used |
| :--- | :--- |
| **Core** | Python 3.x |
| **ML & Data** | Scikit-Learn, Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Pickle (Model Serialization) |

-----

## 🏗 Project Structure

```bash
📦 fraud-detection
 ┣ 📂 data              # Dataset files (CSV)
 ┣ 📂 models            # Saved model files (.pkl)
 ┣ 📂 notebooks         # Jupyter notebooks for experimentation
 ┣ 📜 preprocess.py     # Data cleaning & transformation scripts
 ┣ 📜 train.py          # Script to train the model
 ┣ 📜 predict.py        # Script to generate predictions
 ┣ 📜 requirements.txt  # Project dependencies
 ┗ 📜 README.md         # Project documentation
```

-----

## 📊 Dataset

This model is designed to work with standard transaction datasets (e.g., Kaggle Credit Card Fraud, Synthetic Logs).

**Required Columns:**

  * `Amount`: Transaction value
  * `Type`: Transfer, Cash\_out, Payment, etc.
  * `Old Balance / New Balance`: Account states before/after transaction
  * `Time`: Timestamp of transaction

-----

## ⚙️ Installation

1.  **Clone the repository**

    ```bash
    git clone [https://github.com/pratham-ctrl/fraud-detection.git](https://github.com/pratham-ctrl/fraud-detection.git)
    cd fraud-detection
    ```

2.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

-----

## ▶️ Usage

### 1\. Train the Model

Run the training script to preprocess data and save the model to the `models/` folder.

```bash
python train.py
```

### 2\. Make Predictions (CLI)

Predict if a specific batch of transactions is fraudulent.

```bash
python predict.py --input sample_transactions.csv
```

### 3\. Use inside Python Code

You can import the predictor into your own backend API.

```python
from predict import predict_transaction

# Example transaction data
data = {
    "amount": 12000.50,
    "oldbalanceOrg": 50000.00,
    "newbalanceOrig": 37999.50,
    "type": "TRANSFER"
}

result = predict_transaction(data)
print(f"Transaction Status: {result}")
```

-----

## 📈 Model Performance

*Note: Results based on the test set evaluation.*

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **98%** |
| **Precision** | 97% |
| **Recall** | 95% |
| **F1-Score** | 96% |

-----

## 🔮 Future Improvements

  - [ ] Add **XGBoost / LightGBM** models for performance comparison.
  - [ ] Integrate a real-time API using **FastAPI**.
  - [ ] Build a **Streamlit dashboard** for visual analytics.
  - [ ] Add **SHAP** values to explain individual fraud predictions.

-----

## 📄 License

This project is open-source and available under the **MIT License**.

\<div align="center"\>
\<small\>Made with ❤️ by \<a href="https://www.google.com/search?q=https://github.com/pratham-ctrl"\>Pratham Parikh\</a\>\</small\>
\</div\>

```
