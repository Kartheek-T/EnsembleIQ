\# **EnsembleIQ – Credit Card Fraud Detection**



An ensemble machine learning-based credit card fraud detection system that uses multiple Bagging and Boosting algorithms to identify potentially fraudulent transactions.



\## Project Overview



EnsembleIQ is designed to detect fraudulent credit card transactions by leveraging multiple machine learning models. It provides transaction-level predictions, fraud probability analysis, and comparison of multiple ensemble algorithms through an interactive Streamlit dashboard.



\## Key Features



\- Credit card fraud detection using ensemble machine learning.

\- Seven trained machine learning models.

\- Individual model predictions and fraud probabilities.

\- Model comparison and consensus analysis.

\- Transaction CSV upload and batch prediction.

\- Interactive Streamlit dashboard.



\## Machine Learning Models



\*\*Bagging Algorithms\*\*

\- Random Forest

\- Extra Trees



\*\*Boosting Algorithms\*\*

\- AdaBoost

\- Gradient Boosting

\- XGBoost

\- LightGBM

\- CatBoost



\## Technology Stack



\- Python

\- Streamlit

\- NumPy

\- Pandas

\- Scikit-learn

\- XGBoost

\- LightGBM

\- CatBoost

\- Joblib



\## Project Structure



```text

EnsembleIQ/

│

├── app.py

├── requirements.txt

├── README.md

│

└── models/

&#x20;   ├── adaboost.pkl

&#x20;   ├── catboost.pkl

&#x20;   ├── extra\_trees.pkl

&#x20;   ├── gradient\_boosting.pkl

&#x20;   ├── lightgbm.pkl

&#x20;   ├── random\_forest.pkl

&#x20;   ├── scaler.pkl

&#x20;   └── xgboost.pkl

```



\## Installation and Setup



\### 1. Clone the repository



```bash

git clone https://github.com/Kartheek-T/EnsembleIQ.git

```



\### 2. Navigate to the project directory



```bash

cd EnsembleIQ

```



\### 3. Create a virtual environment (optional but recommended)



```bash

python -m venv venv

```



Activate it on Windows:



```bash

venv\\Scripts\\activate

```



\### 4. Install dependencies



```bash

pip install -r requirements.txt

```



\### 5. Run the application



```bash

python -m streamlit run app.py

```



The dashboard will open in your browser.



\## Dataset



The original credit card transaction dataset is not included in this repository because of its large file size.



\- Demo Transaction mode requires the original `creditcard.csv` file inside the `data/` directory.

\- Upload Transaction CSV mode allows users to upload their own transaction CSV file using the application's uploader.



The uploaded CSV must contain the required transaction features expected by the application.



\## Trained Models



The repository includes all seven trained model files and the feature scaler required for prediction.



\## Author



\*\*Kartheek Tadimeti\*\*



B.Tech – Computer Science and Engineering (AI \& ML)



GitHub: https://github.com/Kartheek-T



\---



\*EnsembleIQ – Intelligent Fraud Detection Through Ensemble Learning.\*



