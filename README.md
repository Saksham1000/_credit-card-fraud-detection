# Credit Card Fraud Detection

This project is an end-to-end machine learning solution for detecting fraudulent credit card transactions using the popular Kaggle dataset. It is designed to showcase practical data science and ML skills.

## 🚀 Features
- Data loading and preprocessing
- Exploratory Data Analysis (EDA) with visualizations
- Handling class imbalance with SMOTE
- Model training and evaluation (Logistic Regression, XGBoost)
- ROC curve, confusion matrix, and feature importance plots
- Streamlit dashboard for interactive predictions

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- Streamlit
- Jupyter (optional for notebooks)

## 📂 Project Structure
```
load_data.py      # Data loading and splitting
eda.py            # Exploratory data analysis
train_model.py    # Model training, evaluation, and saving
predict.py        # Test predictions on new data
app.py            # Streamlit dashboard
requirements.txt  # Dependencies
README.md         # Project documentation
creditcard.csv    # Dataset (not included, download from Kaggle)
```

## 📥 Setup Instructions
1. Clone this repo and navigate to the project folder.
2. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project folder.
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## ▶️ Usage
- **Data Loading & EDA:**
  ```sh
  python load_data.py
  python eda.py
  ```
- **Model Training:**
  ```sh
  python train_model.py
  ```
- **Test Predictions:**
  ```sh
  python predict.py
  ```
- **Streamlit Dashboard:**
  ```sh
  streamlit run app.py
  ```

## 📊 Results
- Balanced class distribution with SMOTE
- Evaluation metrics: precision, recall, F1-score, ROC-AUC
- Visualizations: ROC curve, confusion matrix, feature importance, EDA plots
- Interactive dashboard for real-time predictions

