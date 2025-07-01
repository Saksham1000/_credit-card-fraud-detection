import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

# Function to load the trained model
def load_model(model_path):
    # I am loading the trained model
    model = joblib.load(model_path)
    return model

# Function to make predictions
def make_predictions(model, data):
    # I am making predictions
    predictions = model.predict(data)
    return predictions

# Streamlit app
st.set_page_config(page_title='Credit Card Fraud Detection', layout='centered')
st.sidebar.title('About This App')
st.sidebar.info('''\
This dashboard lets you upload credit card transaction data and predicts which transactions are fraudulent using a machine learning model.\n\n- Built by Saksham Lamsal\n- Model: XGBoost (trained on Kaggle dataset)\n''')

st.title('Credit Card Fraud Detection')
st.write('''\
**Instructions:**\n
1. Upload a CSV file with the same columns as the original dataset (except 'Class').\n2. The app will predict which transactions are fraudulent (1) or not (0).\n3. You will see a summary and a pie chart of the results.\n''')

uploaded_file = st.file_uploader('Upload your credit card transaction CSV file', type=['csv'])
if uploaded_file is not None:
    # I am reading the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    # I am validating the input columns
    required_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    missing_cols = [col for col in required_columns if col not in data.columns]
    extra_cols = [col for col in data.columns if col not in required_columns]
    if missing_cols:
        st.error(f"The uploaded file is missing the following required columns: {missing_cols}")
    elif 'Class' in data.columns:
        st.error("The uploaded file should NOT contain the 'Class' column. Please remove it and try again.")
    else:
        # I am showing a warning if there are extra columns
        if extra_cols:
            st.warning(f"The uploaded file has extra columns that will be ignored: {extra_cols}")
        st.write('Preview of uploaded data:')
        st.write(data.head())
        model = load_model('best_model.joblib')
        predictions = make_predictions(model, data)
        if hasattr(model, 'predict_proba'):
            fraud_probs = model.predict_proba(data)[:, 1]
        else:
            fraud_probs = [None] * len(predictions)
        st.write('Fraud Predictions (1 = Fraud, 0 = Not Fraud):')
        st.write(predictions)
        st.write('Fraud Probability (model confidence for fraud):')
        st.write(fraud_probs)
        # I am adding a download button for predictions and probabilities
        output_df = data.copy()
        output_df['Prediction'] = predictions
        output_df['Fraud_Probability'] = fraud_probs
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='Download Predictions as CSV',
            data=csv,
            file_name='fraud_predictions.csv',
            mime='text/csv'
        )
        # I am visualizing the prediction results
        result_counts = pd.Series(predictions).value_counts().sort_index()
        st.write('Prediction Summary:')
        st.write(result_counts.rename(index={0: 'Not Fraud', 1: 'Fraud'}))
        fig, ax = plt.subplots()
        # I am mapping prediction values to human-readable labels
        label_map = {0: 'Not Fraud', 1: 'Fraud'}
        labels = [label_map.get(i, str(i)) for i in result_counts.index]
        ax.pie(result_counts, labels=labels, autopct='%1.1f%%', colors=['#66b3ff','#ff6666'][:len(labels)])
        ax.set_title('Fraud vs. Not Fraud Predictions')
        st.pyplot(fig)

        # I am adding SHAP explainability
        st.write('### Model Explainability (SHAP)')
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)
        plt.figure(figsize=(10,6))
        shap.summary_plot(shap_values, data, plot_type='bar', show=False)
        st.pyplot(plt.gcf())
        plt.clf()

        # I am adding a bar chart of XGBoost feature importances
        st.write('### XGBoost Feature Importances')
        importances = model.feature_importances_
        feature_names = data.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.bar(importance_df['Feature'], importance_df['Importance'])
        ax2.set_title('Feature Importances')
        ax2.set_ylabel('Importance')
        ax2.set_xlabel('Feature')
        plt.xticks(rotation=90)
        st.pyplot(fig2) 