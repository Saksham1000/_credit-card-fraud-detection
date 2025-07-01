import pandas as pd
import joblib

# Function to load the trained model
def load_model(model_path):
    # I am loading the trained model
    model = joblib.load(model_path)
    return model

# Function to make predictions on new data
def predict(model, data):
    # I am making predictions on new data
    predictions = model.predict(data)
    return predictions 

if __name__ == "__main__":
    # I am loading the trained model
    model = load_model('best_model.joblib')
    # I am loading a sample of new data (first 5 rows from creditcard.csv, dropping 'Class')
    data = pd.read_csv('creditcard.csv')
    sample = data.drop('Class', axis=1).head()
    print('Sample data:')
    print(sample)
    # I am making predictions
    predictions = predict(model, sample)
    print('Predictions (1 = Fraud, 0 = Not Fraud):')
    print(predictions) 