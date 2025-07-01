import pandas as pd
from sklearn.model_selection import train_test_split

# Function to load the dataset from a CSV file
def load_data(filepath):
    # I am loading the dataset
    data = pd.read_csv(filepath)
    return data

# Function to split the data into train and test sets
def split_data(data, test_size=0.2, random_state=42):
    # I am splitting the data into features and target
    X = data.drop('Class', axis=1)
    y = data['Class']
    # I am splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test 

if __name__ == "__main__":
    # I am loading the dataset
    data = load_data('creditcard.csv')
    print('Shape of the full dataset:', data.shape)
    # I am splitting the data
    X_train, X_test, y_train, y_test = split_data(data)
    print('Shape of X_train:', X_train.shape)
    print('Shape of X_test:', X_test.shape)
    print('Shape of y_train:', y_train.shape)
    print('Shape of y_test:', y_test.shape)
    print('Class distribution in y_train:', y_train.value_counts()) 