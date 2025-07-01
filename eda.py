import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Function to plot class distribution
def plot_class_distribution(data):
    # I am plotting the class distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x='Class', data=data)
    plt.title('Class Distribution (0: Non-Fraud, 1: Fraud)')
    plt.show()

# Function to plot correlation heatmap
def plot_correlation_heatmap(data):
    # I am plotting the correlation heatmap
    plt.figure(figsize=(16,12))
    sns.heatmap(data.corr(), cmap='coolwarm', annot=False)
    plt.title('Feature Correlation Heatmap')
    plt.show()

# Function to plot PCA visualization
def plot_pca(data):
    # I am plotting a PCA visualization for the first two principal components
    features = data.drop(['Class'], axis=1)
    pca = PCA(n_components=2)
    components = pca.fit_transform(features)
    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2']) # type: ignore   
    pca_df['Class'] = data['Class'].values
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='PC1', y='PC2', hue='Class', data=pca_df, alpha=0.3, palette='Set1')
    plt.title('PCA of Credit Card Transactions')
    plt.show()

# Function to plot boxplot of Amount by Class
def plot_amount_boxplot(data):
    # I am plotting a boxplot of transaction Amount by Class
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Class', y='Amount', data=data)
    plt.title('Transaction Amount by Class')
    plt.show()

if __name__ == "__main__":
    # I am loading the dataset
    data = pd.read_csv('creditcard.csv')
    # I am printing summary statistics
    print('Summary statistics:')
    print(data.describe())
    # I am plotting the class distribution
    plot_class_distribution(data)
    # I am plotting the correlation heatmap
    plot_correlation_heatmap(data)
    # I am plotting the boxplot of Amount by Class
    plot_amount_boxplot(data)
    # I am plotting the PCA visualization
    plot_pca(data) 