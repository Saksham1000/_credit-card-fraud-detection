import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Function to apply SMOTE to balance classes
def apply_smote(X_train, y_train):
    # I am applying SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

# Function to train and evaluate models
def train_and_evaluate(X_train, y_train, X_test, y_test):
    # I am training Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    # I am training XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    # I am evaluating models
    for name, pred in zip(['Logistic Regression', 'XGBoost'], [lr_pred, xgb_pred]):
        print(f'\n{name} Results:')
        print(classification_report(y_test, pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, pred))
    # I am saving the best model (XGBoost)
    joblib.dump(xgb, 'best_model.joblib')

# Function to plot ROC curve
def plot_roc_curve(y_test, lr_probs, xgb_probs):
    # I am plotting ROC curves for both models
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
    plt.figure(figsize=(8,6))
    plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
    plt.plot(fpr_xgb, tpr_xgb, label='XGBoost')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(y_test, preds, model_name):
    # I am plotting the confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Function to plot XGBoost feature importance
def plot_xgb_feature_importance(model, feature_names):
    # I am plotting XGBoost feature importance
    importance = model.feature_importances_
    indices = importance.argsort()[::-1]
    plt.figure(figsize=(10,6))
    plt.title('XGBoost Feature Importance')
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # I am loading the dataset
    data = pd.read_csv('creditcard.csv')
    # I am splitting the data
    X = data.drop('Class', axis=1)
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print('Before SMOTE:')
    print('Class distribution in y_train:')
    print(y_train.value_counts())
    # I am applying SMOTE
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    print('After SMOTE:')
    print('Class distribution in y_train_res:')
    print(pd.Series(y_train_res).value_counts())
    # I am training Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_res, y_train_res)
    lr_pred = lr.predict(X_test)
    lr_probs = lr.predict_proba(X_test)[:,1]
    # I am training XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train_res, y_train_res)
    xgb_pred = xgb.predict(X_test)
    xgb_probs = xgb.predict_proba(X_test)[:,1]
    # I am evaluating models
    for name, pred in zip(['Logistic Regression', 'XGBoost'], [lr_pred, xgb_pred]):
        print(f'\n{name} Results:')
        print(classification_report(y_test, pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, pred))
    # I am saving the best model (XGBoost)
    joblib.dump(xgb, 'best_model.joblib')
    # I am plotting ROC curves
    plot_roc_curve(y_test, lr_probs, xgb_probs)
    # I am plotting confusion matrices
    plot_confusion_matrix(y_test, lr_pred, 'Logistic Regression')
    plot_confusion_matrix(y_test, xgb_pred, 'XGBoost')
    # I am plotting XGBoost feature importance
    plot_xgb_feature_importance(xgb, X.columns) 