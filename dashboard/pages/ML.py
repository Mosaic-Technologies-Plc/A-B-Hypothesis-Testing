import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import sys
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))

from util import get_categorical_columns, get_numerical_columns
from ml_utils import label_encoder



@st.cache
def loadData():
    df = pd.read_csv('data/AdSmartClean_data.csv')
    main_df = df.copy()
    # print(df.head())
    numerical_cols = get_numerical_columns(df)
    categorical_cols = get_categorical_columns(df)
    # print(categorical_cols)
    categorical_cols.remove('auction_id')
    categorical_cols_encoded = label_encoder(df)
    categorical_cols_encoded.drop(columns=['auction_id'], inplace=True)


    X = df.copy()
    #Dropping duplicate column names
    X.drop(['yes', 'no', 'platform_os', 'hour'], axis=1, inplace=True)

    X.drop(categorical_cols, axis=1, inplace=True)

    X = pd.concat([X, categorical_cols_encoded], axis=1)

    X['target'] = 1

    X.loc[X['no'] == 1, 'target'] = 0

    y = X['target']
    X.drop(['target'], axis=1, inplace=True)
    X.drop(['yes', 'no'], axis=1, inplace=True)

    return X, y

def get_feature_importance(model, x):
    feature_importance = None
    # print(str(model))
    if "LogisticRegression" in str(model):
        feature_importance = model.coef_[0]
    else:
        feature_importance = model.feature_importances_
    feature_array = {}
    for i, v in enumerate(feature_importance):
        feature_array[x.columns[i]] = round(float(v), 2)
    return feature_array



def cross_validation(model, X, y, fold=5):
    """Perform 5 Folds Cross-Validation Parameters.
    ----------
    model: Python Class, default=None
            This is the machine learning algorithm to be used for training.
    X: array
        This is the matrix of features.
    y: array
        This is the target variable.
    fold: int, default=5
        Determines the number of folds for cross-validation.
    Returns
    -------
    The function returns a dictionary containing the metrics 'accuracy', 'precision',
    'recall', 'f1' for both training set and validation set.
    """
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    coef = []
    results = cross_validate(estimator=model,
                                X=X,
                                y=y,
                                cv=fold,
                                scoring=scoring,
                                return_train_score=True, return_estimator=True)

    #Print the coefficients of the features in the decision tree
    # print(results['estimator'])
    # print("Coefficients: \n", results.best_estimator_.feature_importances_)

    # for model in results['estimator']:
    #     print(model.coef_)
        
    # coef = get_feature_importance(results['estimator'][-1], X)
    for m in results['estimator']:
        coef.append(get_feature_importance(m, X))
    # print(coef)
    return {"Training Accuracy scores": results['train_accuracy'],
        "Mean Training Accuracy": results['train_accuracy'].mean()*100,
        "Training Precision scores": results['train_precision'],
        "Mean Training Precision": results['train_precision'].mean(),
        "Training Recall scores": results['train_recall'],
        "Mean Training Recall": results['train_recall'].mean(),
        "Training F1 scores": results['train_f1'],
        "Mean Training F1 Score": results['train_f1'].mean(),
        "Validation Accuracy scores": results['test_accuracy'],
        "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
        "Validation Precision scores": results['test_precision'],
        "Mean Validation Precision": results['test_precision'].mean(),
        "Validation Recall scores": results['test_recall'],
        "Mean Validation Recall": results['test_recall'].mean(),
        "Validation F1 scores": results['test_f1'],
        "Mean Validation F1 Score": results['test_f1'].mean(),
        "Coefficients": coef
        }



def plot_result(x_label, y_label, plot_title, train_data, val_data, num_folds):
    """Plot a grouped bar chart showing the training and validation results of the ML model in each fold after applying K-fold cross-validation.
        Parameters
        ----------
        x_label: str,
        Name of the algorithm used for training e.g 'Decision Tree'
        y_label: str,
        Name of metric being visualized e.g 'Accuracy'
        plot_title: str,
        This is the title of the plot e.g 'Accuracy Plot'
        train_result: list, array
        This is the list containing either training precision, accuracy, or f1 score.
        val_result: list, array
        This is the list containing either validation precision, accuracy, or f1 score.
        Returns
        -------
        The function returns a Grouped Barchart showing the training and validation result
        in each fold.
    """

    plt.figure(figsize=(12, 6))
    labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
    X_axis = np.arange(num_folds)
    fig, ax = plt.subplots()

    
    plt.ylim(0.40000, 1)
    plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
    plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
    plt.title(plot_title, fontsize=30)
    plt.xticks(X_axis, labels[:num_folds])
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend()
    plt.grid(True)
    return fig


def plot_feature_importance(feature_importance):
    importance = pd.DataFrame({
        'features': feature_importance.keys(),
        'importance_score': feature_importance.values()
    })
    fig = plt.figure(figsize=[8, 5])
    ax = sns.barplot(x=importance['features'],
                        y=importance['importance_score'])
    ax.set_title("Feature importance")
    ax.set_xlabel("Features", fontsize=20)
    ax.set_ylabel("Importance", fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    
    # figure = ax.get_figure()
    return fig



def setMLTitle():
    st.header('Modeling for Our Cleaned Initial Data Set')
    logisticsRegressionRun()


def logisticsRegressionRun():
    model_name = "Logistic Regression"
    st.write("")
    st.header("Logistic Regression on Cleaned Data Set")
    st.write("")
    X, y = loadData()
    # option = st.selectbox(
    #  'Choose penality for LogisticRegression',
    #  ('l1', 'l2'))
    C_param_range = [0.001,0.01,0.1,1,10,100]
    c_s = st.selectbox(
     'Choose regularization strength',
     (0.001,0.01,0.1,1,10,100))
    model = LogisticRegression(C=c_s)
    # scores = ['accuracy', 'precision', 'recall', 'f1']
    # selectedScores = st.multiselect("choose combaniation of scores", scores)
    # if selectedScores:
    num_folds = st.slider("Select number of Folds", 1, 5, 5, key=21)
    model_result = cross_validation(model, X, y, num_folds)
    st.write(f"Training data accuracy: {model_result['Training Accuracy scores'][0]}")
    st.write(f"Validation data accuracy: {model_result['Validation Accuracy scores'][0]}")
    p = plot_result(model_name, f"Accuracy", "Accuracy scores in {num_folds} Folds",
            model_result["Training Accuracy scores"],
            model_result["Validation Accuracy scores"],num_folds)

    st.pyplot(p)
    labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
    num = st.slider("Select fold number", 1, num_folds, 1, key=22)

    st.write(f"Feature importance for the {labels[num-1]}")
    st.write(model_result['Coefficients'][num-1])
    st.write("")
    pl = plot_feature_importance(model_result['Coefficients'][num-1])
    st.pyplot(pl)