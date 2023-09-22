# Standard Data Science Libraries
import numpy as np
import pandas as pd

# Model Training
from models.DecisionTree import tune_decision_tree
from models.KNearestNeighbor import tune_knn
from models.NeuralNetwork import tune_neural_network, evaluate_model
from models.SupportVectorMachine import tune_svm
from models.XGBoost import tune_xgboost

# Modeling Libraries
from sklearn.model_selection import train_test_split

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Model Evaluation
from sklearn.metrics import roc_auc_score, accuracy_score

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Python Standard Libraries
import time
import math

# Saving Models
import joblib


def filename_dataset_assertions(filename=None, dataset_type=None) -> None:
    assert filename in [
        "../data/auction_verification_dataset/data.csv",
        "../data/student_dropout_dataset/data.csv",
    ], "filename argument must be in a valid location."

    if filename == "../data/auction_verification_dataset/data.csv":
        assert (
            dataset_type == "auction"
        ), "dataset_type argument must be 'auction' when the filename points to the auction dataset."
    elif filename == "../data/auction_verification_dataset/data.csv":
        assert (
            dataset_type == "dropout"
        ), "dataset_type argument must be 'dropout' when the filename points to the dropout dataset."


def data_loading(
    filename=None, dataset_type=None
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if dataset_type == "auction":
        data = pd.read_csv(filename)
    elif dataset_type == "dropout":
        data = pd.read_csv(filename, delimiter=";")

        data["Target"] = data["Target"].replace(
            {"Graduate": 0, "Dropout": 1, "Enrolled": 2}
        )

    train_val_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42)

    if dataset_type == "auction":
        X_train = train_df.iloc[:, :-2].copy()
        y_train = train_df.iloc[:, -2].copy().astype(int)

        X_val = val_df.iloc[:, :-2].copy()
        y_val = val_df.iloc[:, -2].copy().astype(int)

        X_test = test_df.iloc[:, :-2].copy()
        y_test = test_df.iloc[:, -2].copy().astype(int)
    elif dataset_type == "dropout":
        X_train = train_df.iloc[:, :-1].copy()
        y_train = train_df.iloc[:, -1].copy().astype(int)

        X_val = val_df.iloc[:, :-1].copy()
        y_val = val_df.iloc[:, -1].copy().astype(int)

        X_test = test_df.iloc[:, :-1].copy()
        y_test = test_df.iloc[:, -1].copy().astype(int)

    return (X_train, y_train, X_val, y_val, X_test, y_test)


def get_correlation_heatmap(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
) -> None:
    np.random.seed(1234)

    filename_dataset_assertions(filename, dataset_type)

    (X_train, y_train, X_val, y_val, X_test, y_test) = data_loading(
        filename, dataset_type
    )

    if dataset_type == "dropout":
        rename_dict = {
            "Curricular units 1st sem (approved)": "CU 1st SEM (appr)",
            "Curricular units 1st sem (grade)": "CU 1st SEM (grade)",
            "Curricular units 1st sem (evaluations)": "CU 1st SEM (eval)",
            "Curricular units 2nd sem (approved)": "CU 2nd SEM (appr)",
            "Curricular units 2nd sem (grade)": "CU 2nd SEM (grade)",
            "Curricular units 2nd sem (evaluations)": "CU 2nd SEM (eval)",
        }
        X_train = X_train[
            [
                "Curricular units 1st sem (approved)",
                "Curricular units 1st sem (grade)",
                "Curricular units 1st sem (evaluations)",
                "Curricular units 2nd sem (approved)",
                "Curricular units 2nd sem (grade)",
                "Curricular units 2nd sem (evaluations)",
                "Course",
                "Tuition fees up to date",
                "Age at enrollment",
            ]
        ].rename(columns = rename_dict)
        X_test = X_test[
            [
                "Curricular units 1st sem (approved)",
                "Curricular units 1st sem (grade)",
                "Curricular units 1st sem (evaluations)",
                "Curricular units 2nd sem (approved)",
                "Curricular units 2nd sem (grade)",
                "Curricular units 2nd sem (evaluations)",
                "Course",
                "Tuition fees up to date",
                "Age at enrollment",
            ]
        ].rename(columns = rename_dict)

    corr = X_train.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True)

    # plt.xticks(rotation=45)
    # plt.yticks(rotation=45)

    plt.tight_layout()

    plt.savefig(f"../outputs/EDA/correlation_heatmap_{dataset_type}.png")
