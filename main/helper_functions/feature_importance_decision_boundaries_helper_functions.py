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

# Feature Explanations
import shap

# Helper Functions
from helper_functions.model_training_evaluation_helper_functions import (
    get_all_best_models,
)


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


def load_models(
    filename="../data/auction_verification_dataset/data.csv", dataset_type="auction"
) -> tuple[any, any, any, any]:
    filename_dataset_assertions(filename, dataset_type)

    def all_joblib_loads() -> tuple[any, any, any, any]:
        best_model_dt = joblib.load(f"../artifacts/best_model_{dataset_type}_dt.pkl")
        best_model_xgb = joblib.load(f"../artifacts/best_model_{dataset_type}_xgb.pkl")
        best_model_svm = joblib.load(f"../artifacts/best_model_{dataset_type}_svm.pkl")
        best_model_knn = joblib.load(f"../artifacts/best_model_{dataset_type}_knn.pkl")
        return (best_model_dt, best_model_xgb, best_model_svm, best_model_knn)

    try:
        (
            best_model_dt,
            best_model_xgb,
            best_model_svm,
            best_model_knn,
        ) = all_joblib_loads()
    except:  # If the load fails, create/re-create the pickle files
        get_all_best_models(filename=filename, dataset_type=dataset_type)
        (
            best_model_dt,
            best_model_xgb,
            best_model_svm,
            best_model_knn,
        ) = all_joblib_loads()

        pass

    return (best_model_dt, best_model_xgb, best_model_svm, best_model_knn)


def get_model_SHAP(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
    model=None,
    model_object=None,
) -> None:
    filename_dataset_assertions(filename, dataset_type)

    assert model in [
        "dt",
        "xgb",
        "svm",
        "knn",
    ], "Model argument must be in ['dt', 'xgb', 'svm', 'knn']."

    (X_train, y_train, X_val, y_val, X_test, y_test) = data_loading(
        filename, dataset_type
    )

    if model == "svm":
        explainer = shap.KernelExplainer(model_object._predict_proba_lr, shap.sample(X_train, 100))
        shap_values = explainer.shap_values(X_test, nsamples=100)
    elif model == "knn":
        explainer = shap.KernelExplainer(model_object.predict_proba, shap.sample(X_train, 100))
        shap_values = explainer.shap_values(X_test, nsamples=100)
    else:
        explainer = shap.Explainer(model_object, X_train)
        shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test)

    plt.savefig(f"../outputs/SHAP/{model}_SHAP_{dataset_type}.png")
    plt.clf()


def get_all_models_SHAP(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
) -> None:
    filename_dataset_assertions(filename, dataset_type)

    best_model_dt, best_model_xgb, best_model_svm, best_model_knn = load_models(
        filename=filename,
        dataset_type=dataset_type,
    )

    model_object_list = [best_model_dt, best_model_xgb, best_model_svm[0], best_model_knn]
    model_list = ["dt", "xgb", "svm", "knn"]

    for model_object, model in zip(model_object_list, model_list):
        get_model_SHAP(
            filename=filename,
            dataset_type=dataset_type,
            model=model,
            model_object=model_object,
        )
