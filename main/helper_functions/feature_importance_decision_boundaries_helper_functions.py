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

# Helper Functions
from helper_functions.model_training_evaluation_helper_functions import get_all_best_models


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

    return None
