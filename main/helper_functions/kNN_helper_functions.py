# Standard Data Science Libraries
import numpy as np
import pandas as pd

# Model Training
from sklearn.neighbors import KNeighborsClassifier

# Modeling Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

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


def data_loading_knn(
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
        
        # One-hot encode the "Course" column
        data = pd.get_dummies(data, columns=["Course"], prefix=["Course"])
        
        data["Target"] = data["Target"].replace(
            {"Graduate": 0, "Dropout": 1, "Enrolled": 2}
        )
    
    # Split the data
    train_val_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42)

    # Separate the features and target variable for each dataset type
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

        scaler = StandardScaler()

        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    return (X_train, y_train, X_val, y_val, X_test, y_test)

def get_pre_processed_performance_by_value_of_k_knn(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
) -> None:
    np.random.seed(1234)

    filename_dataset_assertions(filename, dataset_type)

    (X_train, y_train, X_val, y_val, X_test, y_test) = data_loading_knn(
        filename, dataset_type
    )

    k_values = np.arange(1, 21).astype(int)

    train_accuracies = []
    test_accuracies = []
    train_aucs = []
    test_aucs = []
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        if dataset_type=="auction":
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            y_train_prob = model.predict_proba(X_train)[:, 1]
            y_test_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_train_pred = model.predict(np.array(X_train))
            y_test_pred = model.predict(np.array(X_test))

            y_train_prob = model.predict_proba(np.array(X_train))
            y_test_prob = model.predict_proba(np.array(X_test))

        train_accuracies.append(accuracy_score(y_train, y_train_pred))
        test_accuracies.append(accuracy_score(y_test, y_test_pred))

        y_train_bin = label_binarize(y_train, classes=[0, 1, 2])
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

        train_aucs.append(roc_auc_score(y_train_bin, y_train_prob, multi_class="ovo", average="macro"))
        test_aucs.append(roc_auc_score(y_test_bin, y_test_prob, multi_class="ovo", average="macro"))

    plt.figure(figsize=(10, 6))

    plt.plot(k_values, train_accuracies, marker="o", label="Train Accuracy")
    plt.plot(k_values, test_accuracies, marker="o", label="Test Accuracy")
    plt.plot(k_values, train_aucs, marker="o", label="Train AUC")
    plt.plot(k_values, test_aucs, marker="o", label="Test AUC")

    plt.xticks(range(int(min(k_values)), int(max(k_values)) + 1))

    plt.xlabel("K Neighbors")
    plt.ylabel("Performance")
    plt.title("kNN \n Performance as a function of K-Neighbors \n Pre-processed Data")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"../outputs/kNN/pre_processed_k_Neighbors_Performance_{dataset_type}_knn.png"
    )

def multi_class_roc_auc(y_true, y_prob, average="macro"):
    return roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)

def tune_knn_with_pre_processing(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
    n_iter_search=5,
    cv=5
) -> None:
    
    filename_dataset_assertions(filename, dataset_type)

    (X_train, y_train, X_val, y_val, X_test, y_test) = data_loading_knn(
        filename, dataset_type
    )

    param_dist = {
        "n_neighbors": np.arange(1, 51),
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": np.arange(1, 51),
        "p": [1, 2],  # 1: Manhattan distance, 2: Euclidean distance
    }

    knn = KNeighborsClassifier()

    random_search = RandomizedSearchCV(
        knn,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        cv=cv,
        scoring=make_scorer(multi_class_roc_auc, needs_proba=True, average="macro"),
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)

    # Calculate accuracy
    accuracy = random_search.best_estimator_.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Calculate AUC
    y_prob = random_search.best_estimator_.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_prob)
    print(f"Test AUC: {auc:.4f}")
