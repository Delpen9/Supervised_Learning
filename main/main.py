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


def decision_tree_metrics(
    X_train, y_train, X_test, y_test
) -> tuple[float, float, float, float]:
    best_model = tune_decision_tree(X_train, y_train, n_iter_search=20)

    test_accuracy = best_model.score(X_test, y_test)
    train_accuracy = best_model.score(X_train, y_train)

    y_prob = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)

    y_prob = best_model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_prob)

    return test_accuracy, test_auc, train_accuracy, train_auc


def knn_metrics(X_train, y_train, X_test, y_test) -> tuple[float, float, float, float]:
    best_model = tune_knn(X_train, y_train, n_iter_search=20)

    test_accuracy = best_model.score(X_test, y_test)
    train_accuracy = best_model.score(X_train, y_train)

    y_prob = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)

    y_prob = best_model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_prob)

    return test_accuracy, test_auc, train_accuracy, train_auc


def neural_network_metrics(
    train_loader, val_loader, test_loader, input_size, num_epochs
) -> tuple[float, float, float, float]:
    best_model = tune_neural_network(train_loader, val_loader, input_size, num_epochs)

    test_auc, test_accuracy = evaluate_model(best_model, test_loader)
    train_auc, train_accuracy = evaluate_model(best_model, train_loader)

    return test_accuracy, test_auc, train_accuracy, train_auc


def svm_metrics(X_train, y_train, X_test, y_test) -> tuple[float, float, float, float]:
    best_model, best_model_prob = tune_svm(X_train, y_train, n_iter_search=5)

    test_accuracy = best_model.score(X_test, y_test)
    train_accuracy = best_model.score(X_train, y_train)

    y_prob = best_model_prob.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)

    y_prob = best_model_prob.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_prob)

    return test_accuracy, test_auc, train_accuracy, train_auc


def xgboost_metrics(
    X_train, y_train, X_test, y_test
) -> tuple[float, float, float, float]:
    best_model = tune_xgboost(X_train, y_train, n_iter_search=20)

    test_accuracy = best_model.score(X_test, y_test)
    train_accuracy = best_model.score(X_train, y_train)

    y_prob = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)

    y_prob = best_model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_prob)

    return test_accuracy, test_auc, train_accuracy, train_auc


def get_auction_verification_model_metrics(
    filename="../data/auction_verification_dataset/data.csv",
) -> None:
    data = pd.read_csv(filename)

    train_val_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42)

    # For use in scikit-learn models
    X_train = train_df.iloc[:, :-2].copy()
    y_train = train_df.iloc[:, -2].copy().astype(int)

    X_val = val_df.iloc[:, :-2].copy()
    y_val = val_df.iloc[:, -2].copy().astype(int)

    X_test = test_df.iloc[:, :-2].copy()
    y_test = test_df.iloc[:, -2].copy().astype(int)

    # For use in PyTorch model
    train_dataset = TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val.values, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test.values, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Get metrics
    start_time = time.time()
    (
        dt_test_accuracy,
        dt_test_auc,
        dt_train_accuracy,
        dt_train_auc,
    ) = decision_tree_metrics(X_train, y_train, X_test, y_test)
    end_time = time.time()
    dt_elapsed_time = end_time - start_time

    #############
    start_time = time.time()
    knn_test_accuracy, knn_test_auc, knn_train_accuracy, knn_train_auc = knn_metrics(
        X_train, y_train, X_test, y_test
    )
    end_time = time.time()
    knn_elapsed_time = end_time - start_time

    #############
    start_time = time.time()
    input_size, num_epochs = X_train.shape[1], 500
    (
        nn_test_accuracy,
        nn_test_auc,
        nn_train_accuracy,
        nn_train_auc,
    ) = neural_network_metrics(
        train_loader, val_loader, test_loader, input_size, num_epochs
    )
    end_time = time.time()
    nn_elapsed_time = end_time - start_time

    #############
    start_time = time.time()
    svm_test_accuracy, svm_test_auc, svm_train_accuracy, svm_train_auc = svm_metrics(
        X_train, y_train, X_test, y_test
    )
    end_time = time.time()
    svm_elapsed_time = end_time - start_time

    #############
    start_time = time.time()
    (
        xgb_test_accuracy,
        xgb_test_auc,
        xgb_train_accuracy,
        xgb_train_auc,
    ) = xgboost_metrics(X_train, y_train, X_test, y_test)
    end_time = time.time()
    xgb_elapsed_time = end_time - start_time

    # Create Outputs
    accuracy_auc_metrics_np = np.array(
        [
            [
                "Decision Tree",
                dt_test_accuracy,
                dt_test_auc,
                dt_train_accuracy,
                dt_train_auc,
                dt_elapsed_time,
            ],
            [
                "K-Nearest Neighbors",
                knn_test_accuracy,
                knn_test_auc,
                knn_train_accuracy,
                knn_train_auc,
                knn_elapsed_time,
            ],
            [
                "Neural Network",
                nn_test_accuracy,
                nn_test_auc,
                nn_train_accuracy,
                nn_train_auc,
                nn_elapsed_time,
            ],
            [
                "Support Vector Machine",
                svm_test_accuracy,
                svm_test_auc,
                svm_train_accuracy,
                svm_train_auc,
                svm_elapsed_time,
            ],
            [
                "XGBoost",
                xgb_test_accuracy,
                xgb_test_auc,
                xgb_train_accuracy,
                xgb_train_auc,
                xgb_elapsed_time,
            ],
        ]
    )
    accuracy_auc_metrics_df = pd.DataFrame(
        accuracy_auc_metrics_np,
        columns=["Model", "Test Accuracy", "Test AUC", "Train Accuracy", "Train AUC", "Time to Train (TTT) [seconds]"],
    )

    headers = accuracy_auc_metrics_df.columns.values.tolist()
    table = tabulate(accuracy_auc_metrics_df, headers, tablefmt="grid", showindex=False)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    plt.text(0, 1, table, size=12, family="monospace")

    plt.tight_layout()
    plt.savefig("../outputs/auction_verification_model_accuracy_auc.png", dpi=300)


if __name__ == "__main__":
    get_auction_verification_model_metrics()
