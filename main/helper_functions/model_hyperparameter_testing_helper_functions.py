# Standard Data Science Libraries
import numpy as np
import pandas as pd

# Model Training
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# Modeling Libraries
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV

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
import itertools

# Saving Models
import joblib

# Helper Functions
from helper_functions.model_training_evaluation_helper_functions import (
    get_all_best_models,
)

from models.NeuralNetwork import (
    tune_neural_network,
    evaluate_model,
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


def compare_svm_kernels(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
) -> None:
    np.random.seed(1234)

    filename_dataset_assertions(filename, dataset_type)

    (X_train, y_train, X_val, y_val, X_test, y_test) = data_loading(
        filename, dataset_type
    )

    kernels = ["linear", "poly", "rbf", "sigmoid"]
    C_values = [0.1, 1, 10, 100]
    results = []

    for kernel in kernels:
        for C in C_values:
            svm = SVC(kernel=kernel, C=C, probability=True, random_state=42)
            svm_prob = CalibratedClassifierCV(svm, method="sigmoid", cv="prefit")
            svm.fit(X_train, y_train)
            svm_prob.fit(X_train, y_train)

            y_train_pred = svm.predict(X_train)
            y_test_pred = svm.predict(X_test)

            if dataset_type == "auction":
                y_train_prob = svm_prob.predict_proba(X_train)[:, 1]
                y_test_prob = svm_prob.predict_proba(X_test)[:, 1]
            else:
                y_train_prob = svm_prob.predict_proba(X_train)
                y_test_prob = svm_prob.predict_proba(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            train_auc = roc_auc_score(
                y_train, y_train_prob, multi_class="ovr", average="macro"
            )
            test_auc = roc_auc_score(
                y_test, y_test_prob, multi_class="ovr", average="macro"
            )

            result = {
                "Kernel": kernel,
                "C Parameter": C,
                "Train Accuracy": train_accuracy,
                "Test Accuracy": test_accuracy,
                "Train AUC": train_auc,
                "Test AUC": test_auc,
            }
            results.append(result)
            print(result)

    results_df = pd.DataFrame(results)

    headers = results_df.columns.values.tolist()
    table = tabulate(results_df, headers, tablefmt="grid", showindex=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    plt.text(0, 1, table, size=12, family="monospace")

    plt.tight_layout()
    plt.savefig(
        f"../outputs/Hyperparameter_Testing/svm_kernel_testing_{dataset_type}.png",
        dpi=300,
    )


def ccp_decision_tree_performance_graph(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
) -> None:
    np.random.seed(1234)

    filename_dataset_assertions(filename, dataset_type)

    (X_train, y_train, X_val, y_val, X_test, y_test) = data_loading(
        filename, dataset_type
    )

    if dataset_type=="auction":
        ccp_alphas = np.linspace(0, 0.05, 10)
    else:
        ccp_alphas = np.linspace(0, 0.25, 10)

    train_accuracies = []
    test_accuracies = []
    train_aucs = []
    test_aucs = []
    for ccp_alpha in ccp_alphas:
        model = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=1234)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        if dataset_type=="auction":
            y_train_prob = model.predict_proba(X_train)[:, 1]
            y_test_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_train_prob = model.predict_proba(X_train)
            y_test_prob = model.predict_proba(X_test)

        train_accuracies.append(accuracy_score(y_train, y_train_pred))
        test_accuracies.append(accuracy_score(y_test, y_test_pred))

        train_aucs.append(roc_auc_score(y_train, y_train_prob, multi_class="ovr", average="macro"))
        test_aucs.append(roc_auc_score(y_test, y_test_prob, multi_class="ovr", average="macro"))

    plt.figure(figsize=(10, 6))

    plt.plot(ccp_alphas, train_accuracies, marker="o", label="Train Accuracy")
    plt.plot(ccp_alphas, test_accuracies, marker="o", label="Test Accuracy")
    plt.plot(ccp_alphas, train_aucs, marker="o", label="Train AUC")
    plt.plot(ccp_alphas, test_aucs, marker="o", label="Test AUC")

    plt.xlabel("CCP Alpha")
    plt.ylabel("Performance")
    plt.title("Decision Tree \n Performance as a function of CCP Alpha")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"../outputs/Hyperparameter_Testing/CCP_Alpha_Performance_{dataset_type}_dt.png"
    )

def max_depth_decision_tree_performance_graph(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
) -> None:
    np.random.seed(1234)

    filename_dataset_assertions(filename, dataset_type)

    (X_train, y_train, X_val, y_val, X_test, y_test) = data_loading(
        filename, dataset_type
    )

    max_depths = np.arange(1, 21)

    train_accuracies = []
    test_accuracies = []
    train_aucs = []
    test_aucs = []
    for max_depth in max_depths:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=1234)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        if dataset_type=="auction":
            y_train_prob = model.predict_proba(X_train)[:, 1]
            y_test_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_train_prob = model.predict_proba(X_train)
            y_test_prob = model.predict_proba(X_test)

        train_accuracies.append(accuracy_score(y_train, y_train_pred))
        test_accuracies.append(accuracy_score(y_test, y_test_pred))

        train_aucs.append(roc_auc_score(y_train, y_train_prob, multi_class="ovr", average="macro"))
        test_aucs.append(roc_auc_score(y_test, y_test_prob, multi_class="ovr", average="macro"))

    plt.figure(figsize=(10, 6))

    plt.plot(max_depths, train_accuracies, marker="o", label="Train Accuracy")
    plt.plot(max_depths, test_accuracies, marker="o", label="Test Accuracy")
    plt.plot(max_depths, train_aucs, marker="o", label="Train AUC")
    plt.plot(max_depths, test_aucs, marker="o", label="Test AUC")

    plt.xlabel("Max Depth")
    plt.ylabel("Performance")
    plt.title("Decision Tree \n Performance as a function of Max Depth")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"../outputs/Hyperparameter_Testing/Max_Depth_Performance_{dataset_type}_dt.png"
    )

def min_samples_per_leaf_decision_tree_performance_graph(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
) -> None:
    np.random.seed(1234)

    filename_dataset_assertions(filename, dataset_type)

    (X_train, y_train, X_val, y_val, X_test, y_test) = data_loading(
        filename, dataset_type
    )

    min_samples = np.arange(1, 200, 5)

    train_accuracies = []
    test_accuracies = []
    train_aucs = []
    test_aucs = []
    for min_sample in min_samples:
        model = DecisionTreeClassifier(min_samples_leaf=min_sample, random_state=1234)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        if dataset_type=="auction":
            y_train_prob = model.predict_proba(X_train)[:, 1]
            y_test_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_train_prob = model.predict_proba(X_train)
            y_test_prob = model.predict_proba(X_test)

        train_accuracies.append(accuracy_score(y_train, y_train_pred))
        test_accuracies.append(accuracy_score(y_test, y_test_pred))

        train_aucs.append(roc_auc_score(y_train, y_train_prob, multi_class="ovr", average="macro"))
        test_aucs.append(roc_auc_score(y_test, y_test_prob, multi_class="ovr", average="macro"))

    plt.figure(figsize=(10, 6))

    plt.plot(min_samples, train_accuracies, marker="o", label="Train Accuracy")
    plt.plot(min_samples, test_accuracies, marker="o", label="Test Accuracy")
    plt.plot(min_samples, train_aucs, marker="o", label="Train AUC")
    plt.plot(min_samples, test_aucs, marker="o", label="Test AUC")

    plt.xlabel("Minimum Samples per Leaf")
    plt.ylabel("Performance")
    plt.title("Decision Tree \n Performance as a function of Minimum Samples per Leaf")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"../outputs/Hyperparameter_Testing/Min_Samples_per_Leaf_Performance_{dataset_type}_dt.png"
    )

def min_impurity_decrease_decision_tree_performance_graph(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
) -> None:
    np.random.seed(1234)

    filename_dataset_assertions(filename, dataset_type)

    (X_train, y_train, X_val, y_val, X_test, y_test) = data_loading(
        filename, dataset_type
    )

    min_impurity_decreases = np.arange(0.0, 0.005, 0.0001)

    train_accuracies = []
    test_accuracies = []
    train_aucs = []
    test_aucs = []
    for min_impurity_decrease in min_impurity_decreases:
        model = DecisionTreeClassifier(min_impurity_decrease=min_impurity_decrease, random_state=1234)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        if dataset_type=="auction":
            y_train_prob = model.predict_proba(X_train)[:, 1]
            y_test_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_train_prob = model.predict_proba(X_train)
            y_test_prob = model.predict_proba(X_test)

        train_accuracies.append(accuracy_score(y_train, y_train_pred))
        test_accuracies.append(accuracy_score(y_test, y_test_pred))

        train_aucs.append(roc_auc_score(y_train, y_train_prob, multi_class="ovr", average="macro"))
        test_aucs.append(roc_auc_score(y_test, y_test_prob, multi_class="ovr", average="macro"))

    plt.figure(figsize=(10, 6))

    plt.plot(min_impurity_decreases, train_accuracies, marker="o", label="Train Accuracy")
    plt.plot(min_impurity_decreases, test_accuracies, marker="o", label="Test Accuracy")
    plt.plot(min_impurity_decreases, train_aucs, marker="o", label="Train AUC")
    plt.plot(min_impurity_decreases, test_aucs, marker="o", label="Test AUC")

    plt.xlabel("Minimum Impurity Decrease")
    plt.ylabel("Performance")
    plt.title("Decision Tree \n Performance as a function of Minimum Impurity Decrease")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"../outputs/Hyperparameter_Testing/Min_impurity_decrease_Performance_{dataset_type}_dt.png"
    )


def get_performance_by_value_of_k_knn(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
) -> None:
    np.random.seed(1234)

    filename_dataset_assertions(filename, dataset_type)

    (X_train, y_train, X_val, y_val, X_test, y_test) = data_loading(
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

        train_aucs.append(roc_auc_score(y_train, y_train_prob, multi_class="ovr", average="macro"))
        test_aucs.append(roc_auc_score(y_test, y_test_prob, multi_class="ovr", average="macro"))

    plt.figure(figsize=(10, 6))

    plt.plot(k_values, train_accuracies, marker="o", label="Train Accuracy")
    plt.plot(k_values, test_accuracies, marker="o", label="Test Accuracy")
    plt.plot(k_values, train_aucs, marker="o", label="Train AUC")
    plt.plot(k_values, test_aucs, marker="o", label="Test AUC")

    plt.xticks(range(int(min(k_values)), int(max(k_values)) + 1))

    plt.xlabel("K Neighbors")
    plt.ylabel("Performance")
    plt.title("kNN \n Performance as a function of K-Neighbors")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"../outputs/Hyperparameter_Testing/k_Neighbors_Performance_{dataset_type}_knn.png"
    )

def get_neural_network_performance_by_learning_rate(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
) -> None:
    np.random.seed(1234)

    filename_dataset_assertions(filename, dataset_type)

    (X_train, y_train, X_val, y_val, X_test, y_test) = data_loading(
        filename, dataset_type
    )

    learning_rates = np.logspace(2, -6, 9)

    val_dataset = TensorDataset(
        torch.tensor(X_val.values, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test.values, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.float32),
    )
    train_dataset = TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32),
    )
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    train_accuracies = []
    test_accuracies = []
    train_aucs = []
    test_aucs = []
    for learning_rate in learning_rates:
        if dataset_type == "dropout":
            input_size, num_epochs, multiclass, num_classes = (
                X_train.shape[1],
                100,
                True,
                3,
            )
        elif dataset_type == "auction":
            input_size, num_epochs, multiclass, num_classes = (
                X_train.shape[1],
                100,
                False,
                2,
            )

        best_model, _, _ = tune_neural_network(
            train_loader, val_loader,
            input_size,
            num_epochs, learning_rate,
            multiclass, num_classes
        )

        train_auc, train_accuracy = evaluate_model(best_model, train_loader, num_classes=num_classes)
        test_auc, test_accuracy = evaluate_model(best_model, test_loader, num_classes=num_classes)

        train_accuracies.append(train_accuracy)
        train_aucs.append(train_auc)
        test_accuracies.append(test_accuracy)
        test_aucs.append(test_auc)

    plt.figure(figsize=(10, 6))

    plt.plot(learning_rates, train_accuracies, marker="o", label="Train Accuracy")
    plt.plot(learning_rates, test_accuracies, marker="o", label="Test Accuracy")
    plt.plot(learning_rates, train_aucs, marker="o", label="Train AUC")
    plt.plot(learning_rates, test_aucs, marker="o", label="Test AUC")

    plt.xticks(learning_rates)
    plt.xscale("log")

    plt.xlabel("Learning Rate")
    plt.ylabel("Performance")
    plt.title("Neural Network \n Performance as a function of Learning Rate (100 Epochs)")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"../outputs/Hyperparameter_Testing/Learning_Rate_Performance_{dataset_type}_nn.png"
    )

def get_performance_by_value_of_c_svm(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
) -> None:
    np.random.seed(1234)

    filename_dataset_assertions(filename, dataset_type)

    (X_train, y_train, X_val, y_val, X_test, y_test) = data_loading(
        filename, dataset_type
    )

    c_values = np.logspace(-4, 4, 20)

    train_accuracies = []
    test_accuracies = []
    train_aucs = []
    test_aucs = []
    for c in c_values:
        model = LinearSVC(C=c)
        model_prob_ = CalibratedClassifierCV(
            model, method="sigmoid", cv="prefit"
        )

        model.fit(X_train, y_train)
        model_prob_.fit(X_train, y_train)

        if dataset_type=="auction":
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            y_train_prob = model_prob_.predict_proba(X_train)[:, 1]
            y_test_prob = model_prob_.predict_proba(X_test)[:, 1]
        else:
            y_train_pred = model.predict(np.array(X_train))
            y_test_pred = model.predict(np.array(X_test))

            y_train_prob = model_prob_.predict_proba(np.array(X_train))
            y_test_prob = model_prob_.predict_proba(np.array(X_test))

        train_accuracies.append(accuracy_score(y_train, y_train_pred))
        test_accuracies.append(accuracy_score(y_test, y_test_pred))

        train_aucs.append(roc_auc_score(y_train, y_train_prob, multi_class="ovr", average="macro"))
        test_aucs.append(roc_auc_score(y_test, y_test_prob, multi_class="ovr", average="macro"))

    plt.figure(figsize=(10, 6))

    plt.plot(c_values, train_accuracies, marker="o", label="Train Accuracy")
    plt.plot(c_values, test_accuracies, marker="o", label="Test Accuracy")
    plt.plot(c_values, train_aucs, marker="o", label="Train AUC")
    plt.plot(c_values, test_aucs, marker="o", label="Test AUC")

    plt.xticks(range(int(min(c_values)), int(max(c_values)) + 1))
    plt.xscale("log")

    plt.xlabel("C")
    plt.ylabel("Performance")
    plt.title("SVM \n Performance as a function of C")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"../outputs/Hyperparameter_Testing/C_Performance_{dataset_type}_svm.png"
    )

def get_performance_by_value_of_learning_rate_xgboost(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
) -> None:
    np.random.seed(1234)

    filename_dataset_assertions(filename, dataset_type)

    (X_train, y_train, X_val, y_val, X_test, y_test) = data_loading(
        filename, dataset_type
    )

    learning_rates = np.linspace(0.01, 1, 100)

    train_accuracies = []
    test_accuracies = []
    train_aucs = []
    test_aucs = []
    for learning_rate in learning_rates:
        model = xgb.XGBClassifier(
            objective="binary:logistic", use_label_encoder=False, eval_metric="logloss",
            learning_rate=learning_rate
        )

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

        train_aucs.append(roc_auc_score(y_train, y_train_prob, multi_class="ovr", average="macro"))
        test_aucs.append(roc_auc_score(y_test, y_test_prob, multi_class="ovr", average="macro"))

    plt.figure(figsize=(10, 6))

    plt.plot(learning_rates, train_accuracies, marker="o", label="Train Accuracy")
    plt.plot(learning_rates, test_accuracies, marker="o", label="Test Accuracy")
    plt.plot(learning_rates, train_aucs, marker="o", label="Train AUC")
    plt.plot(learning_rates, test_aucs, marker="o", label="Test AUC")

    plt.xticks(range(int(min(learning_rates)), int(max(learning_rates)) + 1))
    if dataset_type=="auction":
        plt.ylim([0.98, 1.0])

    plt.xlabel("Learning Rate")
    plt.ylabel("Performance")
    plt.title("XGBoost \n Performance as a Function of Learning Rate")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"../outputs/Hyperparameter_Testing/Learning_Rate_Performance_{dataset_type}_xgb.png"
    )

def get_performance_by_value_of_max_depth_xgboost(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
) -> None:
    np.random.seed(1234)

    filename_dataset_assertions(filename, dataset_type)

    (X_train, y_train, X_val, y_val, X_test, y_test) = data_loading(
        filename, dataset_type
    )

    max_depths = np.arange(1, 11).astype(int)

    train_accuracies = []
    test_accuracies = []
    train_aucs = []
    test_aucs = []
    for max_depth in max_depths:
        model = xgb.XGBClassifier(
            objective="binary:logistic", use_label_encoder=False, eval_metric="logloss",
            max_depth=max_depth
        )

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

        train_aucs.append(roc_auc_score(y_train, y_train_prob, multi_class="ovr", average="macro"))
        test_aucs.append(roc_auc_score(y_test, y_test_prob, multi_class="ovr", average="macro"))

    plt.figure(figsize=(10, 6))

    plt.plot(max_depths, train_accuracies, marker="o", label="Train Accuracy")
    plt.plot(max_depths, test_accuracies, marker="o", label="Test Accuracy")
    plt.plot(max_depths, train_aucs, marker="o", label="Train AUC")
    plt.plot(max_depths, test_aucs, marker="o", label="Test AUC")

    plt.xticks(range(int(min(max_depths)), int(max(max_depths)) + 1))
    if dataset_type=="auction":
        plt.ylim([0.9, 1.0])

    plt.xlabel("Max Depth")
    plt.ylabel("Performance")
    plt.title("XGBoost \n Performance as a Function of Max Depth")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"../outputs/Hyperparameter_Testing/Max_Depth_Performance_{dataset_type}_xgb.png"
    )

def get_performance_by_value_of_n_estimators_xgboost(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
) -> None:
    np.random.seed(1234)

    filename_dataset_assertions(filename, dataset_type)

    (X_train, y_train, X_val, y_val, X_test, y_test) = data_loading(
        filename, dataset_type
    )

    n_estimator_values = np.arange(10, 1000, 5).astype(int)

    train_accuracies = []
    test_accuracies = []
    train_aucs = []
    test_aucs = []
    for n_estimators in n_estimator_values:
        model = xgb.XGBClassifier(
            objective="binary:logistic", use_label_encoder=False, eval_metric="logloss",
            n_estimators=n_estimators
        )

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

        train_aucs.append(roc_auc_score(y_train, y_train_prob, multi_class="ovr", average="macro"))
        test_aucs.append(roc_auc_score(y_test, y_test_prob, multi_class="ovr", average="macro"))

    plt.figure(figsize=(10, 6))

    plt.plot(n_estimator_values, train_accuracies, marker="o", label="Train Accuracy")
    plt.plot(n_estimator_values, test_accuracies, marker="o", label="Test Accuracy")
    plt.plot(n_estimator_values, train_aucs, marker="o", label="Train AUC")
    plt.plot(n_estimator_values, test_aucs, marker="o", label="Test AUC")

    plt.xlabel("N Estimators")
    plt.xlim([0, 200])
    plt.ylabel("Performance")
    plt.yscale("log")
    plt.title("XGBoost \n Performance as a Function of N Estimators")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"../outputs/Hyperparameter_Testing/Num_Estimators_Performance_{dataset_type}_xgb.png"
    )

def get_neural_network_performance_by_hidden_dimensions(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
) -> None:
    np.random.seed(1234)

    filename_dataset_assertions(filename, dataset_type)

    (X_train, y_train, X_val, y_val, X_test, y_test) = data_loading(
        filename, dataset_type
    )

    dimensions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    dimension_combinations = list(itertools.combinations(dimensions, 2))

    val_dataset = TensorDataset(
        torch.tensor(X_val.values, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test.values, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.float32),
    )
    train_dataset = TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32),
    )
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    train_accuracies = []
    test_accuracies = []
    train_aucs = []
    test_aucs = []
    for dimension_combination in dimension_combinations:
        if dataset_type == "dropout":
            input_size, num_epochs, learning_rate, multiclass, num_classes, hidden_dimension_1, hidden_dimension_2 = (
                X_train.shape[1],
                100,
                1e-4,
                True,
                3,
                dimension_combination[0],
                dimension_combination[1],
            )
        elif dataset_type == "auction":
            input_size, num_epochs, learning_rate, multiclass, num_classes, hidden_dimension_1, hidden_dimension_2 = (
                X_train.shape[1],
                100,
                1e-3,
                False,
                2,
                dimension_combination[0],
                dimension_combination[1],
            )

        best_model, _, _ = tune_neural_network(
            train_loader, val_loader,
            input_size,
            num_epochs, learning_rate,
            multiclass, num_classes,
            hidden_dimension_1, hidden_dimension_2
        )

        train_auc, train_accuracy = evaluate_model(best_model, train_loader, num_classes=num_classes)
        test_auc, test_accuracy = evaluate_model(best_model, test_loader, num_classes=num_classes)

        train_accuracies.append(train_accuracy)
        train_aucs.append(train_auc)
        test_accuracies.append(test_accuracy)
        test_aucs.append(test_auc)

    dimensions_combination_performance = pd.DataFrame([], columns=[
        "Dimension 1",
        "Dimension 2",
        "Train Accuracy",
        "Train AUC",
        "Test Accuracy",
        "Test AUC",
    ])

    dimensions_combination_performance[["Dimension 1", "Dimension 2"]] = dimension_combinations
    dimensions_combination_performance["Train Accuracy"] = train_accuracies
    dimensions_combination_performance["Train AUC"] = train_aucs
    dimensions_combination_performance["Test Accuracy"] = test_accuracies
    dimensions_combination_performance["Test AUC"] = test_aucs

    dimensions_combination_performance.to_csv(f"../outputs/Hyperparameter_Testing/neural_network_dimension_combinations_performance_{dataset_type}.csv", index=False)

def get_neural_network_performance_heatmap(
    dataset_type="auction",
    set_version="test",
    metric="AUC",
) -> None:
    assert dataset_type in ["auction", "dropout"], "dataset_type must be either 'auction' or 'dropout'"
    assert set_version in ["test", "train"], "set_version must be in 'test' or 'train'"
    assert metric in ["AUC", "Accuracy"], "metric must in 'AUC' or 'Accuracy'"

    dimension_performance_df = pd.read_csv(f"../outputs/Hyperparameter_Testing/neural_network_dimension_combinations_performance_{dataset_type}.csv")

    heatmap_data = dimension_performance_df.pivot(index='Dimension 2', columns='Dimension 1', values=f'Test {metric}')

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis')

    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title(f'{dataset_type.capitalize()}: Heatmap of Test {metric}')

    plt.savefig(
        f"../outputs/Hyperparameter_Testing/neural_network_dimension_heatmap_{dataset_type.capitalize()}_{set_version}_{metric}.png"
    )