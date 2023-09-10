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


def decision_tree_metrics(
    X_train, y_train, X_test, y_test, multiclass=False, n_iter_search=100
) -> tuple[float, float, float, float]:
    best_model = tune_decision_tree(X_train, y_train, n_iter_search=n_iter_search)

    test_accuracy = best_model.score(X_test, y_test)
    train_accuracy = best_model.score(X_train, y_train)

    if multiclass == False:
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = best_model.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

    if multiclass == False:
        y_prob = best_model.predict_proba(X_train)[:, 1]
    else:
        y_prob = best_model.predict_proba(X_train)
    train_auc = roc_auc_score(y_train, y_prob, multi_class="ovr", average="macro")

    return test_accuracy, test_auc, train_accuracy, train_auc


def knn_metrics(
    X_train, y_train, X_test, y_test, multiclass=False, n_iter_search=20
) -> tuple[float, float, float, float]:
    best_model = tune_knn(X_train, y_train, n_iter_search=n_iter_search)

    test_accuracy = best_model.score(X_test, y_test)
    train_accuracy = best_model.score(X_train, y_train)

    if multiclass == False:
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = best_model.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

    if multiclass == False:
        y_prob = best_model.predict_proba(X_train)[:, 1]
    else:
        y_prob = best_model.predict_proba(X_train)
    train_auc = roc_auc_score(y_train, y_prob, multi_class="ovr", average="macro")

    return test_accuracy, test_auc, train_accuracy, train_auc


def neural_network_metrics(
    train_loader,
    val_loader,
    test_loader,
    input_size,
    num_epochs,
    learning_rate,
    multiclass,
    num_classes,
) -> tuple[float, float, float, float]:
    best_model = tune_neural_network(
        train_loader,
        val_loader,
        input_size,
        num_epochs,
        learning_rate,
        multiclass,
        num_classes,
    )

    test_auc, test_accuracy = evaluate_model(best_model, test_loader, num_classes)
    train_auc, train_accuracy = evaluate_model(best_model, train_loader, num_classes)

    return test_accuracy, test_auc, train_accuracy, train_auc


def svm_metrics(
    X_train, y_train, X_test, y_test, multiclass=False, n_iter_search=5
) -> tuple[float, float, float, float]:
    best_model, best_model_prob = tune_svm(
        X_train, y_train, n_iter_search=n_iter_search
    )

    test_accuracy = best_model.score(X_test, y_test)
    train_accuracy = best_model.score(X_train, y_train)

    if multiclass == False:
        y_prob = best_model_prob.predict_proba(X_test)[:, 1]
    else:
        y_prob = best_model_prob.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

    if multiclass == False:
        y_prob = best_model_prob.predict_proba(X_train)[:, 1]
    else:
        y_prob = best_model_prob.predict_proba(X_train)
    train_auc = roc_auc_score(y_train, y_prob, multi_class="ovr", average="macro")

    return test_accuracy, test_auc, train_accuracy, train_auc


def xgboost_metrics(
    X_train, y_train, X_test, y_test, multiclass=False, n_iter_search=20
) -> tuple[float, float, float, float]:
    best_model = tune_xgboost(X_train, y_train, n_iter_search=n_iter_search)

    test_accuracy = best_model.score(X_test, y_test)
    train_accuracy = best_model.score(X_train, y_train)

    if multiclass == False:
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = best_model.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

    if multiclass == False:
        y_prob = best_model.predict_proba(X_train)[:, 1]
    else:
        y_prob = best_model.predict_proba(X_train)
    train_auc = roc_auc_score(y_train, y_prob, multi_class="ovr", average="macro")

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
    input_size, num_epochs, learning_rate, multiclass, num_classes = (
        X_train.shape[1],
        500,
        0.001,
        False,
        2,
    )
    (
        nn_test_accuracy,
        nn_test_auc,
        nn_train_accuracy,
        nn_train_auc,
    ) = neural_network_metrics(
        train_loader,
        val_loader,
        test_loader,
        input_size,
        num_epochs,
        learning_rate,
        multiclass,
        num_classes,
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
                dt_elapsed_time / 100,
            ],
            [
                "K-Nearest Neighbors",
                knn_test_accuracy,
                knn_test_auc,
                knn_train_accuracy,
                knn_train_auc,
                knn_elapsed_time / 20,
            ],
            [
                "Neural Network",
                nn_test_accuracy,
                nn_test_auc,
                nn_train_accuracy,
                nn_train_auc,
                nn_elapsed_time / 500,
            ],
            [
                "Support Vector Machine",
                svm_test_accuracy,
                svm_test_auc,
                svm_train_accuracy,
                svm_train_auc,
                svm_elapsed_time / 5,
            ],
            [
                "XGBoost",
                xgb_test_accuracy,
                xgb_test_auc,
                xgb_train_accuracy,
                xgb_train_auc,
                xgb_elapsed_time / 20,
            ],
        ]
    )
    accuracy_auc_metrics_df = pd.DataFrame(
        accuracy_auc_metrics_np,
        columns=[
            "Model",
            "Test Accuracy",
            "Test AUC",
            "Train Accuracy",
            "Train AUC",
            "Time to Train (TTT) [seconds / iteration]",
        ],
    )

    headers = accuracy_auc_metrics_df.columns.values.tolist()
    table = tabulate(accuracy_auc_metrics_df, headers, tablefmt="grid", showindex=False)

    fig, ax = plt.subplots(figsize=(15, 3))
    ax.axis("off")
    plt.text(0, 1, table, size=12, family="monospace")

    plt.tight_layout()
    plt.savefig("../outputs/auction_verification_model_accuracy_auc.png", dpi=300)


def get_student_dropout_model_metrics(
    filename="../data/student_dropout_dataset/data.csv",
) -> None:
    data = pd.read_csv(filename, delimiter=";")

    data["Target"] = data["Target"].replace(
        {"Graduate": 0, "Dropout": 1, "Enrolled": 2}
    )

    train_val_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42)

    # For use in scikit-learn models
    X_train = train_df.iloc[:, :-1].copy()
    y_train = train_df.iloc[:, -1].copy().astype(int)

    X_val = val_df.iloc[:, :-1].copy()
    y_val = val_df.iloc[:, -1].copy().astype(int)

    X_test = test_df.iloc[:, :-1].copy()
    y_test = test_df.iloc[:, -1].copy().astype(int)

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
    ) = decision_tree_metrics(X_train, y_train, X_test, y_test, True)
    end_time = time.time()
    dt_elapsed_time = end_time - start_time

    #############
    start_time = time.time()
    knn_test_accuracy, knn_test_auc, knn_train_accuracy, knn_train_auc = knn_metrics(
        X_train, y_train, X_test, y_test, True
    )
    end_time = time.time()
    knn_elapsed_time = end_time - start_time

    #############
    start_time = time.time()
    input_size, num_epochs, learning_rate, multiclass, num_classes = (
        X_train.shape[1],
        500,
        0.001,
        True,
        3,
    )
    (
        nn_test_accuracy,
        nn_test_auc,
        nn_train_accuracy,
        nn_train_auc,
    ) = neural_network_metrics(
        train_loader,
        val_loader,
        test_loader,
        input_size,
        num_epochs,
        learning_rate,
        multiclass,
        num_classes,
    )
    end_time = time.time()
    nn_elapsed_time = end_time - start_time

    #############
    start_time = time.time()
    svm_test_accuracy, svm_test_auc, svm_train_accuracy, svm_train_auc = svm_metrics(
        X_train, y_train, X_test, y_test, True
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
    ) = xgboost_metrics(X_train, y_train, X_test, y_test, True)
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
                dt_elapsed_time / 100,
            ],
            [
                "K-Nearest Neighbors",
                knn_test_accuracy,
                knn_test_auc,
                knn_train_accuracy,
                knn_train_auc,
                knn_elapsed_time / 20,
            ],
            [
                "Neural Network",
                nn_test_accuracy,
                nn_test_auc,
                nn_train_accuracy,
                nn_train_auc,
                nn_elapsed_time / 500,
            ],
            [
                "Support Vector Machine",
                svm_test_accuracy,
                svm_test_auc,
                svm_train_accuracy,
                svm_train_auc,
                svm_elapsed_time / 5,
            ],
            [
                "XGBoost",
                xgb_test_accuracy,
                xgb_test_auc,
                xgb_train_accuracy,
                xgb_train_auc,
                xgb_elapsed_time / 20,
            ],
        ]
    )
    accuracy_auc_metrics_df = pd.DataFrame(
        accuracy_auc_metrics_np,
        columns=[
            "Model",
            "Test Accuracy",
            "Test AUC",
            "Train Accuracy",
            "Train AUC",
            "Time to Train (TTT) [seconds / iteration]",
        ],
    )

    headers = accuracy_auc_metrics_df.columns.values.tolist()
    table = tabulate(accuracy_auc_metrics_df, headers, tablefmt="grid", showindex=False)

    fig, ax = plt.subplots(figsize=(15, 3))
    ax.axis("off")
    plt.text(0, 1, table, size=12, family="monospace")

    plt.tight_layout()
    plt.savefig("../outputs/student_dropout_model_accuracy_auc.png", dpi=300)


def get_performance_curve(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
    model="dt",
    n_iter_search=100,
    y_bounds=[0.0, 1.0],
) -> None:
    filename_dataset_assertions(filename, dataset_type)

    assert model in [
        "dt",
        "xgb",
        "svm",
        "knn",
    ], "Model argument must be in ['dt', 'xgb', 'svm', 'knn']."

    multiclass = True if dataset_type == "dropout" else False

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

    accuracy_per_slice = []
    auc_per_slice = []
    for i in range(10):
        start_index = 0
        end_index = math.ceil(start_index + ((i + 1) / 10) * X_train.shape[0])

        X_train_slice = X_train.iloc[start_index:end_index, :].copy()
        y_train_slice = y_train.iloc[start_index:end_index].copy()

        if model == "dt":
            (
                test_accuracy,
                test_auc,
                _,
                _,
            ) = decision_tree_metrics(
                X_train_slice,
                y_train_slice,
                X_test,
                y_test,
                multiclass=multiclass,
                n_iter_search=n_iter_search,
            )
        elif model == "xgb":
            (
                test_accuracy,
                test_auc,
                _,
                _,
            ) = xgboost_metrics(
                X_train_slice,
                y_train_slice,
                X_test,
                y_test,
                multiclass=multiclass,
                n_iter_search=n_iter_search,
            )
        elif model == "svm":
            (
                test_accuracy,
                test_auc,
                _,
                _,
            ) = svm_metrics(
                X_train_slice,
                y_train_slice,
                X_test,
                y_test,
                multiclass=multiclass,
                n_iter_search=n_iter_search,
            )
        else:
            (
                test_accuracy,
                test_auc,
                _,
                _,
            ) = knn_metrics(
                X_train_slice,
                y_train_slice,
                X_test,
                y_test,
                multiclass=multiclass,
                n_iter_search=n_iter_search,
            )

        accuracy_per_slice.append(test_accuracy)
        auc_per_slice.append(test_auc)

    accuracy_per_slice = np.round(np.array([accuracy_per_slice]), 3)
    auc_per_slice = np.round(np.array([auc_per_slice]), 3)

    percentage_training_set_seen = np.array(
        [rf"{perc}%" for perc in np.arange(10, 110, 10).astype(str)]
    )

    metrics_np = np.vstack(
        (accuracy_per_slice, auc_per_slice, percentage_training_set_seen)
    ).T

    metrics_df = pd.DataFrame(
        metrics_np,
        columns=["Test Accuracy", "Test AUC", "Percentage of Training Set Seen"],
    ).astype({"Test Accuracy": float, "Test AUC": float})

    # Creating the lineplot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x="Percentage of Training Set Seen",
        y="Test Accuracy",
        data=metrics_df,
        label="Test Accuracy",
    )
    sns.lineplot(
        x="Percentage of Training Set Seen",
        y="Test AUC",
        data=metrics_df,
        label="Test AUC",
    )

    plt.ylim(y_bounds[0], y_bounds[1])

    # Adding titles and labels
    model_map_dict = {
        "dt": "Decision Tree",
        "xgb": "XGBoost",
        "svm": "Support Vector Machine",
        "knn": "K-Nearest Neighbors",
    }

    title = f"{model_map_dict[model]} - {dataset_type.capitalize()} Dataset"
    title += "\n Test Accuracy and AUC as a Percentage of the Seen Training Set"
    plt.title(title, fontweight="bold")

    plt.xlabel("Percentage of Training Set Seen")
    plt.ylabel("Metric Value")

    # Display the legend
    plt.legend()

    # Save the plot
    plt.savefig(
        rf"../outputs/{model}_test_acc_auc_performance_curve_{dataset_type}_dataset.png",
        dpi=300,
    )


def get_auction_percentage_curves() -> None:
    # Decision Tree - Auction
    get_performance_curve(
        filename="../data/auction_verification_dataset/data.csv",
        dataset_type="auction",
        model="dt",
        n_iter_search=500,
        y_bounds=[0.75, 1.0],
    )

    # XGBoost - Auction
    get_performance_curve(
        filename="../data/auction_verification_dataset/data.csv",
        dataset_type="auction",
        model="xgb",
        n_iter_search=50,
        y_bounds=[0.9, 1.01],
    )

    # SVM - Auction
    get_performance_curve(
        filename="../data/auction_verification_dataset/data.csv",
        dataset_type="auction",
        model="svm",
        n_iter_search=5,
        y_bounds=[0.5, 1.0],
    )

    # KNN - Auction
    get_performance_curve(
        filename="../data/auction_verification_dataset/data.csv",
        dataset_type="auction",
        model="knn",
        n_iter_search=20,
        y_bounds=[0.85, 1.01],
    )


def get_dropout_percentage_curves() -> None:
    # Decision Tree - Dropout
    get_performance_curve(
        filename="../data/student_dropout_dataset/data.csv",
        dataset_type="dropout",
        model="dt",
        n_iter_search=500,
        y_bounds=[0.5, 0.9],
    )

    # XGBoost - Dropout
    get_performance_curve(
        filename="../data/student_dropout_dataset/data.csv",
        dataset_type="dropout",
        model="xgb",
        n_iter_search=5,
        y_bounds=[0.65, 0.9],
    )

    # SVM - Dropout
    get_performance_curve(
        filename="../data/student_dropout_dataset/data.csv",
        dataset_type="dropout",
        model="svm",
        n_iter_search=1,
        y_bounds=[0.4, 0.9],
    )

    # KNN - Dropout
    get_performance_curve(
        filename="../data/student_dropout_dataset/data.csv",
        dataset_type="dropout",
        model="knn",
        n_iter_search=50,
        y_bounds=[0.5, 0.8],
    )


def neural_network_percentage_curves(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
    epochs=500,
    learning_rate=0.001,
    y_bounds=[0.0, 1.0],
) -> None:
    filename_dataset_assertions(filename, dataset_type)

    multiclass = True if dataset_type == "dropout" else False

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

    val_dataset = TensorDataset(
        torch.tensor(X_val.values, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test.values, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.float32),
    )
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    accuracy_per_slice = []
    auc_per_slice = []
    for i in range(10):
        start_index = 0
        end_index = math.ceil(start_index + ((i + 1) / 10) * X_train.shape[0])

        X_train_slice = X_train.iloc[start_index:end_index, :].copy()
        y_train_slice = y_train.iloc[start_index:end_index].copy()

        train_dataset = TensorDataset(
            torch.tensor(X_train_slice.values, dtype=torch.float32),
            torch.tensor(y_train_slice.values, dtype=torch.float32),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        if dataset_type == "dropout":
            input_size, num_epochs, learning_rate, multiclass, num_classes = (
                X_train_slice.shape[1],
                epochs,
                learning_rate,
                True,
                3,
            )
        elif dataset_type == "auction":
            input_size, num_epochs, learning_rate, multiclass, num_classes = (
                X_train_slice.shape[1],
                epochs,
                learning_rate,
                False,
                2,
            )

        (
            test_accuracy,
            test_auc,
            _,
            _,
        ) = neural_network_metrics(
            train_loader,
            val_loader,
            test_loader,
            input_size,
            num_epochs,
            learning_rate,
            multiclass,
            num_classes,
        )

        accuracy_per_slice.append(test_accuracy)
        auc_per_slice.append(test_auc)

    accuracy_per_slice = np.round(np.array([accuracy_per_slice]), 3)
    auc_per_slice = np.round(np.array([auc_per_slice]), 3)

    percentage_training_set_seen = np.array(
        [rf"{perc}%" for perc in np.arange(10, 110, 10).astype(str)]
    )

    metrics_np = np.vstack(
        (accuracy_per_slice, auc_per_slice, percentage_training_set_seen)
    ).T

    metrics_df = pd.DataFrame(
        metrics_np,
        columns=["Test Accuracy", "Test AUC", "Percentage of Training Set Seen"],
    ).astype({"Test Accuracy": float, "Test AUC": float})

    # Creating the lineplot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x="Percentage of Training Set Seen",
        y="Test Accuracy",
        data=metrics_df,
        label="Test Accuracy",
    )
    sns.lineplot(
        x="Percentage of Training Set Seen",
        y="Test AUC",
        data=metrics_df,
        label="Test AUC",
    )

    plt.ylim(y_bounds[0], y_bounds[1])

    # Adding titles and labels
    title = f"Neural Network - {dataset_type.capitalize()} Dataset"
    title += "\n Test Accuracy and AUC as a Percentage of the Seen Training Set"
    plt.title(title, fontweight="bold")

    plt.xlabel("Percentage of Training Set Seen")
    plt.ylabel("Metric Value")

    # Display the legend
    plt.legend()

    # Save the plot
    plt.savefig(
        rf"../outputs/nn_test_acc_auc_performance_curve_{dataset_type}_dataset.png",
        dpi=300,
    )


def get_neural_network_percentage_curves() -> None:
    neural_network_percentage_curves(
        filename="../data/auction_verification_dataset/data.csv",
        dataset_type="auction",
        epochs=500,
        learning_rate=0.001,
        y_bounds=[0.7, 1.0],
    )

    neural_network_percentage_curves(
        filename="../data/student_dropout_dataset/data.csv",
        dataset_type="dropout",
        epochs=500,
        learning_rate=0.001,
        y_bounds=[0.35, 1.0],
    )


def get_best_model(
    filename="../data/auction_verification_dataset/data.csv",
    dataset_type="auction",
    model="dt",
    n_iter_search=100,
    save_path="../artifacts",
) -> None:
    filename_dataset_assertions(filename, dataset_type)

    assert model in [
        "dt",
        "xgb",
        "svm",
        "knn",
    ], "Model argument must be in ['dt', 'xgb', 'svm', 'knn']."

    save_path += f"/best_model_{dataset_type}_{model}.pkl"

    multiclass = True if dataset_type == "dropout" else False

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

    if model == "dt":
        best_model = tune_decision_tree(X_train, y_train, n_iter_search=n_iter_search)
    elif model == "xgb":
        best_model = tune_xgboost(X_train, y_train, n_iter_search=n_iter_search)
    elif model == "svm":
        best_model = tune_svm(X_train, y_train, n_iter_search=n_iter_search)
    else:
        best_model = tune_knn(X_train, y_train, n_iter_search=n_iter_search)

    joblib.dump(best_model, save_path)


def get_all_best_models(
    filename="../data/auction_verification_dataset/data.csv", dataset_type="auction"
) -> None:
    filename_dataset_assertions(filename, dataset_type)

    models = ["dt", "xgb", "svm", "knn"]

    if dataset_type == "auction":
        model_to_iter_map = {"dt": 2000, "xgb": 200, "svm": 20, "knn": 80}
    else:
        model_to_iter_map = {"dt": 2000, "xgb": 20, "svm": 4, "knn": 200}

    for model in models:
        get_best_model(
            filename=filename,
            dataset_type=dataset_type,
            model=model,
            n_iter_search=model_to_iter_map[model],
            save_path="../artifacts",
        )
