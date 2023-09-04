# Ignore warnings
import warnings

warnings.simplefilter(action="ignore", category=Warning)

# Standard Libraries
import numpy as np
import pandas as pd

# Modeling Libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Model Evaluation
from sklearn.metrics import roc_auc_score, make_scorer


def multi_class_roc_auc(y_true, y_prob, average="macro"):
    return roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)


def tune_knn(X, y, n_iter_search=50, cv=5):
    """
    Tune the hyperparameters of a k-NN classifier using randomized search.

    Parameters:
    - X : array-like, shape (n_samples, n_features)
          Training input samples.
    - y : array-like, shape (n_samples)
          Target values.
    - n_iter_search : int, default=10
          Number of parameter settings that are sampled.
          Trade-off between runtime vs quality of the solution.

    Returns:
    - best_estimator_ : KNeighborsClassifier
          Estimator that was chosen by the search.
    """
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

    random_search.fit(X, y)

    return random_search.best_estimator_


if __name__ == "__main__":
    data = pd.read_csv("../../data/auction_verification_dataset/data.csv")

    X = data.iloc[:, :-2].copy()
    y = data.iloc[:, -2].copy().astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    best_model = tune_knn(X_train, y_train, n_iter_search=20)

    # Calculate accuracy
    accuracy = best_model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Calculate AUC
    y_prob = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"Test AUC: {auc:.4f}")
