# Ignore warnings
import warnings

warnings.simplefilter(action="ignore", category=Warning)

# Standard Libraries
import numpy as np
import pandas as pd

# Modeling Libraries
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Model Evaluation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score


def tune_svm(X, y, n_iter_search=50, cv=5, max_iteration=10000):
    """
    Tune the hyperparameters of a Linear SVM classifier using randomized search.

    Parameters:
    - X : array-like, shape (n_samples, n_features)
          Training input samples.
    - y : array-like, shape (n_samples)
          Target values.
    - n_iter_search : int, default=10
          Number of parameter settings that are sampled.
          Trade-off between runtime vs quality of the solution.

    Returns:
    - best_estimator_ : LinearSVC
          Estimator that was chosen by the search.
    """
    param_dist = {
        "C": np.logspace(-4, 4, 20),
        "penalty": ["l1", "l2"],
        "loss": ["hinge", "squared_hinge"],
        "dual": [
            True,
            False,
        ],
        "class_weight": [None, "balanced"],  # For imbalance
        "multi_class": [
            "ovr",
            "crammer_singer",
        ],
        "fit_intercept": [
            True,
            False,
        ],
        "intercept_scaling": np.logspace(-4, 4, 20),
    }

    classifier = LinearSVC(max_iter=max_iteration)

    random_search = RandomizedSearchCV(
        classifier,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    random_search.fit(X, y)

    best_estimator_prob_ = CalibratedClassifierCV(random_search.best_estimator_, method='sigmoid', cv='prefit')
    best_estimator_prob_.fit(X, y)

    return random_search.best_estimator_, best_estimator_prob_


if __name__ == "__main__":
    data = pd.read_csv("../../data/auction_verification_dataset/data.csv")

    X = data.iloc[:, :-2].copy()
    y = data.iloc[:, -2].copy().astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    best_model, best_model_prob = tune_svm(X_train, y_train, n_iter_search=5)

    # Calculate accuracy
    accuracy = best_model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Calculate AUC
    y_prob = best_model_prob.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"Test AUC: {auc:.4f}")
