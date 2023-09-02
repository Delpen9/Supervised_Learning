# Ignore warnings
import warnings

warnings.simplefilter(action="ignore", category=Warning)

# Standard Libraries
import numpy as np
import pandas as pd

# Modeling Libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Model Evaluation
from sklearn.metrics import roc_auc_score


def tune_decision_tree(X, y, n_iter_search=10, cv=5):
    """
    Tune the hyperparameters of a decision tree using randomized search.

    Parameters:
    - X : array-like, shape (n_samples, n_features)
          Training input samples.
    - y : array-like, shape (n_samples)
          Target values.
    - n_iter_search : int, default=10
          Number of parameter settings that are sampled.
          Trade-off between runtime vs quality of the solution.

    Returns:
    - best_estimator_ : DecisionTreeClassifier
          Estimator that was chosen by the search.
    """
    param_dist = {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [None] + list(np.arange(1, 51)),
        "min_samples_split": np.arange(2, 21),
        "min_samples_leaf": np.arange(1, 21),
        "max_features": [None, "auto", "sqrt", "log2"] + list(np.arange(0.1, 1.1, 0.1)),
        "max_leaf_nodes": [None] + list(np.arange(2, 51)),
        "min_impurity_decrease": np.linspace(0, 0.5, 51),
        "class_weight": [None, "balanced"],
        "ccp_alpha": np.linspace(0, 0.05, 11),
    }

    tree = DecisionTreeClassifier()

    random_search = RandomizedSearchCV(
        tree,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
    )

    random_search.fit(X, y)

    return random_search.best_estimator_


if __name__ == "__main__":
    data = pd.read_csv("../../data/auction_verification_dataset/data.csv")

    X = data.iloc[:, :-2].copy()
    y = data.iloc[:, -2].copy().astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    best_model = tune_decision_tree(X_train, y_train, n_iter_search=20)

    # Calculate accuracy
    accuracy = best_model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Calculate AUC
    y_prob = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"Test AUC: {auc:.4f}")
