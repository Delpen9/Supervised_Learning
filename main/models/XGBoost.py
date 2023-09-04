# Ignore warnings
import warnings

warnings.simplefilter(action="ignore", category=Warning)

# Standard Libraries
import numpy as np
import pandas as pd

# Modeling Libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Model Evaluation
from sklearn.metrics import roc_auc_score, make_scorer


def multi_class_roc_auc(y_true, y_prob, average="macro"):
    return roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)


def tune_xgboost(X, y, n_iter_search=50, cv=5):
    """
    Tune the hyperparameters of an XGBoost classifier using randomized search.

    Parameters:
    - X : array-like, shape (n_samples, n_features)
          Training input samples.
    - y : array-like, shape (n_samples)
          Target values.
    - n_iter_search : int, default=10
          Number of parameter settings that are sampled.
          Trade-off between runtime vs quality of the solution.

    Returns:
    - best_estimator_ : XGBClassifier
          Estimator that was chosen by the search.
    """
    param_dist = {
        "learning_rate": np.linspace(0.01, 1, 100),
        "n_estimators": np.arange(10, 1000, 5),
        "subsample": np.linspace(0.1, 1, 10),
        "max_depth": list(range(1, 11)),
        "colsample_bytree": np.linspace(0.1, 1, 10),
        "min_child_weight": list(range(1, 21)),
        "gamma": np.linspace(0, 0.5, 51),
        "scale_pos_weight": [
            1,
            y[y == 0].shape[0] / y[y == 1].shape[0],
        ],  # for imbalance
    }

    classifier = xgb.XGBClassifier(
        objective="binary:logistic", use_label_encoder=False, eval_metric="logloss"
    )

    random_search = RandomizedSearchCV(
        classifier,
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
    best_model = tune_xgboost(X_train, y_train, n_iter_search=20)

    # Calculate accuracy
    accuracy = best_model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Calculate AUC
    y_prob = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"Test AUC: {auc:.4f}")
