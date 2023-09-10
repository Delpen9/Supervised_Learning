# Standard Data Science Libraries
import numpy as np
import pandas as pd

# Helper Functions
from helper_functions.model_training_evaluation_helper_functions import (
    get_auction_verification_model_metrics,
    get_student_dropout_model_metrics,
    get_auction_percentage_curves,
    get_dropout_percentage_curves,
    get_neural_network_percentage_curves,
    get_all_best_models,
)
from helper_functions.feature_importance_decision_boundaries_helper_functions import (
    load_models,
)

if __name__ == "__main__":
    np.random.seed(1234)

    # run_cases = np.arange(1, 10).astype(int) # Run all cases
    run_cases = [6]

    if 1 in run_cases:
        get_auction_verification_model_metrics()
        get_student_dropout_model_metrics()

    if 2 in run_cases:
        get_auction_percentage_curves()
        get_dropout_percentage_curves()

    if 3 in run_cases:
        get_neural_network_percentage_curves()

    if 4 in run_cases:
        get_all_best_models(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        get_all_best_models(
            filename="../data/student_dropout_dataset/data.csv", dataset_type="dropout"
        )

    if 5 in run_cases:
        best_model_dt, best_model_xgb, best_model_svm, best_model_knn = load_models(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )

    if 6 in run_cases:
        get_model_SHAP(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
            model="dt",
            model_object=best_model_dt,
        )

    ## TODO: Get hyperparameter information and feature importances (i.e. SHAP)
    ## Consider getting decision boundary illustrations for each model; this may involve PCA
    ## %
    ##
    ##
    ## %
