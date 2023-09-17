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
    get_best_neural_network,
)
from helper_functions.feature_importance_decision_boundaries_helper_functions import (
    load_models,
    get_all_models_SHAP,
    get_neural_network_SHAP,
    get_all_models_decision_boundaries,
    get_neural_network_decision_boundary,
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
        get_best_neural_network(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        get_best_neural_network(
            filename="../data/student_dropout_dataset/data.csv", dataset_type="dropout"
        )

    if 5 in run_cases:
        get_all_models_SHAP(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        get_all_models_SHAP(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )
        get_neural_network_SHAP(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        get_neural_network_SHAP(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )

    if 6 in run_cases:
        get_all_models_decision_boundaries(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        get_all_models_decision_boundaries(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )
        get_neural_network_decision_boundary(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        get_neural_network_decision_boundary(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )
    
    ## TODO: Get individual model performance values by varying individual features
    ## i.e. ccp_alpha for decision tree
    ## i.e. learning_rate for neural_network
    ## TODO: Get univariate and multivariate descriptive statistics of the datasets against each other and their targets.
    ## TODO: Current KNN plots show results without pre-processing; do pre-processing and compare new performance.
    ## TODO: Get graph that shows average performance over specific iteration counts for grid search. (10 vs. 100 vs. 500)
    ## %
    ## 
    ## 
    ## %
