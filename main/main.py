# Standard Data Science Libraries
import numpy as np
import pandas as pd

# Helper Functions
from model_training_evaluation_helper_functions import (
    get_auction_verification_model_metrics,
    get_student_dropout_model_metrics,
    get_auction_percentage_curves,
    get_dropout_percentage_curves,
    get_neural_network_percentage_curves,
    get_all_best_models,
)

if __name__ == "__main__":
    np.random.seed(1234)

    # get_auction_verification_model_metrics()
    # get_student_dropout_model_metrics()

    # get_auction_percentage_curves()
    # get_dropout_percentage_curves()

    # get_neural_network_percentage_curves()

    # get_all_best_models(
    #     filename="../data/auction_verification_dataset/data.csv", dataset_type="auction"
    # )
    get_all_best_models(
        filename="../data/student_dropout_dataset/data.csv", dataset_type="dropout"
    )

    ## TODO: Get hyperparameter information and feature importances (i.e. SHAP)
    ## Consider getting decision boundary illustrations for each model; this may involve PCA
    ## %
    ##
    ##
    ## %
