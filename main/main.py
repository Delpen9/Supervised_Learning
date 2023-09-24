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
from helper_functions.model_hyperparameter_testing_helper_functions import (
    compare_svm_kernels,
    ccp_decision_tree_performance_graph,
    max_depth_decision_tree_performance_graph,
    min_samples_per_leaf_decision_tree_performance_graph,
    min_impurity_decrease_decision_tree_performance_graph,
    get_performance_by_value_of_k_knn,
    get_neural_network_performance_by_learning_rate,
    get_performance_by_value_of_c_svm,
    get_performance_by_value_of_learning_rate_xgboost,
    get_performance_by_value_of_max_depth_xgboost,
    get_performance_by_value_of_n_estimators_xgboost,
    get_neural_network_performance_by_hidden_dimensions,
    get_neural_network_performance_heatmap,
)
from helper_functions.univariate_multivariate_eda_helper_functions import (
    get_correlation_heatmap,
    get_scatter_plots,
)
from helper_functions.kNN_helper_functions import (
    get_pre_processed_performance_by_value_of_k_knn,
    tune_knn_with_pre_processing,
)

if __name__ == "__main__":
    np.random.seed(1234)

    # run_cases = np.arange(1, 13).astype(int) # Run all cases
    run_cases = [14]

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
    
    if 7 in run_cases:
        compare_svm_kernels(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        compare_svm_kernels(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )
    
    if 8 in run_cases:
        ccp_decision_tree_performance_graph(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        ccp_decision_tree_performance_graph(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )

        max_depth_decision_tree_performance_graph(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        max_depth_decision_tree_performance_graph(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )

        min_samples_per_leaf_decision_tree_performance_graph(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        min_samples_per_leaf_decision_tree_performance_graph(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )

        min_impurity_decrease_decision_tree_performance_graph(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        min_impurity_decrease_decision_tree_performance_graph(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )

    if 9 in run_cases:
        get_performance_by_value_of_k_knn(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        get_performance_by_value_of_k_knn(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )

    if 10 in run_cases:
        get_neural_network_performance_by_learning_rate(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        get_neural_network_performance_by_learning_rate(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )

    if 11 in run_cases:
        get_performance_by_value_of_c_svm(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        get_performance_by_value_of_c_svm(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )
    
    if 12 in run_cases:
        get_performance_by_value_of_learning_rate_xgboost(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        get_performance_by_value_of_learning_rate_xgboost(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )
        get_performance_by_value_of_max_depth_xgboost(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        get_performance_by_value_of_max_depth_xgboost(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )
        get_performance_by_value_of_n_estimators_xgboost(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        get_performance_by_value_of_n_estimators_xgboost(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )
        get_neural_network_performance_by_hidden_dimensions(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        get_neural_network_performance_by_hidden_dimensions(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )
        get_neural_network_performance_heatmap(
            dataset_type="auction",
            set_version="test",
            metric="AUC",
        )
        get_neural_network_performance_heatmap(
            dataset_type="auction",
            set_version="test",
            metric="Accuracy",
        )
        get_neural_network_performance_heatmap(
            dataset_type="dropout",
            set_version="test",
            metric="AUC",
        )
        get_neural_network_performance_heatmap(
            dataset_type="dropout",
            set_version="test",
            metric="Accuracy",
        )
    
    if 13 in run_cases:
        get_correlation_heatmap(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        get_correlation_heatmap(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )
        get_scatter_plots(
            filename="../data/auction_verification_dataset/data.csv",
            dataset_type="auction",
        )
        get_scatter_plots(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )

    if 14 in run_cases:
        # get_pre_processed_performance_by_value_of_k_knn(
        #     filename="../data/student_dropout_dataset/data.csv",
        #     dataset_type="dropout",
        # )
        tune_knn_with_pre_processing(
            filename="../data/student_dropout_dataset/data.csv",
            dataset_type="dropout",
        )