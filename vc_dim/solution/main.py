
import numpy as np
import matplotlib.pyplot as plt

# Pandas
import pandas as pd

# Model Import
from knn import *
from logistic_regression import *
from decision_tree import *

def vc_dimension_upper_bound(train_loss, vc_dim, sample_count, failure_rate):
    upper_bound = train_loss \
        + np.sqrt((2*vc_dim*np.log((np.exp(1) * sample_count) / vc_dim)) / sample_count) \
        + np.sqrt(np.log(1/failure_rate) /(2 *sample_count))

    return upper_bound

if __name__ == "__main__":
    # failtuer_rate
    failure_rate = .05
    
    # Index
    dimension_list = []
    train_count_list = []

    # Train loss list
    logistic_regression_loss_train = []
    knn_loss_train = []
    decision_tree_loss_train = []

    # loss Upper bound list
    logistic_regression_upper = []
    knn_loss_upper = []
    decision_tree_loss_upper = []

    # Test loss list
    logistic_regression_loss_test = []
    knn_loss_test = []
    decision_tree_loss_test = []

    for i in range(1,10,1):
        for j in range(10, 101, 20):
            # Experimental settings
            dimensions = i
            train_count = j
            test_count = j

            # Model Train
            train_result_logistic, test_result_logistic \
                = logistic_regression_train(dimensions, train_count, test_count, 10)
            train_result_knn, test_result_knn \
                = knn_train(dimensions, train_count, test_count, 10)
            train_result_tree, test_result_tree, average_terminal_count \
                = decision_tree_train(dimensions, train_count, test_count, 10)

            # VC dimension
            vc_dimension_linear = i + 1
            vc_dimension_knn = j
            vc_dimension_decision_tree = average_terminal_count

            # Upper bound
            upper_bound_logistic \
                = vc_dimension_upper_bound(train_result_logistic[0], vc_dimension_linear, train_count, failure_rate)
            upper_bound_knn \
                = vc_dimension_upper_bound(train_result_knn[0], vc_dimension_knn, train_count, failure_rate)
            upper_bound_tree \
                = vc_dimension_upper_bound(train_result_knn[1], vc_dimension_decision_tree, train_count, failure_rate)

            # Append
            dimension_list.append(dimensions)
            train_count_list.append(train_count)
            
            # logistic
            logistic_regression_loss_train.append(train_result_logistic[0])
            logistic_regression_upper.append(upper_bound_logistic)
            logistic_regression_loss_test.append(test_result_logistic[0])

            # knn
            knn_loss_train.append(train_result_knn[0])
            knn_loss_upper.append(upper_bound_knn)
            knn_loss_test.append(test_result_knn[0])

            # tree
            decision_tree_loss_train.append(train_result_tree[0])
            decision_tree_loss_upper.append(upper_bound_tree)
            decision_tree_loss_test.append(test_result_tree[0])

            

    # Make a dataframe
    result_data = [dimension_list, train_count_list, \
        logistic_regression_loss_train, logistic_regression_upper, logistic_regression_loss_test, \
        knn_loss_train, knn_loss_upper, knn_loss_test,\
        decision_tree_loss_train, decision_tree_loss_upper, decision_tree_loss_test]

    result_data = np.array(result_data)
    result_data = result_data.T

    
    result_column_name = ["dimensions", "train_count",\
        "logistic_train_loss", "logistic_upper_bound", "logistic_test_loss", \
        "knn_train_loss", "knn_upper_bound", "knn_test_loss",\
        "tree_train_loss", "tree_upper_bound", "tree_test_loss"]

    result_DF = pd.DataFrame(result_data, columns=result_column_name)

    print(result_DF)




    
        

    
    

    





