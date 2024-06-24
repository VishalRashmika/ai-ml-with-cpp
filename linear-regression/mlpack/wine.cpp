#include <iostream>

#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

using namespace arma;
using namespace mlpack;

int main(){
    // loading both training and testing datsets
    mat train_dataset;
    mat test_dataset;

    bool train_dataset_load_status = mlpack::data::Load("WineQT.csv", train_dataset);
    bool test_dataset_load_status = mlpack::data::Load("wine_test_set.csv", test_dataset);

    //dropping the ID column
    train_dataset.shed_row(12);
    test_dataset.shed_row(12);

    // seperating features and targets
    mat train_features = train_dataset.cols(1,train_dataset.n_cols - 2); // load all the rows (excluding header row and the last row) range = (1:last - 1) [last - 1 : because the last entry is used as a prediction to the model]
    rowvec train_targets = train_features.row(11);
    train_features.shed_row(11);
    
    mat test_features = test_dataset.cols(1,test_dataset.n_rows - 1);
    rowvec test_targets = test_features.row(11);
    test_features.shed_row(11);

    // extract a single entry to predict from train_dataset (in this case the last element in the train_dataset)
    mat predict_features = train_dataset.col(train_dataset.n_cols - 1); // std.n_cols/rows - 1 := last index of the sth
    rowvec predict_target = predict_features.row(11);
    predict_features.shed_row(11);


    // Model Building
    LinearRegression regressor;
    regressor.Train(train_features,train_targets);

    // Model Evaluation
    std::cout << "------------------------\n---------Train Set------" << std::endl;

    double train_mse;
    train_mse = regressor.ComputeError(train_features,train_targets);
    std::cout << "Train MSE: " << train_mse << std::endl;

    double train_r1_score;
    train_r1_score = R2Score<false>::Evaluate(regressor,train_features, train_targets);
    std::cout << "Train R2 Score: " << train_r1_score << std::endl;

    std::cout << "\n------------------------\n---------Test Set-------" << std::endl;

    double test_mse;
    test_mse = regressor.ComputeError(test_features,test_targets);
    std::cout << "Train MSE: " << test_mse << std::endl;

    double test_r1_score;
    test_r1_score = R2Score<false>::Evaluate(regressor,test_features, test_targets);
    std::cout << "Train R2 Score: " << test_r1_score << std::endl;

    // prediction
    rowvec prediction_result;
    regressor.Predict(predict_features,prediction_result);
    std::cout << "\nOrginal Target : " << predict_target << "\nPredicted Target: " << prediction_result << std::endl;

    return 0;
}