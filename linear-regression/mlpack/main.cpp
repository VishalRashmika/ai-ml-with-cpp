#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

using namespace arma;
using namespace mlpack;
using namespace std;

// dataset shape :: 7239,244

int main(){
    // load data
    mat dataset;
    bool loaded = mlpack::data::Load("cleaned-house-prices.csv",dataset);

    mat train_data = dataset.cols(1,dataset.n_cols - 2);
    rowvec train_target = train_data.row(train_data.n_rows - 242);

    train_data.shed_row(train_data.n_rows - 242);
    //train_data.shed_row(0);

    mat test_dataset;
    bool loaded_status = mlpack::data::Load("exported.csv",test_dataset);
    mat test_features = test_dataset.cols(1,test_dataset.n_cols - 2);
    rowvec test_targets = test_features.row(test_features.n_rows - 242);
    test_features.shed_row(test_features.n_rows - 242);


/*
    mat test_data = dataset.col(dataset.n_cols - 1);
    rowvec test_target = test_data.row(test_data.n_rows - 242);
    test_data.shed_row(test_data.n_rows - 242);
    //test_data.shed_row(0);
    mat test_data2 = dataset.col(5000);
    rowvec test_target2 = test_data2.row(2);
    test_data2.shed_row(2);
*/
    LinearRegression regressor;

    regressor.Train(train_data,train_target);

    // test
    double MSE;
    MSE = regressor.ComputeError(train_data, train_target);

    //predict
    // rowvec prediction;
    // regressor.Predict(test_data,prediction);


    // cout << "test target 1: " << test_target << endl;
    // cout << "prediction  1: " << prediction << endl;
    // cout << "MSE          : " << MSE << endl;

    // // prediction2
    // regressor.Predict(test_data2,prediction);
    // cout << "test target 2: " << test_target2 << endl;
    // cout << "prediction  2: " << prediction << endl;
    // cout << "MSE          : " << MSE << endl;

    // r2 score
    double container;
    container =  R2Score<false>::Evaluate(regressor,test_features,test_targets);
    // F1::Evaluate(regressor,test_features,test_targets);
    KFoldCV<LinearRegression, F1> cv(regressor,test_features,test_targets,241);
    
    cout << "R2: " << container << endl;

    return 0;

}