#include <iostream>
#include <cmath>
#include <vector>

class LinearRegression {
public:
    LinearRegression(int numFeatures) {
        weights = std::vector<double>(numFeatures, 0);
        bias = 0;
    }

    void train(const std::vector<std::vector<double>>& features, const std::vector<double>& targets) {
        int numSamples = features.size();
        int numFeatures = features[0].size();

        for (int i = 0; i < numFeatures; i++) {
            weights[i] = 0;
        }

        for (int j = 0; j < numSamples; j++) {
            double yPredicted = 0;
            for (int i = 0; i < numFeatures; i++) {
                yPredicted += features[j][i] * weights[i];
            }
            yPredicted += bias;

            double error = targets[j] - yPredicted;
            for (int i = 0; i < numFeatures; i++) {
                weights[i] += 0.01 * error * features[j][i];
            }
            bias += 0.01 * error;
        }
    }

    double predict(const std::vector<double>& feature) {
        double yPredicted = 0;
        for (int i = 0; i < feature.size(); i++) {
            yPredicted += feature[i] * weights[i];
        }
        yPredicted += bias;
        return yPredicted;
    }

private:
    std::vector<double> weights;
    double bias;
};

int main() {
    // Sample dataset
    // std::vector<std::vector<double>> features = {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}};
    // std::vector<double> targets = {3, 6, 9, 12, 15};

    std::vector<std::vector<double>> features = {{2600,20},{3000,15},{3200,18},{3600,30},{4000,8},{4100,8}};
    std::vector<double> targets = {550000,565000,610000, 595000, 760000,810000};
/*
{2600,3,20},{3000,4,15},{3200,4,18},{3600,3,30},{4000,5,8},{4100,6,8},
*/

    // Create a LinearRegression object
    LinearRegression model(features[0].size());

    // Train the model
    model.train(features, targets);

    // Make predictions
    std::vector<double> predictedTargets;
    for (int i = 0; i < features.size(); i++) {
        double predictedValue = model.predict(features[i]);
        predictedTargets.push_back(predictedValue);
    }

    // Print predictions
    for (int i = 0; i < predictedTargets.size(); i++) {
        std::cout << "Predicted value: " << predictedTargets[i] << std::endl;
    }
    std::cout << "test" << std::endl;
    return 0;
}