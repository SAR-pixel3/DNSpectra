package com.dnspectra.ml;

import com.dnspectra.data.WekaDatasetBuilder;

import weka.core.Instances;
import weka.classifiers.functions.Logistic;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class FeatureScalingExperiment {

    public static void main(String[] args) throws Exception {

        runExperiment("STATEFUL");
        runExperiment("STATELESS");
    }

    public static void runExperiment(String featureGroup)
            throws Exception {

        System.out.println("\n================================");
        System.out.println(featureGroup + " - SCALED LOGISTIC REGRESSION");
        System.out.println("================================");

        // Load the original training and testing datasets
        Instances[] datasets =
                WekaDatasetBuilder.loadExperiment(featureGroup);

        Instances training = datasets[0];
        Instances testing = datasets[1];

        // Create the Standardize filter
        Standardize standardize = new Standardize();

        // Learn scaling values from training data
        standardize.setInputFormat(training);

        // Apply scaling to training data
        Instances scaledTraining =
                Filter.useFilter(training, standardize);

        // Apply the same scaling to testing data
        Instances scaledTesting =
                Filter.useFilter(testing, standardize);
        System.out.println("\nSCALING VERIFICATION");

        System.out.println("Original first feature value: "
                + training.instance(0).value(0));

        System.out.println("Scaled first feature value: "
                + scaledTraining.instance(0).value(0));

        // Train Logistic Regression
        Logistic model = new Logistic();
        model.setMaxIts(200);
        model.buildClassifier(scaledTraining);
        // Train another model using unscaled data
        Logistic originalModel = new Logistic();
        originalModel.setMaxIts(200);
        originalModel.buildClassifier(training);

// Compare prediction probabilities
        int changedPredictions = 0;
        double totalProbabilityDifference = 0;

        for (int i = 0; i < testing.numInstances(); i++) {

            double[] originalProbabilities =
                    originalModel.distributionForInstance(testing.instance(i));

            double[] scaledProbabilities =
                    model.distributionForInstance(scaledTesting.instance(i));

            for (int j = 0; j < originalProbabilities.length; j++) {

                totalProbabilityDifference +=
                        Math.abs(originalProbabilities[j] - scaledProbabilities[j]);
            }

            if (originalModel.classifyInstance(testing.instance(i))
                    != model.classifyInstance(scaledTesting.instance(i))) {

                changedPredictions++;
            }
        }

        double averageDifference =
                totalProbabilityDifference /
                        (testing.numInstances() * 3);

        System.out.println("\nCONFIDENCE COMPARISON");
        System.out.println("Changed predictions: " + changedPredictions);

        System.out.println("Average probability difference: "
                + averageDifference);

        // Evaluate using scaled testing data
        Evaluation evaluation =
                new Evaluation(scaledTraining);

        evaluation.evaluateModel(model, scaledTesting);

        System.out.printf(
                "Accuracy: %.2f%%%n",
                evaluation.pctCorrect()
        );

        System.out.println(
                "Macro F1-score: " +
                        calculateMacroF1(evaluation.confusionMatrix())
        );

        System.out.println("\nConfusion Matrix:");
        System.out.println(evaluation.toMatrixString());
    }

    private static double calculateMacroF1(double[][] matrix) {

        double totalF1 = 0;

        for (int i = 0; i < matrix.length; i++) {

            double tp = matrix[i][i];
            double fp = 0;
            double fn = 0;

            for (int j = 0; j < matrix.length; j++) {

                if (j != i) {
                    fp += matrix[j][i];
                    fn += matrix[i][j];
                }
            }

            double precision =
                    (tp + fp == 0) ? 0 : tp / (tp + fp);

            double recall =
                    (tp + fn == 0) ? 0 : tp / (tp + fn);

            double f1 =
                    (precision + recall == 0)
                            ? 0
                            : 2 * precision * recall /
                            (precision + recall);

            totalF1 += f1;
        }

        return totalF1 / matrix.length;
    }
}
