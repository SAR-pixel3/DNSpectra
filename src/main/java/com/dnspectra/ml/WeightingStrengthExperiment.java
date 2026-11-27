
package com.dnspectra.ml;

import com.dnspectra.data.WekaDatasetBuilder;

import weka.core.Instances;
import weka.classifiers.functions.Logistic;
import weka.classifiers.Evaluation;

public class WeightingStrengthExperiment {

    // Five weighting strengths to test
    private static final double[] ALPHAS = {
            0.0, 0.25, 0.50, 0.75, 1.0
    };

    public static void main(String[] args) throws Exception {

        runExperiment("STATEFUL");
        runExperiment("STATELESS");
    }

    public static void runExperiment(String featureGroup)
            throws Exception {

        System.out.println("\n================================");
        System.out.println(
                featureGroup + " - WEIGHTING STRENGTH EXPERIMENT"
        );
        System.out.println("================================");

        // Load datasets
        Instances[] datasets =
                WekaDatasetBuilder.loadExperiment(featureGroup);

        Instances training = datasets[0];
        Instances testing = datasets[1];

        // Count training records in each class
        double[] classCounts = new double[training.numClasses()];

        for (int i = 0; i < training.numInstances(); i++) {

            int classIndex =
                    (int) training.instance(i).classValue();

            classCounts[classIndex]++;
        }

        System.out.println("\nCLASS DISTRIBUTION");

        for (int i = 0; i < classCounts.length; i++) {

            System.out.printf(
                    "%s: %.0f%n",
                    training.classAttribute().value(i),
                    classCounts[i]
            );
        }

        // Test all weighting strengths
        System.out.println("\nEXPERIMENT RESULTS");

        System.out.println(
                "Alpha | Accuracy | Macro F1 | LIGHT Precision | LIGHT Recall | LIGHT F1"
        );

        for (double alpha : ALPHAS) {

            // Create weighted training data
            Instances weightedTraining =
                    applyClassWeights(training, classCounts, alpha);

            // Train Logistic Regression
            Logistic model = new Logistic();
            model.setMaxIts(200);
            model.buildClassifier(weightedTraining);

            // Evaluate using unchanged test data
            Evaluation evaluation =
                    new Evaluation(weightedTraining);

            evaluation.evaluateModel(model, testing);

            double[][] matrix = evaluation.confusionMatrix();

            double macroF1 = calculateMacroF1(matrix);
            double lightPrecision = calculatePrecision(matrix, 1);
            double lightRecall = calculateRecall(matrix, 1);
            double lightF1 = calculateF1(matrix, 1);

            System.out.printf(
                    "%.2f  | %.2f%% | %.4f | %.4f | %.4f | %.4f%n",
                    alpha,
                    evaluation.pctCorrect(),
                    macroF1,
                    lightPrecision,
                    lightRecall,
                    lightF1
            );
        }

        // Print confusion matrices for alpha 0.75 and 1.00
        printConfusionMatrix(
                training, testing, classCounts, 0.75
        );

        printConfusionMatrix(
                training, testing, classCounts, 1.00
        );
    }

    // Apply class weights and return a new training dataset
    private static Instances applyClassWeights(
            Instances training,
            double[] classCounts,
            double alpha) {

        Instances weightedTraining = new Instances(training);

        for (int i = 0; i < weightedTraining.numInstances(); i++) {

            int classIndex =
                    (int) weightedTraining.instance(i).classValue();

            double baseWeight =
                    training.numInstances() /
                            (double) (training.numClasses()
                                    * classCounts[classIndex]);

            double weight = Math.pow(baseWeight, alpha);

            weightedTraining.instance(i).setWeight(weight);
        }

        return weightedTraining;
    }

    // Print confusion matrix for a selected alpha
    private static void printConfusionMatrix(
            Instances training,
            Instances testing,
            double[] classCounts,
            double alpha) throws Exception {

        // Create weighted training data
        Instances weightedTraining =
                applyClassWeights(training, classCounts, alpha);

        // Train model
        Logistic model = new Logistic();
        model.setMaxIts(200);
        model.buildClassifier(weightedTraining);

        // Evaluate model
        Evaluation evaluation =
                new Evaluation(weightedTraining);

        evaluation.evaluateModel(model, testing);

        // Display confusion matrix
        System.out.println(
                "\nCONFUSION MATRIX (Alpha = " + alpha + ")"
        );

        System.out.println(evaluation.toMatrixString());
    }

    // Calculate Macro F1
    private static double calculateMacroF1(double[][] matrix) {

        double totalF1 = 0;

        for (int i = 0; i < matrix.length; i++) {
            totalF1 += calculateF1(matrix, i);
        }

        return totalF1 / matrix.length;
    }

    // Calculate Precision
    private static double calculatePrecision(
            double[][] matrix, int classIndex) {

        double tp = matrix[classIndex][classIndex];
        double fp = 0;

        for (int i = 0; i < matrix.length; i++) {

            if (i != classIndex) {
                fp += matrix[i][classIndex];
            }
        }

        return (tp + fp == 0) ? 0 : tp / (tp + fp);
    }

    // Calculate Recall
    private static double calculateRecall(
            double[][] matrix, int classIndex) {

        double tp = matrix[classIndex][classIndex];
        double fn = 0;

        for (int j = 0; j < matrix.length; j++) {

            if (j != classIndex) {
                fn += matrix[classIndex][j];
            }
        }

        return (tp + fn == 0) ? 0 : tp / (tp + fn);
    }

    // Calculate F1 Score
    private static double calculateF1(
            double[][] matrix, int classIndex) {

        double precision =
                calculatePrecision(matrix, classIndex);

        double recall =
                calculateRecall(matrix, classIndex);

        return (precision + recall == 0)
                ? 0
                : 2 * precision * recall /
                (precision + recall);
    }
}
