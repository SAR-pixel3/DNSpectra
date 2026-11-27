package com.dnspectra.ml;

import com.dnspectra.data.WekaDatasetBuilder;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.File;

public class DecisionTreeTrainer {

    public static void main(String[] args) throws Exception {

        System.out.println("DNSPECTRA - DECISION TREE TRAINING");
        System.out.println("==================================");

        // Train and evaluate using stateful features
        runExperiment("STATEFUL");

        // Train and evaluate using stateless features
        runExperiment("STATELESS");
    }

    private static void runExperiment(String experiment)
            throws Exception {

        System.out.println("\n========== " + experiment + " ==========");

        // Load training and testing datasets
        Instances[] datasets =
                WekaDatasetBuilder.loadExperiment(experiment);

        Instances training = datasets[0];
        Instances testing = datasets[1];

        // Create the Decision Tree classifier
        J48 model = new J48();

        // Train the model using training data only
        System.out.println("\nTraining Decision Tree...");
        model.buildClassifier(training);

        System.out.println("Training completed.");

        // Evaluate using the separate testing dataset
        Evaluation evaluation = new Evaluation(training);
        evaluation.evaluateModel(model, testing);

        // Display evaluation results
        System.out.println("\nMODEL PERFORMANCE");
        System.out.println("-----------------");

        System.out.printf("Accuracy: %.2f%%%n",
                evaluation.pctCorrect());

        System.out.printf("Macro F1-score: %.4f%n",
                calculateMacroF1(evaluation, training));

        System.out.println("\nCLASSIFICATION DETAILS");
        System.out.println(evaluation.toClassDetailsString());

        System.out.println("\nCONFUSION MATRIX");
        System.out.println(evaluation.toMatrixString());

        // Save the trained model
        File modelDirectory = new File("models");
        modelDirectory.mkdirs();

        String modelPath =
                "models/decision-tree-" +
                        experiment.toLowerCase() + ".model";

        SerializationHelper.write(modelPath, model);

        System.out.println("\nModel saved to: " + modelPath);
    }

    private static double calculateMacroF1(
            Evaluation evaluation, Instances training) throws Exception {

        double[][] matrix = evaluation.confusionMatrix();

        double totalF1 = 0.0;
        int numberOfClasses = training.numClasses();

        for (int i = 0; i < numberOfClasses; i++) {

            double truePositive = matrix[i][i];

            double falsePositive = 0.0;
            double falseNegative = 0.0;

            for (int j = 0; j < numberOfClasses; j++) {
                if (j != i) {
                    falsePositive += matrix[j][i];
                    falseNegative += matrix[i][j];
                }
            }

            double precision =
                    (truePositive + falsePositive) == 0
                            ? 0
                            : truePositive / (truePositive + falsePositive);

            double recall =
                    (truePositive + falseNegative) == 0
                            ? 0
                            : truePositive / (truePositive + falseNegative);

            double f1 =
                    (precision + recall) == 0
                            ? 0
                            : 2 * precision * recall / (precision + recall);

            totalF1 += f1;
        }

        return totalF1 / numberOfClasses;
    }
}
