package com.dnspectra.evaluation;

import com.dnspectra.data.WekaDatasetBuilder;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;

public class ModelComparison {

    public static void main(String[] args) throws Exception {

        System.out.println("DNSPECTRA - MODEL COMPARISON");
        System.out.println("============================");

        String[] experiments = {"STATEFUL", "STATELESS"};

        for (String experiment : experiments) {

            System.out.println("\n================================");
            System.out.println(experiment + " FEATURES");
            System.out.println("================================");

            Instances[] datasets =
                    WekaDatasetBuilder.loadExperiment(experiment);

            Instances training = datasets[0];
            Instances testing = datasets[1];

            evaluateModel(
                    "Decision Tree",
                    new J48(),
                    training,
                    testing
            );

            RandomForest forest = new RandomForest();
            forest.setNumIterations(100);
            forest.setSeed(42);

            evaluateModel(
                    "Random Forest",
                    forest,
                    training,
                    testing
            );

            Logistic logistic = new Logistic();
            logistic.setMaxIts(200);

            evaluateModel(
                    "Logistic Regression",
                    logistic,
                    training,
                    testing
            );
        }
    }

    private static void evaluateModel(
            String name,
            Classifier model,
            Instances training,
            Instances testing) throws Exception {

        System.out.println("\nMODEL: " + name);
        System.out.println("----------------------------");

        // Train using training data only
        model.buildClassifier(training);

        // Evaluate using held-out test data
        Evaluation evaluation = new Evaluation(training);
        evaluation.evaluateModel(model, testing);

        System.out.printf(
                "Accuracy: %.2f%%%n",
                evaluation.pctCorrect()
        );

        System.out.printf(
                "Macro F1-score: %.4f%n",
                calculateMacroF1(evaluation, training)
        );

        System.out.println("\nPer-class results:");
        System.out.println(evaluation.toClassDetailsString());

        System.out.println("Confusion matrix:");
        System.out.println(evaluation.toMatrixString());
    }

    private static double calculateMacroF1(
            Evaluation evaluation,
            Instances training) {

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
                            : truePositive /
                            (truePositive + falsePositive);

            double recall =
                    (truePositive + falseNegative) == 0
                            ? 0
                            : truePositive /
                            (truePositive + falseNegative);

            double f1 =
                    (precision + recall) == 0
                            ? 0
                            : 2 * precision * recall /
                            (precision + recall);

            totalF1 += f1;
        }

        return totalF1 / numberOfClasses;
    }
}
