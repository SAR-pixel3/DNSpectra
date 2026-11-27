package com.dnspectra.ml;

import com.dnspectra.data.WekaDatasetBuilder;

import weka.core.Instances;
import weka.classifiers.functions.Logistic;
import weka.classifiers.Evaluation;

public class ModerateClassWeightingExperiment {

    public static void main(String[] args) throws Exception {

        runExperiment("STATEFUL");
        runExperiment("STATELESS");
    }

    public static void runExperiment(String featureGroup)
            throws Exception {

        System.out.println("\n================================");
        System.out.println(featureGroup + " - MODERATE CLASS WEIGHTING");
        System.out.println("================================");

        // Load the training and testing datasets
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

        // Calculate moderate class weights
        double[] weights = new double[training.numClasses()];

        System.out.println("\nMODERATE CLASS WEIGHTS");

        for (int i = 0; i < weights.length; i++) {

            double originalWeight =
                    training.numInstances() /
                            (double) (training.numClasses() * classCounts[i]);

            weights[i] = Math.sqrt(originalWeight);

            System.out.printf(
                    "%s | Records: %.0f | Weight: %.4f%n",
                    training.classAttribute().value(i),
                    classCounts[i],
                    weights[i]
            );
        }

        // Apply weights to training instances
        Instances weightedTraining = new Instances(training);

        for (int i = 0; i < weightedTraining.numInstances(); i++) {

            int classIndex =
                    (int) weightedTraining.instance(i).classValue();

            weightedTraining.instance(i).setWeight(weights[classIndex]);
        }

        // Train Logistic Regression
        Logistic model = new Logistic();
        model.setMaxIts(200);
        model.buildClassifier(weightedTraining);

        // Evaluate on the original test dataset
        Evaluation evaluation = new Evaluation(weightedTraining);
        evaluation.evaluateModel(model, testing);

        System.out.printf(
                "\nAccuracy: %.2f%%%n",
                evaluation.pctCorrect()
        );

        System.out.printf(
                "Macro F1-score: %.4f%n",
                calculateMacroF1(evaluation.confusionMatrix())
        );

        System.out.println("\nPer-class results:");
        System.out.println(evaluation.toClassDetailsString());

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
