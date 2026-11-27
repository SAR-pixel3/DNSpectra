
package com.dnspectra.evaluation;

import com.dnspectra.data.WekaDatasetBuilder;

import weka.core.Instances;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;

public class LightAttackFeatureAnalysis {

    public static void main(String[] args) throws Exception {

        runExperiment("STATEFUL");
    }

    public static void runExperiment(String featureGroup)
            throws Exception {

        System.out.println("\n================================");
        System.out.println(
                featureGroup + " - LIGHT ATTACK FEATURE ANALYSIS"
        );
        System.out.println("================================");

        // Load the existing capture-level train/test split
        Instances[] datasets =
                WekaDatasetBuilder.loadExperiment(featureGroup);

        Instances training = datasets[0];
        Instances testing = datasets[1];

        // Create Random Forest
        RandomForest model = new RandomForest();

        model.setNumIterations(100);
        model.setSeed(42);

        // Enable global feature importance
        model.setComputeAttributeImportance(true);

        System.out.println("\nTraining Random Forest...");

        model.buildClassifier(training);

        System.out.println("Training completed.");

        // Evaluate on held-out test data
        Evaluation evaluation = new Evaluation(training);

        evaluation.evaluateModel(model, testing);

        // Display class-specific evaluation
        System.out.println("\nCLASS PERFORMANCE");
        System.out.println("--------------------------------");

        for (int i = 0; i < testing.numClasses(); i++) {

            String className =
                    testing.classAttribute().value(i);

            double precision =
                    evaluation.precision(i);

            double recall =
                    evaluation.recall(i);

            double f1 =
                    evaluation.fMeasure(i);

            System.out.printf(
                    "%s | Precision: %.4f | Recall: %.4f | F1: %.4f%n",
                    className,
                    precision,
                    recall,
                    f1
            );
        }

        // Find LIGHT_ATTACK class index
        int lightIndex =
                testing.classAttribute().indexOfValue("LIGHT_ATTACK");

        if (lightIndex < 0) {
            throw new IllegalStateException(
                    "LIGHT_ATTACK class not found."
            );
        }

        // Display LIGHT_ATTACK confusion matrix details
        double[][] matrix = evaluation.confusionMatrix();

        System.out.println("\nLIGHT_ATTACK PREDICTION ANALYSIS");
        System.out.println("--------------------------------");

        double trueLight = matrix[lightIndex][lightIndex];

        double missedLight = 0;
        double falseLight = 0;

        for (int i = 0; i < matrix.length; i++) {

            if (i != lightIndex) {
                missedLight += matrix[lightIndex][i];
                falseLight += matrix[i][lightIndex];
            }
        }

        System.out.println(
                "Correct LIGHT_ATTACK predictions: " + (int) trueLight
        );

        System.out.println(
                "Missed LIGHT_ATTACK records: " + (int) missedLight
        );

        System.out.println(
                "False LIGHT_ATTACK predictions: " + (int) falseLight
        );

        // Display overall feature importance
        System.out.println("\nGLOBAL FEATURE IMPORTANCE");
        System.out.println("--------------------------------");

        System.out.println(model.toString());

        System.out.println("\n================================");
        System.out.println("END OF LIGHT ATTACK ANALYSIS");
        System.out.println("================================");
    }
}

