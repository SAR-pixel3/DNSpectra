
package com.dnspectra.evaluation;

import com.dnspectra.data.WekaDatasetBuilder;

import weka.core.Instances;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;

public class FeatureImportanceExperiment {

    public static void main(String[] args) throws Exception {

        runExperiment("STATELESS");
    }

    public static void runExperiment(String featureGroup)
            throws Exception {

        System.out.println("\n================================");
        System.out.println(
                featureGroup + " - FEATURE IMPORTANCE EXPERIMENT"
        );
        System.out.println("================================");

        // Load training and testing datasets
        Instances[] datasets =
                WekaDatasetBuilder.loadExperiment(featureGroup);

        Instances training = datasets[0];
        Instances testing = datasets[1];

        // Create Random Forest model
        RandomForest model = new RandomForest();

        model.setNumIterations(100);
        model.setSeed(42);

        // Enable built-in feature importance calculation
        model.setComputeAttributeImportance(true);

        // Train using training data only
        System.out.println("\nTraining Random Forest...");

        model.buildClassifier(training);

        System.out.println("Training completed.");

        // Evaluate on the held-out test dataset
        Evaluation evaluation = new Evaluation(training);

        evaluation.evaluateModel(model, testing);

        System.out.printf(
                "\nTest Accuracy: %.2f%%%n",
                evaluation.pctCorrect()
        );

        // Display feature importance information
        System.out.println("\n================================");
        System.out.println("FEATURE IMPORTANCE RANKING");
        System.out.println("================================");

        System.out.println(model.toString());

        System.out.println("\n================================");
        System.out.println("END OF FEATURE IMPORTANCE EXPERIMENT");
        System.out.println("================================");
    }
}
