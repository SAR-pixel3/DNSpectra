
package com.dnspectra.ml;

import com.dnspectra.data.WekaDatasetBuilder;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.Logistic;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class BinaryAttackClassificationExperiment {

    private static final int LIGHT = 0;
    private static final int HEAVY = 1;

    private static class PredictionScore {
        double score;
        boolean isLight;

        PredictionScore(double score, boolean isLight) {
            this.score = score;
            this.isLight = isLight;
        }
    }

    public static void main(String[] args) throws Exception {

        runExperiment("STATEFUL");
        runExperiment("STATELESS");
    }

    private static void runExperiment(String featureGroup)
            throws Exception {

        System.out.println("\n================================");
        System.out.println(
                featureGroup + " - BINARY CLASSIFICATION"
        );
        System.out.println("================================");

        // Load the original capture-level split
        Instances[] datasets =
                WekaDatasetBuilder.loadExperiment(featureGroup);

        // Convert three classes into two classes
        Instances training =
                filterAttackClasses(datasets[0]);

        Instances testing =
                filterAttackClasses(datasets[1]);

        System.out.println("\nBINARY DATASET");
        System.out.println("--------------------------------");

        System.out.println(
                "Training records: " + training.numInstances()
        );

        System.out.println(
                "Testing records: " + testing.numInstances()
        );

        System.out.println(
                "Features: " + (training.numAttributes() - 1)
        );

        System.out.println(
                "Classes: " + training.numClasses()
        );

        printDistribution(training, "Training");
        printDistribution(testing, "Testing");

        // Majority-class baseline
        printMajorityBaseline(training, testing);

        // Train and evaluate all three models
        evaluateModel(
                "Decision Tree",
                new J48(),
                training,
                testing
        );

        RandomForest randomForest = new RandomForest();
        randomForest.setNumIterations(100);
        randomForest.setSeed(42);

        evaluateModel(
                "Random Forest",
                randomForest,
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

    // Convert the original three-class dataset into two classes
    private static Instances filterAttackClasses(
            Instances original) {

        FastVector attributes = new FastVector();

        // Copy all feature attributes
        for (int i = 0; i < original.numAttributes(); i++) {

            if (i == original.classIndex()) {
                continue;
            }

            attributes.addElement(
                    (Attribute) original.attribute(i).copy()
            );
        }

        // Create the two-class label
        FastVector classValues = new FastVector();

        classValues.addElement("LIGHT_ATTACK");
        classValues.addElement("HEAVY_ATTACK");

        Attribute binaryClass =
                new Attribute("label", classValues);

        attributes.addElement(binaryClass);

        // Create the new dataset
        Instances filtered = new Instances(
                "BinaryAttackDataset",
                attributes,
                original.numInstances()
        );

        filtered.setClassIndex(filtered.numAttributes() - 1);

        int lightIndex =
                original.classAttribute()
                        .indexOfValue("LIGHT_ATTACK");

        int heavyIndex =
                original.classAttribute()
                        .indexOfValue("HEAVY_ATTACK");

        if (lightIndex < 0 || heavyIndex < 0) {
            throw new IllegalStateException(
                    "Required attack classes not found."
            );
        }

        // Copy only LIGHT and HEAVY records
        for (int i = 0; i < original.numInstances(); i++) {

            Instance source = original.instance(i);

            int oldLabel = (int) source.classValue();

            if (oldLabel != lightIndex && oldLabel != heavyIndex) {
                continue;
            }

            double[] values =
                    new double[filtered.numAttributes()];

            int newIndex = 0;

            // Copy feature values, excluding old class attribute
            for (int j = 0; j < original.numAttributes(); j++) {

                if (j == original.classIndex()) {
                    continue;
                }

                values[newIndex++] = source.value(j);
            }

            // LIGHT = 0, HEAVY = 1
            values[newIndex] =
                    oldLabel == lightIndex ? LIGHT : HEAVY;

            DenseInstance binaryInstance =
                    new DenseInstance(source.weight(), values);

            filtered.add(binaryInstance);
        }

        return filtered;
    }

    // Print the number of records in each class
    private static void printDistribution(
            Instances data,
            String datasetName) {

        System.out.println("\n" + datasetName + " distribution:");

        for (int classIndex = 0;
             classIndex < data.numClasses();
             classIndex++) {

            int count = 0;

            for (int i = 0; i < data.numInstances(); i++) {

                if ((int) data.instance(i).classValue()
                        == classIndex) {
                    count++;
                }
            }

            System.out.println(
                    data.classAttribute().value(classIndex)
                            + ": " + count
            );
        }
    }

    // Baseline: always predict the training majority class
    private static void printMajorityBaseline(
            Instances training,
            Instances testing) {

        int lightCount = 0;
        int heavyCount = 0;

        for (int i = 0; i < training.numInstances(); i++) {

            if ((int) training.instance(i).classValue() == LIGHT) {
                lightCount++;
            } else {
                heavyCount++;
            }
        }

        int majorityClass =
                lightCount >= heavyCount ? LIGHT : HEAVY;

        int correct = 0;

        for (int i = 0; i < testing.numInstances(); i++) {

            if ((int) testing.instance(i).classValue()
                    == majorityClass) {
                correct++;
            }
        }

        double accuracy =
                100.0 * correct / testing.numInstances();

        System.out.println("\nMAJORITY-CLASS BASELINE");
        System.out.println("--------------------------------");

        System.out.println(
                "Training majority class: "
                        + training.classAttribute()
                        .value(majorityClass)
        );

        System.out.printf(
                "Test accuracy: %.2f%%%n",
                accuracy
        );
    }

    // Train one model and evaluate it
    private static void evaluateModel(
            String modelName,
            Classifier model,
            Instances training,
            Instances testing) throws Exception {

        System.out.println("\n================================");
        System.out.println(modelName.toUpperCase());
        System.out.println("================================");

        // Train using training records only
        model.buildClassifier(training);

        // Confusion matrix:
        // rows = actual class
        // columns = predicted class
        long[][] confusion = new long[2][2];

        List<PredictionScore> scores = new ArrayList<>();

        int correct = 0;

        for (int i = 0; i < testing.numInstances(); i++) {

            Instance instance = testing.instance(i);

            int actual = (int) instance.classValue();

            double predictedValue =
                    model.classifyInstance(instance);

            int predicted = (int) predictedValue;

            confusion[actual][predicted]++;

            if (actual == predicted) {
                correct++;
            }

            // Probability that this record is LIGHT_ATTACK
            double[] probabilities =
                    model.distributionForInstance(instance);

            scores.add(
                    new PredictionScore(
                            probabilities[LIGHT],
                            actual == LIGHT
                    )
            );
        }

        double accuracy =
                (double) correct / testing.numInstances();

        // Calculate per-class metrics
        double lightPrecision =
                precision(confusion, LIGHT);

        double lightRecall =
                recall(confusion, LIGHT);

        double lightF1 =
                f1(lightPrecision, lightRecall);

        double heavyPrecision =
                precision(confusion, HEAVY);

        double heavyRecall =
                recall(confusion, HEAVY);

        double heavyF1 =
                f1(heavyPrecision, heavyRecall);

        double macroF1 = (lightF1 + heavyF1) / 2.0;

        double auc = calculateAUC(scores);

        // Print evaluation results
        System.out.println("\nPERFORMANCE METRICS");
        System.out.println("--------------------------------");

        System.out.printf("Accuracy: %.4f%%%n", accuracy * 100);
        System.out.printf("Macro F1: %.4f%n", macroF1);
        System.out.printf("ROC-AUC (LIGHT): %.4f%n", auc);

        System.out.println("\nLIGHT_ATTACK:");
        System.out.printf("Precision: %.4f%n", lightPrecision);
        System.out.printf("Recall:    %.4f%n", lightRecall);
        System.out.printf("F1-score:  %.4f%n", lightF1);

        System.out.println("\nHEAVY_ATTACK:");
        System.out.printf("Precision: %.4f%n", heavyPrecision);
        System.out.printf("Recall:    %.4f%n", heavyRecall);
        System.out.printf("F1-score:  %.4f%n", heavyF1);

        System.out.println("\nCONFUSION MATRIX");
        System.out.println("Rows = Actual, Columns = Predicted");
        System.out.println("--------------------------------");

        System.out.printf(
                "%-18s %12s %12s%n",
                "Actual",
                "LIGHT",
                "HEAVY"
        );

        System.out.printf(
                "%-18s %12d %12d%n",
                "LIGHT_ATTACK",
                confusion[LIGHT][LIGHT],
                confusion[LIGHT][HEAVY]
        );

        System.out.printf(
                "%-18s %12d %12d%n",
                "HEAVY_ATTACK",
                confusion[HEAVY][LIGHT],
                confusion[HEAVY][HEAVY]
        );
    }

    // Precision = TP / (TP + FP)
    private static double precision(
            long[][] matrix,
            int classIndex) {

        long truePositive = matrix[classIndex][classIndex];

        long falsePositive = 0;

        for (int i = 0; i < 2; i++) {
            if (i != classIndex) {
                falsePositive += matrix[i][classIndex];
            }
        }

        long denominator = truePositive + falsePositive;

        return denominator == 0
                ? 0.0
                : (double) truePositive / denominator;
    }

    // Recall = TP / (TP + FN)
    private static double recall(
            long[][] matrix,
            int classIndex) {

        long truePositive = matrix[classIndex][classIndex];

        long falseNegative = 0;

        for (int i = 0; i < 2; i++) {
            if (i != classIndex) {
                falseNegative += matrix[classIndex][i];
            }
        }

        long denominator = truePositive + falseNegative;

        return denominator == 0
                ? 0.0
                : (double) truePositive / denominator;
    }

    // F1 = 2 * Precision * Recall / (Precision + Recall)
    private static double f1(
            double precision,
            double recall) {

        if (precision + recall == 0) {
            return 0.0;
        }

        return 2.0 * precision * recall
                / (precision + recall);
    }

    // Calculate ROC-AUC using the rank-based method
    private static double calculateAUC(
            List<PredictionScore> scores) {

        Collections.sort(
                scores,
                Comparator.comparingDouble(
                        item -> item.score
                )
        );

        int positiveCount = 0;
        int negativeCount = 0;

        for (PredictionScore item : scores) {

            if (item.isLight) {
                positiveCount++;
            } else {
                negativeCount++;
            }
        }

        if (positiveCount == 0 || negativeCount == 0) {
            return Double.NaN;
        }

        double positiveRankSum = 0.0;

        int position = 0;

        while (position < scores.size()) {

            int end = position + 1;

            // Find tied prediction scores
            while (end < scores.size()
                    && Double.compare(
                    scores.get(position).score,
                    scores.get(end).score
            ) == 0) {

                end++;
            }

            // Average rank for tied scores
            double averageRank =
                    ((position + 1) + end) / 2.0;

            for (int i = position; i < end; i++) {

                if (scores.get(i).isLight) {
                    positiveRankSum += averageRank;
                }
            }

            position = end;
        }

        double minimumRankSum =
                positiveCount * (positiveCount + 1) / 2.0;

        double uStatistic =
                positiveRankSum - minimumRankSum;

        return uStatistic /
                ((double) positiveCount * negativeCount);
    }
}
