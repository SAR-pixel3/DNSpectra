
package com.dnspectra.evaluation;

import com.dnspectra.data.WekaDatasetBuilder;

import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class BinaryThresholdExperiment {

    private static final double[] THRESHOLDS = {
            0.10, 0.20, 0.30, 0.40, 0.50
    };

    private static final int TREES = 100;
    private static final int SEED = 42;
    private static final int FOLDS = 5;

    public static void main(String[] args) {

        System.out.println("========================================");
        System.out.println("FINAL BINARY THRESHOLD EXPERIMENT");
        System.out.println("========================================");

        runExperiment("STATEFUL");
        runExperiment("STATELESS");
    }

    private static void runExperiment(String featureGroup) {

        System.out.println("\n\n========================================");
        System.out.println(featureGroup + " FEATURE GROUP");
        System.out.println("========================================");

        try {

            Instances[] datasets =
                    WekaDatasetBuilder.loadExperiment(featureGroup);

            Instances fullTraining =
                    createBinaryDataset(datasets[0]);

            Instances testing =
                    createBinaryDataset(datasets[1]);

            // Randomize and stratify the training dataset.
            // This creates a separate validation fold.
            fullTraining.randomize(new Random(SEED));
            fullTraining.stratify(FOLDS);

            Instances validation =
                    fullTraining.testCV(FOLDS, 0);

            Instances modelTraining =
                    fullTraining.trainCV(FOLDS, 0);

            System.out.println("\nDATASET SUMMARY");
            System.out.println("------------------------------");

            System.out.println(
                    "Original training records: "
                            + fullTraining.numInstances()
            );

            System.out.println(
                    "Model-training records: "
                            + modelTraining.numInstances()
            );

            System.out.println(
                    "Validation records: "
                            + validation.numInstances()
            );

            System.out.println(
                    "Held-out test records: "
                            + testing.numInstances()
            );

            System.out.println(
                    "Binary class labels: "
                            + fullTraining.classAttribute()
                            .value(0)
                            + ", "
                            + fullTraining.classAttribute()
                            .value(1)
            );

            System.out.println("\nCLASS DISTRIBUTION");

            printClassDistribution(
                    "Model training",
                    modelTraining
            );

            printClassDistribution(
                    "Validation",
                    validation
            );

            printClassDistribution(
                    "Test",
                    testing
            );

            // Phase 1: Train using only the model-training fold.
            RandomForest validationModel =
                    createRandomForest();

            validationModel.buildClassifier(modelTraining);

            // Evaluate thresholds on separate validation records.
            List<double[]> validationScores =
                    getScores(validationModel, validation);

            System.out.println("\nVALIDATION THRESHOLD RESULTS");

            System.out.println(
                    "----------------------------------------------------------------"
            );

            System.out.printf(
                    "%-10s %-12s %-12s %-12s %-12s %-12s%n",
                    "Threshold",
                    "Precision",
                    "Recall",
                    "F1",
                    "Macro F1",
                    "HEAVY FPR"
            );

            double bestThreshold = THRESHOLDS[0];
            double bestMacroF1 = -1.0;

            for (double threshold : THRESHOLDS) {

                Metrics metrics =
                        calculateMetrics(
                                validationScores,
                                threshold
                        );

                System.out.printf(
                        "%-10.2f %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f%n",
                        threshold,
                        metrics.precision,
                        metrics.recall,
                        metrics.f1,
                        metrics.macroF1,
                        metrics.falsePositiveRate
                );

                if (metrics.macroF1 > bestMacroF1) {

                    bestMacroF1 = metrics.macroF1;
                    bestThreshold = threshold;
                }
            }

            System.out.println("\nSELECTED THRESHOLD");
            System.out.println("------------------------------");

            System.out.printf(
                    "Best validation threshold: %.2f%n",
                    bestThreshold
            );

            System.out.printf(
                    "Validation macro F1: %.4f%n",
                    bestMacroF1
            );

            // Phase 2: Train a fresh final model using ALL
            // available training records.
            RandomForest finalModel =
                    createRandomForest();

            finalModel.buildClassifier(fullTraining);

            System.out.println(
                    "\nFinal model trained using all training records."
            );

            // The held-out test dataset is evaluated only now.
            List<double[]> testScores =
                    getScores(finalModel, testing);

            Metrics finalMetrics =
                    calculateMetrics(
                            testScores,
                            bestThreshold
                    );

            System.out.println("\nFINAL HELD-OUT TEST RESULTS");
            System.out.println("------------------------------");

            System.out.printf(
                    "Selected threshold: %.2f%n",
                    bestThreshold
            );

            System.out.printf(
                    "Accuracy: %.4f%%%n",
                    finalMetrics.accuracy * 100
            );

            System.out.printf(
                    "LIGHT precision: %.4f%n",
                    finalMetrics.precision
            );

            System.out.printf(
                    "LIGHT recall: %.4f%n",
                    finalMetrics.recall
            );

            System.out.printf(
                    "LIGHT F1: %.4f%n",
                    finalMetrics.f1
            );

            System.out.printf(
                    "HEAVY precision: %.4f%n",
                    finalMetrics.heavyPrecision
            );

            System.out.printf(
                    "HEAVY recall: %.4f%n",
                    finalMetrics.heavyRecall
            );

            System.out.printf(
                    "HEAVY false-positive rate: %.4f%n",
                    finalMetrics.falsePositiveRate
            );

            System.out.printf(
                    "Macro F1: %.4f%n",
                    finalMetrics.macroF1
            );

            System.out.println("\nConfusion Matrix:");
            System.out.println("                 Predicted");
            System.out.println("               LIGHT   HEAVY");

            System.out.printf(
                    "Actual LIGHT  %7d %7d%n",
                    finalMetrics.tp,
                    finalMetrics.fn
            );

            System.out.printf(
                    "Actual HEAVY  %7d %7d%n",
                    finalMetrics.fp,
                    finalMetrics.tn
            );

        } catch (Exception e) {

            System.out.println(
                    "Experiment failed: " + featureGroup
            );

            e.printStackTrace();
        }
    }

    private static RandomForest createRandomForest() {

        RandomForest model = new RandomForest();

        model.setNumIterations(TREES);
        model.setSeed(SEED);

        return model;
    }

    private static Instances createBinaryDataset(
            Instances source) {

        int originalClassIndex = source.classIndex();

        if (originalClassIndex < 0) {
            throw new IllegalArgumentException(
                    "Source dataset has no class attribute."
            );
        }

        int lightIndex =
                source.classAttribute().indexOfValue(
                        "LIGHT_ATTACK"
                );

        int heavyIndex =
                source.classAttribute().indexOfValue(
                        "HEAVY_ATTACK"
                );

        if (lightIndex < 0 || heavyIndex < 0) {
            throw new IllegalArgumentException(
                    "LIGHT_ATTACK or HEAVY_ATTACK class not found."
            );
        }

        // Copy all original feature attributes.
        // The new binary class attribute will be added last.
        ArrayList<Attribute> attributes =
                new ArrayList<>();

        for (int i = 0; i < source.numAttributes(); i++) {

            if (i != originalClassIndex) {

                attributes.add(
                        (Attribute) source.attribute(i).copy()
                );
            }
        }

        // Create a genuinely binary class attribute.
        ArrayList<String> classValues =
                new ArrayList<>();

        classValues.add("LIGHT_ATTACK");
        classValues.add("HEAVY_ATTACK");

        attributes.add(
                new Attribute("label", classValues)
        );

        Instances binary =
                new Instances(
                        source.relationName() + "_binary",
                        attributes,
                        source.numInstances()
                );

        binary.setClassIndex(binary.numAttributes() - 1);

        // Copy records and translate their original labels.
        for (int row = 0; row < source.numInstances(); row++) {

            weka.core.Instance original =
                    source.instance(row);

            int originalLabel =
                    (int) original.classValue();

            // Exclude BENIGN records.
            if (originalLabel != lightIndex
                    && originalLabel != heavyIndex) {

                continue;
            }

            double[] values =
                    new double[binary.numAttributes()];

            int newAttributeIndex = 0;

            for (int oldAttributeIndex = 0;
                 oldAttributeIndex < source.numAttributes();
                 oldAttributeIndex++) {

                if (oldAttributeIndex == originalClassIndex) {
                    continue;
                }

                if (original.isMissing(oldAttributeIndex)) {

                    values[newAttributeIndex] =
                            weka.core.Utils.missingValue();

                } else {

                    values[newAttributeIndex] =
                            original.value(oldAttributeIndex);
                }

                newAttributeIndex++;
            }

            // 0 = LIGHT_ATTACK
            // 1 = HEAVY_ATTACK
            values[binary.classIndex()] =
                    originalLabel == lightIndex ? 0 : 1;

            binary.add(
                    new DenseInstance(
                            original.weight(),
                            values
                    )
            );
        }

        return binary;
    }

    private static void printClassDistribution(
            String name,
            Instances data) {

        int light = 0;
        int heavy = 0;

        for (int i = 0; i < data.numInstances(); i++) {

            if ((int) data.instance(i).classValue() == 0) {
                light++;
            } else {
                heavy++;
            }
        }

        System.out.println(
                name + ": LIGHT=" + light
                        + ", HEAVY=" + heavy
        );
    }

    private static List<double[]> getScores(
            RandomForest model,
            Instances data) throws Exception {

        List<double[]> scores = new ArrayList<>();

        for (int i = 0; i < data.numInstances(); i++) {

            double[] probabilities =
                    model.distributionForInstance(
                            data.instance(i)
                    );

            int actual =
                    (int) data.instance(i).classValue();

            scores.add(
                    new double[]{
                            probabilities[0],
                            actual
                    }
            );
        }

        return scores;
    }

    private static Metrics calculateMetrics(
            List<double[]> scores,
            double threshold) {

        Metrics result = new Metrics();

        for (double[] score : scores) {

            double probabilityLight = score[0];
            int actual = (int) score[1];

            int predicted =
                    probabilityLight >= threshold ? 0 : 1;

            if (actual == 0 && predicted == 0) {
                result.tp++;
            } else if (actual == 0 && predicted == 1) {
                result.fn++;
            } else if (actual == 1 && predicted == 0) {
                result.fp++;
            } else {
                result.tn++;
            }
        }

        double total =
                result.tp + result.fn + result.fp + result.tn;

        result.accuracy =
                total == 0 ? 0 :
                        (result.tp + result.tn) / total;

        result.precision =
                divide(
                        result.tp,
                        result.tp + result.fp
                );

        result.recall =
                divide(
                        result.tp,
                        result.tp + result.fn
                );

        result.f1 =
                harmonicMean(
                        result.precision,
                        result.recall
                );

        result.heavyPrecision =
                divide(
                        result.tn,
                        result.tn + result.fn
                );

        result.heavyRecall =
                divide(
                        result.tn,
                        result.tn + result.fp
                );

        result.falsePositiveRate =
                divide(
                        result.fp,
                        result.fp + result.tn
                );

        double heavyF1 =
                harmonicMean(
                        result.heavyPrecision,
                        result.heavyRecall
                );

        result.macroF1 =
                (result.f1 + heavyF1) / 2.0;

        return result;
    }

    private static double divide(
            double numerator,
            double denominator) {

        return denominator == 0
                ? 0.0
                : numerator / denominator;
    }

    private static double harmonicMean(
            double a,
            double b) {

        return a + b == 0
                ? 0.0
                : 2.0 * a * b / (a + b);
    }

    private static class Metrics {

        int tp;
        int fn;
        int fp;
        int tn;

        double accuracy;
        double precision;
        double recall;
        double f1;
        double heavyPrecision;
        double heavyRecall;
        double falsePositiveRate;
        double macroF1;
    }
}
