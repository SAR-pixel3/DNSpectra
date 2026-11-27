
package com.dnspectra.evaluation;

import com.dnspectra.data.WekaDatasetBuilder;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class BinaryFeatureSubsetExperiment {

    private static final int[] FEATURE_COUNTS = {3, 5, 8};

    private static class FeatureScore {
        int index;
        String name;
        double auc;

        FeatureScore(int index, String name, double auc) {
            this.index = index;
            this.name = name;
            this.auc = auc;
        }
    }

    private static class ValueLabel {
        double value;
        boolean isLight;

        ValueLabel(double value, boolean isLight) {
            this.value = value;
            this.isLight = isLight;
        }
    }

    public static void main(String[] args) {

        System.out.println("========================================");
        System.out.println("BINARY RANDOM FOREST FEATURE SUBSET EXPERIMENT");
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

            if (datasets == null || datasets.length < 2) {
                System.out.println("Dataset loading failed.");
                return;
            }

            Instances training =
                    createBinaryDataset(datasets[0]);

            Instances testing =
                    createBinaryDataset(datasets[1]);

            System.out.println("\nBINARY DATASET");
            System.out.println("------------------------------");
            System.out.println("Training records: " + training.numInstances());
            System.out.println("Testing records: " + testing.numInstances());
            System.out.println("Features: " + (training.numAttributes() - 1));

            System.out.println(
                    "LIGHT training: " + countClass(training, 0)
            );

            System.out.println(
                    "HEAVY training: " + countClass(training, 1)
            );

            List<FeatureScore> scores =
                    rankFeatures(training);

            System.out.println("\nFEATURE RANKING");
            System.out.println("------------------------------");

            for (int i = 0; i < scores.size(); i++) {

                FeatureScore score = scores.get(i);

                System.out.printf(
                        "%2d. %-23s AUC: %.4f%n",
                        i + 1,
                        score.name,
                        score.auc
                );
            }

            // Experiment A: All features
            evaluateSubset(
                    "ALL FEATURES",
                    training,
                    testing,
                    null
            );

            // Experiments B, C and D
            for (int count : FEATURE_COUNTS) {

                int actualCount = Math.min(count, scores.size());

                int[] selectedIndices = new int[actualCount];

                System.out.println(
                        "\nSelected top " + actualCount + " features:"
                );

                for (int i = 0; i < actualCount; i++) {

                    FeatureScore score = scores.get(i);

                    selectedIndices[i] = score.index;

                    System.out.println(
                            " - " + score.name
                                    + " | AUC: "
                                    + String.format("%.4f", score.auc)
                    );
                }

                evaluateSubset(
                        "TOP " + actualCount + " FEATURES",
                        training,
                        testing,
                        selectedIndices
                );
            }

        } catch (Exception e) {

            System.out.println(
                    "Experiment failed for " + featureGroup
            );

            e.printStackTrace();
        }
    }

    private static Instances createBinaryDataset(
            Instances source) throws Exception {

        int lightIndex =
                source.classAttribute().indexOfValue("LIGHT_ATTACK");

        int heavyIndex =
                source.classAttribute().indexOfValue("HEAVY_ATTACK");

        if (lightIndex < 0 || heavyIndex < 0) {
            throw new IllegalArgumentException(
                    "LIGHT_ATTACK or HEAVY_ATTACK class not found."
            );
        }

        ArrayList<Attribute> attributes = new ArrayList<>();

        for (int i = 0; i < source.numAttributes(); i++) {

            if (i == source.classIndex()) {
                continue;
            }

            Attribute original = source.attribute(i);

            if (!original.isNumeric()) {
                throw new IllegalArgumentException(
                        "Unexpected non-numeric feature: "
                                + original.name()
                );
            }

            attributes.add(new Attribute(original.name()));
        }

        ArrayList<String> classValues = new ArrayList<>();

        classValues.add("LIGHT_ATTACK");
        classValues.add("HEAVY_ATTACK");

        attributes.add(new Attribute("label", classValues));

        Instances binary = new Instances(
                "BinaryAttackDataset",
                attributes,
                source.numInstances()
        );

        binary.setClassIndex(binary.numAttributes() - 1);

        for (int i = 0; i < source.numInstances(); i++) {

            Instance original = source.instance(i);

            int originalClass =
                    (int) original.classValue();

            if (originalClass != lightIndex
                    && originalClass != heavyIndex) {
                continue;
            }

            DenseInstance copy =
                    new DenseInstance(binary.numAttributes());

            copy.setDataset(binary);

            int targetIndex = 0;

            for (int a = 0; a < source.numAttributes(); a++) {

                if (a == source.classIndex()) {
                    continue;
                }

                if (original.isMissing(a)) {
                    copy.setMissing(targetIndex);
                } else {
                    copy.setValue(
                            targetIndex,
                            original.value(a)
                    );
                }

                targetIndex++;
            }

            if (originalClass == lightIndex) {
                copy.setValue(
                        binary.classAttribute(),
                        "LIGHT_ATTACK"
                );
            } else {
                copy.setValue(
                        binary.classAttribute(),
                        "HEAVY_ATTACK"
                );
            }

            binary.add(copy);
        }

        return binary;
    }

    private static int countClass(
            Instances data,
            int classIndex) {

        int count = 0;

        for (int i = 0; i < data.numInstances(); i++) {

            if ((int) data.instance(i).classValue() == classIndex) {
                count++;
            }
        }

        return count;
    }

    private static List<FeatureScore> rankFeatures(
            Instances training) {

        List<FeatureScore> scores = new ArrayList<>();

        for (int a = 0; a < training.numAttributes(); a++) {

            if (a == training.classIndex()) {
                continue;
            }

            List<ValueLabel> values = new ArrayList<>();

            for (int i = 0; i < training.numInstances(); i++) {

                Instance instance = training.instance(i);

                if (instance.isMissing(a)) {
                    continue;
                }

                values.add(
                        new ValueLabel(
                                instance.value(a),
                                (int) instance.classValue() == 0
                        )
                );
            }

            double rawAuc = calculateAuc(values);

            double adjustedAuc =
                    Math.max(rawAuc, 1.0 - rawAuc);

            scores.add(
                    new FeatureScore(
                            a,
                            training.attribute(a).name(),
                            adjustedAuc
                    )
            );
        }

        scores.sort(
                Comparator.comparingDouble(
                        (FeatureScore score) -> score.auc
                ).reversed()
        );

        return scores;
    }

    private static double calculateAuc(
            List<ValueLabel> values) {

        List<ValueLabel> sorted =
                new ArrayList<>(values);

        Collections.sort(
                sorted,
                Comparator.comparingDouble(v -> v.value)
        );

        int lightCount = 0;
        int heavyCount = 0;

        for (ValueLabel value : sorted) {

            if (value.isLight) {
                lightCount++;
            } else {
                heavyCount++;
            }
        }

        if (lightCount == 0 || heavyCount == 0) {
            return 0.5;
        }

        double rankSumLight = 0.0;

        int i = 0;

        while (i < sorted.size()) {

            int start = i;

            double currentValue = sorted.get(i).value;

            while (i < sorted.size()
                    && Double.compare(
                    sorted.get(i).value,
                    currentValue
            ) == 0) {
                i++;
            }

            int end = i;

            double averageRank =
                    ((start + 1) + end) / 2.0;

            for (int j = start; j < end; j++) {

                if (sorted.get(j).isLight) {
                    rankSumLight += averageRank;
                }
            }
        }

        double uStatistic =
                rankSumLight
                        - (lightCount * (lightCount + 1)) / 2.0;

        return uStatistic /
                ((double) lightCount * heavyCount);
    }

    private static void evaluateSubset(
            String experimentName,
            Instances originalTraining,
            Instances originalTesting,
            int[] selectedIndices) throws Exception {

        Instances training;
        Instances testing;

        if (selectedIndices == null) {

            training = new Instances(originalTraining);
            testing = new Instances(originalTesting);

        } else {

            int[] keepIndices =
                    new int[selectedIndices.length + 1];

            System.arraycopy(
                    selectedIndices,
                    0,
                    keepIndices,
                    0,
                    selectedIndices.length
            );

            keepIndices[selectedIndices.length] =
                    originalTraining.classIndex();

            Remove remove = new Remove();

            remove.setAttributeIndicesArray(keepIndices);
            remove.setInvertSelection(true);

            remove.setInputFormat(originalTraining);

            training = Filter.useFilter(
                    originalTraining,
                    remove
            );

            testing = Filter.useFilter(
                    originalTesting,
                    remove
            );

            training.setClassIndex(
                    training.numAttributes() - 1
            );

            testing.setClassIndex(
                    testing.numAttributes() - 1
            );
        }

        RandomForest model = new RandomForest();

        model.setNumIterations(100);
        model.setSeed(42);

        model.buildClassifier(training);

        evaluateModel(
                experimentName,
                model,
                testing
        );
    }

    private static void evaluateModel(
            String experimentName,
            Classifier model,
            Instances testing) throws Exception {

        double[][] confusion = new double[2][2];

        List<ValueLabel> predictionScores =
                new ArrayList<>();

        int correct = 0;

        for (int i = 0; i < testing.numInstances(); i++) {

            Instance instance = testing.instance(i);

            int actual = (int) instance.classValue();

            double prediction =
                    model.classifyInstance(instance);

            int predicted = (int) prediction;

            confusion[actual][predicted]++;

            if (actual == predicted) {
                correct++;
            }

            double[] probabilities =
                    model.distributionForInstance(instance);

            predictionScores.add(
                    new ValueLabel(
                            probabilities[0],
                            actual == 0
                    )
            );
        }

        double accuracy =
                correct / (double) testing.numInstances();

        double lightPrecision =
                precision(confusion, 0);

        double lightRecall =
                recall(confusion, 0);

        double lightF1 =
                f1(lightPrecision, lightRecall);

        double heavyPrecision =
                precision(confusion, 1);

        double heavyRecall =
                recall(confusion, 1);

        double heavyF1 =
                f1(heavyPrecision, heavyRecall);

        double macroF1 =
                (lightF1 + heavyF1) / 2.0;

        double auc = calculateAuc(predictionScores);

        System.out.println("\n----------------------------------------");
        System.out.println("EXPERIMENT: " + experimentName);
        System.out.println("----------------------------------------");

        System.out.printf("Accuracy: %.4f%%%n", accuracy * 100);
        System.out.printf("Macro F1: %.4f%n", macroF1);

        System.out.printf(
                "LIGHT Precision: %.4f%n",
                lightPrecision
        );

        System.out.printf(
                "LIGHT Recall: %.4f%n",
                lightRecall
        );

        System.out.printf(
                "LIGHT F1: %.4f%n",
                lightF1
        );

        System.out.printf(
                "HEAVY Recall: %.4f%n",
                heavyRecall
        );

        System.out.printf(
                "ROC-AUC (LIGHT): %.4f%n",
                auc
        );

        System.out.println("\nConfusion Matrix:");
        System.out.println("                 Predicted");
        System.out.println("               LIGHT   HEAVY");

        System.out.printf(
                "Actual LIGHT  %7.0f %7.0f%n",
                confusion[0][0],
                confusion[0][1]
        );

        System.out.printf(
                "Actual HEAVY  %7.0f %7.0f%n",
                confusion[1][0],
                confusion[1][1]
        );
    }

    private static double precision(
            double[][] matrix,
            int classIndex) {

        double tp = matrix[classIndex][classIndex];

        double fp = 0.0;

        for (int i = 0; i < 2; i++) {

            if (i != classIndex) {
                fp += matrix[i][classIndex];
            }
        }

        if (tp + fp == 0) {
            return 0.0;
        }

        return tp / (tp + fp);
    }

    private static double recall(
            double[][] matrix,
            int classIndex) {

        double tp = matrix[classIndex][classIndex];

        double fn = 0.0;

        for (int i = 0; i < 2; i++) {

            if (i != classIndex) {
                fn += matrix[classIndex][i];
            }
        }

        if (tp + fn == 0) {
            return 0.0;
        }

        return tp / (tp + fn);
    }

    private static double f1(
            double precision,
            double recall) {

        if (precision + recall == 0) {
            return 0.0;
        }

        return 2.0 * precision * recall
                / (precision + recall);
    }
}
