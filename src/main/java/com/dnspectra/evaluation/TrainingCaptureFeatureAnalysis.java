
package com.dnspectra.evaluation;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class TrainingCaptureFeatureAnalysis {

    private static final String[][] CAPTURES = {
            {"light_audio", "light_attack"},
            {"light_compressed", "light_attack"},
            {"light_exe", "light_attack"},
            {"light_image", "light_attack"},

            {"heavy_audio", "heavy_attack"},
            {"heavy_compressed", "heavy_attack"},
            {"heavy_exe", "heavy_attack"},
            {"heavy_image", "heavy_attack"}
    };

    private static class FeatureStats {
        double mean;
        double median;
        double standardDeviation;
    }

    public static void main(String[] args) {

        System.out.println("========================================");
        System.out.println("STATEFUL TRAINING CAPTURE FEATURE ANALYSIS");
        System.out.println("========================================");

        Map<String, Map<String, FeatureStats>> results =
                new LinkedHashMap<>();

        List<String> featureNames = new ArrayList<>();

        for (String[] capture : CAPTURES) {

            String captureName = capture[0];
            String category = capture[1];

            String path = "data/encoded/"
                    + category
                    + "/stateful/stateful_features-"
                    + captureName
                    + ".pcap.csv";

            File file = new File(path);

            if (!file.exists()) {
                System.out.println("FAILED: " + path);
                continue;
            }

            try {

                CSVLoader loader = new CSVLoader();
                loader.setSource(file);

                Instances data = loader.getDataSet();

                System.out.println(
                        "\nLoaded: " + captureName
                                + " | Records: " + data.numInstances()
                );

                Map<String, FeatureStats> captureResults =
                        new LinkedHashMap<>();

                for (int a = 0; a < data.numAttributes(); a++) {

                    Attribute attribute = data.attribute(a);

                    if (!attribute.isNumeric()) {
                        continue;
                    }

                    String featureName = attribute.name();

                    List<Double> values = new ArrayList<>();

                    double sum = 0.0;

                    for (int i = 0; i < data.numInstances(); i++) {

                        Instance instance = data.instance(i);

                        if (instance.isMissing(a)) {
                            continue;
                        }

                        double value = instance.value(a);

                        values.add(value);
                        sum += value;
                    }

                    if (values.isEmpty()) {
                        continue;
                    }

                    Collections.sort(values);

                    FeatureStats stats = new FeatureStats();

                    stats.mean = sum / values.size();

                    int middle = values.size() / 2;

                    if (values.size() % 2 == 0) {
                        stats.median = (
                                values.get(middle - 1)
                                        + values.get(middle)
                        ) / 2.0;
                    } else {
                        stats.median = values.get(middle);
                    }

                    double squaredDifferenceSum = 0.0;

                    for (double value : values) {

                        double difference = value - stats.mean;

                        squaredDifferenceSum +=
                                difference * difference;
                    }

                    stats.standardDeviation = Math.sqrt(
                            squaredDifferenceSum / values.size()
                    );

                    captureResults.put(featureName, stats);

                    if (!featureNames.contains(featureName)) {
                        featureNames.add(featureName);
                    }
                }

                results.put(captureName, captureResults);

            } catch (Exception e) {

                System.out.println(
                        "ERROR loading " + captureName
                );

                e.printStackTrace();
            }
        }

        printCaptureResults(results, featureNames);

        printClassComparison(results, featureNames);
    }

    private static void printCaptureResults(
            Map<String, Map<String, FeatureStats>> results,
            List<String> featureNames) {

        System.out.println("\n\n========================================");
        System.out.println("INDIVIDUAL CAPTURE FEATURE STATISTICS");
        System.out.println("========================================");

        for (String captureName : results.keySet()) {

            System.out.println("\n========== " + captureName + " ==========");

            System.out.printf(
                    "%-23s %15s %15s %15s%n",
                    "Feature",
                    "Mean",
                    "Median",
                    "Std Deviation"
            );

            System.out.println(
                    "--------------------------------------------------------------------------"
            );

            Map<String, FeatureStats> captureResults =
                    results.get(captureName);

            for (String feature : featureNames) {

                FeatureStats stats = captureResults.get(feature);

                if (stats == null) {
                    continue;
                }

                System.out.printf(
                        "%-23s %15.4f %15.4f %15.4f%n",
                        feature,
                        stats.mean,
                        stats.median,
                        stats.standardDeviation
                );
            }
        }
    }

    private static void printClassComparison(
            Map<String, Map<String, FeatureStats>> results,
            List<String> featureNames) {

        System.out.println("\n\n========================================");
        System.out.println("LIGHT VS HEAVY TRAINING CAPTURE COMPARISON");
        System.out.println("========================================");

        System.out.println(
                "Values below are equally weighted averages "
                        + "of the capture-level statistics."
        );

        System.out.printf(
                "\n%-23s %15s %15s %15s%n",
                "Feature",
                "LIGHT Mean",
                "HEAVY Mean",
                "Difference"
        );

        System.out.println(
                "--------------------------------------------------------------------------"
        );

        for (String feature : featureNames) {

            double lightTotal = 0.0;
            double heavyTotal = 0.0;

            int lightCount = 0;
            int heavyCount = 0;

            for (String captureName : results.keySet()) {

                FeatureStats stats =
                        results.get(captureName).get(feature);

                if (stats == null) {
                    continue;
                }

                if (captureName.startsWith("light_")) {
                    lightTotal += stats.mean;
                    lightCount++;
                } else if (captureName.startsWith("heavy_")) {
                    heavyTotal += stats.mean;
                    heavyCount++;
                }
            }

            if (lightCount == 0 || heavyCount == 0) {
                continue;
            }

            double lightAverage = lightTotal / lightCount;
            double heavyAverage = heavyTotal / heavyCount;

            double difference = lightAverage - heavyAverage;

            System.out.printf(
                    "%-23s %15.4f %15.4f %15.4f%n",
                    feature,
                    lightAverage,
                    heavyAverage,
                    difference
            );
        }
    }
}
