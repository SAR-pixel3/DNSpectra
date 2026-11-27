
package com.dnspectra.evaluation;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class FeaturePatternBaselineAnalyzer {

    private static final Path ROOT = Path.of("data");

    private static final List<String> CLASSES = List.of(
            "BENIGN", "LIGHT_ATTACK", "HEAVY_ATTACK"
    );

    private record Capture(
            String name,
            String label,
            String split
    ) {}

    private record DataRow(
            String pattern,
            String label
    ) {}

    private static class Dataset {
        List<String> featureNames;
        List<DataRow> training = new ArrayList<>();
        List<DataRow> testing = new ArrayList<>();
    }

    public static void main(String[] args) throws Exception {

        System.out.println("DNSPECTRA FEATURE-PATTERN BASELINE ANALYSIS");
        System.out.println("============================================");

        List<Capture> captures = readManifest();

        analyse("STATEFUL", captures, true);
        analyse("STATELESS", captures, false);
    }

    private static List<Capture> readManifest() throws Exception {

        Path manifest = ROOT.resolve("splits/capture-split.csv");

        List<Capture> captures = new ArrayList<>();

        try (Reader reader = Files.newBufferedReader(manifest);
             CSVParser parser = CSVFormat.DEFAULT.builder()
                     .setHeader()
                     .setSkipHeaderRecord(true)
                     .get()
                     .parse(reader)) {

            for (CSVRecord record : parser) {

                captures.add(new Capture(
                        record.get("capture").trim(),
                        record.get("label").trim(),
                        record.get("split").trim().toUpperCase()
                ));
            }
        }

        return captures;
    }

    private static void analyse(
            String featureType,
            List<Capture> captures,
            boolean stateful
    ) throws Exception {

        System.out.println("\n========== " + featureType + " ==========");

        Dataset dataset = new Dataset();

        for (Capture capture : captures) {

            String category = categoryFolder(capture.label());

            String featureFolder = stateful ? "stateful" : "stateless";

            Path baseFolder = stateful
                    ? ROOT.resolve("encoded")
                    : ROOT.resolve("processed");

            String prefix = stateful
                    ? "stateful_features-"
                    : "stateless_features-";

            Path file = baseFolder
                    .resolve(category)
                    .resolve(featureFolder)
                    .resolve(prefix + capture.name() + ".pcap.csv");

            if (!Files.exists(file)) {
                throw new IllegalStateException(
                        "Missing dataset file: " + file
                );
            }

            readDatasetFile(file, capture, dataset);
        }

        printDatasetSummary(dataset);
        calculateBaselines(dataset);
    }

    private static String categoryFolder(String label) {

        return switch (label) {
            case "BENIGN" -> "benign";
            case "LIGHT_ATTACK" -> "light_attack";
            case "HEAVY_ATTACK" -> "heavy_attack";
            default -> throw new IllegalArgumentException(
                    "Unknown label: " + label
            );
        };
    }

    private static void readDatasetFile(
            Path file,
            Capture capture,
            Dataset dataset
    ) throws Exception {

        try (Reader reader = Files.newBufferedReader(file);
             CSVParser parser = CSVFormat.DEFAULT.builder()
                     .setHeader()
                     .setSkipHeaderRecord(true)
                     .get()
                     .parse(reader)) {

            List<String> headers =
                    new ArrayList<>(parser.getHeaderMap().keySet());

            if (!headers.contains("label")) {
                throw new IllegalStateException(
                        "No label column in: " + file
                );
            }

            List<String> features = new ArrayList<>(headers);
            features.remove("label");

            if (dataset.featureNames == null) {
                dataset.featureNames = features;
            } else if (!dataset.featureNames.equals(features)) {
                throw new IllegalStateException(
                        "Feature columns do not match in: " + file
                );
            }

            for (CSVRecord record : parser) {

                String actualLabel = record.get("label").trim();

                if (!actualLabel.equals(capture.label())) {
                    throw new IllegalStateException(
                            "Incorrect label in " + file
                    );
                }

                StringBuilder pattern = new StringBuilder();

                for (String feature : features) {

                    String value = record.get(feature).trim();

                    double number;

                    try {
                        number = Double.parseDouble(value);
                    } catch (NumberFormatException e) {
                        throw new IllegalStateException(
                                "Invalid numeric feature " + feature
                                        + " in " + file
                                        + ": " + value
                        );
                    }

                    if (!Double.isFinite(number)) {
                        throw new IllegalStateException(
                                "Non-finite value in " + file
                        );
                    }

                    if (!pattern.isEmpty()) {
                        pattern.append("|");
                    }

                    // Normalize values such as 1 and 1.0.
                    pattern.append(Double.toString(number));
                }

                DataRow row = new DataRow(
                        pattern.toString(),
                        actualLabel
                );

                if (capture.split().equals("TRAIN")) {
                    dataset.training.add(row);
                } else if (capture.split().equals("TEST")) {
                    dataset.testing.add(row);
                } else {
                    throw new IllegalStateException(
                            "Unknown split: " + capture.split()
                    );
                }
            }
        }

        System.out.println("Loaded: " + file.getFileName()
                + " | " + capture.split());
    }

    private static void printDatasetSummary(Dataset dataset) {

        System.out.println("\nFeatures: "
                + dataset.featureNames.size());

        System.out.println("Training records: "
                + dataset.training.size());

        System.out.println("Testing records: "
                + dataset.testing.size());

        System.out.println("Training unique patterns: "
                + uniquePatterns(dataset.training));

        System.out.println("Testing unique patterns: "
                + uniquePatterns(dataset.testing));
    }

    private static long uniquePatterns(List<DataRow> rows) {

        return rows.stream()
                .map(DataRow::pattern)
                .distinct()
                .count();
    }

    private static void calculateBaselines(Dataset dataset) {

        Map<String, Map<String, Integer>> trainingCounts =
                countLabels(dataset.training);

        Map<String, Map<String, Integer>> testingCounts =
                countLabels(dataset.testing);

        Map<String, Integer> globalTrainingCounts =
                countGlobalLabels(dataset.training);

        String globalMajority =
                majorityLabel(globalTrainingCounts);

        System.out.println("\n--- TRAINING-MAJORITY BASELINE ---");

        System.out.println("Global training majority: "
                + globalMajority);

        long[][] confusion = new long[3][3];

        int unseenRows = 0;
        int matchedRows = 0;

        for (DataRow row : dataset.testing) {

            Map<String, Integer> patternCounts =
                    trainingCounts.get(row.pattern());

            String prediction;

            if (patternCounts == null) {
                // Fallback for a test pattern not seen in training.
                prediction = globalMajority;
                unseenRows++;
            } else {
                prediction = majorityLabel(patternCounts);
                matchedRows++;
            }

            int actualIndex = CLASSES.indexOf(row.label());
            int predictedIndex = CLASSES.indexOf(prediction);

            confusion[actualIndex][predictedIndex]++;
        }

        printConfusionMatrix(confusion);
        printMetrics(confusion);

        System.out.println("Test rows matching a training pattern: "
                + matchedRows);

        System.out.println("Unseen test rows: " + unseenRows);

        System.out.printf(
                Locale.ROOT,
                "Training-pattern coverage: %.4f%%%n",
                100.0 * matchedRows / dataset.testing.size()
        );

        System.out.println("\n--- TEST-SET ORACLE LOWER ERROR ---");

        long oracleCorrect = 0;
        long oracleErrors = 0;
        long conflictingPatterns = 0;
        long conflictingRows = 0;

        for (Map<String, Integer> counts :
                testingCounts.values()) {

            int total = counts.values().stream()
                    .mapToInt(Integer::intValue)
                    .sum();

            int majority = counts.values().stream()
                    .mapToInt(Integer::intValue)
                    .max()
                    .orElse(0);

            oracleCorrect += majority;
            oracleErrors += total - majority;

            if (counts.size() > 1) {
                conflictingPatterns++;
                conflictingRows += total - majority;
            }
        }

        double oracleAccuracy =
                100.0 * oracleCorrect / dataset.testing.size();

        double oracleError =
                100.0 * oracleErrors / dataset.testing.size();

        System.out.println("Conflicting test patterns: "
                + conflictingPatterns);

        System.out.println("Rows in conflicting patterns: "
                + conflictingRows);

        System.out.println("Oracle correct rows: "
                + oracleCorrect);

        System.out.println("Oracle unavoidable errors: "
                + oracleErrors);

        System.out.printf(
                Locale.ROOT,
                "Oracle accuracy ceiling: %.4f%%%n",
                oracleAccuracy
        );

        System.out.printf(
                Locale.ROOT,
                "Oracle minimum error rate: %.4f%%%n",
                oracleError
        );

        System.out.println(
                "\nNOTE: Oracle results use test labels and are not "
                        + "deployable model predictions."
        );
    }

    private static Map<String, Map<String, Integer>> countLabels(
            List<DataRow> rows
    ) {

        Map<String, Map<String, Integer>> result =
                new HashMap<>();

        for (DataRow row : rows) {

            result.computeIfAbsent(
                    row.pattern(),
                    key -> new HashMap<>()
            ).merge(row.label(), 1, Integer::sum);
        }

        return result;
    }

    private static Map<String, Integer> countGlobalLabels(
            List<DataRow> rows
    ) {

        Map<String, Integer> counts = new HashMap<>();

        for (DataRow row : rows) {
            counts.merge(row.label(), 1, Integer::sum);
        }

        return counts;
    }

    private static String majorityLabel(
            Map<String, Integer> counts
    ) {

        String majority = CLASSES.get(0);
        int highestCount = -1;

        // Fixed class order makes ties reproducible.
        for (String label : CLASSES) {

            int count = counts.getOrDefault(label, 0);

            if (count > highestCount) {
                highestCount = count;
                majority = label;
            }
        }

        return majority;
    }

    private static void printConfusionMatrix(long[][] matrix) {

        System.out.println("\nConfusion Matrix");
        System.out.println("Rows = Actual, Columns = Predicted");

        System.out.printf(
                "%-16s %12s %12s %12s%n",
                "Actual",
                "BENIGN",
                "LIGHT",
                "HEAVY"
        );

        for (int i = 0; i < CLASSES.size(); i++) {

            System.out.printf(
                    "%-16s %12d %12d %12d%n",
                    CLASSES.get(i),
                    matrix[i][0],
                    matrix[i][1],
                    matrix[i][2]
            );
        }
    }

    private static void printMetrics(long[][] matrix) {

        long correct = 0;
        long total = 0;

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {

                total += matrix[i][j];

                if (i == j) {
                    correct += matrix[i][j];
                }
            }
        }

        double accuracy = 100.0 * correct / total;
        double macroF1 = 0;

        for (int i = 0; i < 3; i++) {

            long tp = matrix[i][i];
            long fp = 0;
            long fn = 0;

            for (int j = 0; j < 3; j++) {

                if (j != i) {
                    fp += matrix[j][i];
                    fn += matrix[i][j];
                }
            }

            double precision =
                    tp + fp == 0 ? 0 : (double) tp / (tp + fp);

            double recall =
                    tp + fn == 0 ? 0 : (double) tp / (tp + fn);

            double f1 =
                    precision + recall == 0
                            ? 0
                            : 2 * precision * recall
                            / (precision + recall);

            macroF1 += f1;

            System.out.printf(
                    Locale.ROOT,
                    "%s | Precision: %.4f | Recall: %.4f | F1: %.4f%n",
                    CLASSES.get(i),
                    precision,
                    recall,
                    f1
            );
        }

        macroF1 /= 3;

        System.out.printf(
                Locale.ROOT,
                "Accuracy: %.4f%%%n",
                accuracy
        );

        System.out.printf(
                Locale.ROOT,
                "Macro F1: %.4f%n",
                macroF1
        );
    }
}

