
package com.dnspectra.evaluation;

import com.dnspectra.data.WekaDatasetBuilder;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import weka.core.converters.CSVLoader;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.Logistic;

import java.io.File;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

public class BinaryCaptureAnalysis {

    private static final int LIGHT = 0;
    private static final int HEAVY = 1;

    private static class CaptureResult {

        String capture;
        String actualClass;

        int total;
        int correct;

        int lightCorrect;
        int lightTotal;

        int heavyCorrect;
        int heavyTotal;

        int predictedLight;
        int predictedHeavy;
    }

    public static void main(String[] args) throws Exception {

        runExperiment("STATEFUL");
        runExperiment("STATELESS");
    }

    private static void runExperiment(String featureGroup)
            throws Exception {

        System.out.println("\n================================");
        System.out.println(
                featureGroup + " - CAPTURE LEVEL ANALYSIS"
        );
        System.out.println("================================");

        // Load the original capture-level training split
        Instances[] datasets =
                WekaDatasetBuilder.loadExperiment(featureGroup);

        Instances training =
                filterAttackClasses(datasets[0]);

        System.out.println("\nBINARY TRAINING DATA");
        System.out.println("--------------------------------");
        System.out.println(
                "Training records: " + training.numInstances()
        );
        System.out.println(
                "Features: " + (training.numAttributes() - 1)
        );

        // Prepare the three classifiers
        Classifier[] models = new Classifier[3];

        models[0] = new J48();

        RandomForest forest = new RandomForest();
        forest.setNumIterations(100);
        forest.setSeed(42);
        models[1] = forest;

        Logistic logistic = new Logistic();
        logistic.setMaxIts(200);
        models[2] = logistic;

        String[] modelNames = {
                "Decision Tree",
                "Random Forest",
                "Logistic Regression"
        };

        // Train each model once
        for (int i = 0; i < models.length; i++) {

            System.out.println(
                    "\nTraining " + modelNames[i] + "..."
            );

            models[i].buildClassifier(training);
        }

        // Analyse the four held-out attack captures
        for (int modelIndex = 0;
             modelIndex < models.length;
             modelIndex++) {

            System.out.println("\n================================");
            System.out.println(
                    modelNames[modelIndex].toUpperCase()
            );
            System.out.println("================================");

            Map<String, CaptureResult> results =
                    new LinkedHashMap<>();

            analyseCapture(
                    "light_text",
                    "LIGHT_ATTACK",
                    featureGroup,
                    models[modelIndex],
                    training,
                    results
            );

            analyseCapture(
                    "light_video",
                    "LIGHT_ATTACK",
                    featureGroup,
                    models[modelIndex],
                    training,
                    results
            );

            analyseCapture(
                    "heavy_text",
                    "HEAVY_ATTACK",
                    featureGroup,
                    models[modelIndex],
                    training,
                    results
            );

            analyseCapture(
                    "heavy_video",
                    "HEAVY_ATTACK",
                    featureGroup,
                    models[modelIndex],
                    training,
                    results
            );

            printResults(results);
        }
    }

    // Convert three classes into a genuine two-class dataset
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

        // Create binary class attribute
        FastVector classValues = new FastVector();

        classValues.addElement("LIGHT_ATTACK");
        classValues.addElement("HEAVY_ATTACK");

        attributes.addElement(
                new Attribute("label", classValues)
        );

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

            for (int j = 0; j < original.numAttributes(); j++) {

                if (j == original.classIndex()) {
                    continue;
                }

                values[newIndex++] = source.value(j);
            }

            values[newIndex] =
                    oldLabel == lightIndex ? LIGHT : HEAVY;

            filtered.add(
                    new DenseInstance(source.weight(), values)
            );
        }

        return filtered;
    }

    // Analyse one held-out capture
    private static void analyseCapture(
            String captureName,
            String actualClass,
            String featureGroup,
            Classifier model,
            Instances training,
            Map<String, CaptureResult> results)
            throws Exception {

        Path csvPath = getCapturePath(
                captureName,
                actualClass,
                featureGroup
        );

        if (!csvPath.toFile().exists()) {

            throw new IllegalStateException(
                    "Capture file not found: " + csvPath
            );
        }

        System.out.println("\nLoading: " + csvPath);

        // Load the individual capture and match training attributes
        Instances capture = loadCaptureFile(
                csvPath.toFile(),
                training,
                actualClass
        );

        CaptureResult result = new CaptureResult();

        result.capture = captureName;
        result.actualClass = actualClass;

        int expectedLabel =
                actualClass.equals("LIGHT_ATTACK")
                        ? LIGHT : HEAVY;

        for (int i = 0; i < capture.numInstances(); i++) {

            Instance instance = capture.instance(i);

            int predicted =
                    (int) model.classifyInstance(instance);

            result.total++;

            if (predicted == expectedLabel) {
                result.correct++;
            }

            if (expectedLabel == LIGHT) {

                result.lightTotal++;

                if (predicted == LIGHT) {
                    result.lightCorrect++;
                }

            } else {

                result.heavyTotal++;

                if (predicted == HEAVY) {
                    result.heavyCorrect++;
                }
            }

            if (predicted == LIGHT) {
                result.predictedLight++;
            } else {
                result.predictedHeavy++;
            }
        }

        results.put(captureName, result);
    }

    // Determine the exact CSV path from the project structure
    private static Path getCapturePath(
            String captureName,
            String actualClass,
            String featureGroup) {

        String category =
                actualClass.equals("LIGHT_ATTACK")
                        ? "light_attack"
                        : "heavy_attack";

        String subfolder =
                featureGroup.toLowerCase();

        String filename =
                subfolder + "_features-"
                        + captureName + ".pcap.csv";

        String root =
                featureGroup.equals("STATEFUL")
                        ? "encoded"
                        : "processed";

        return Path.of(
                "data",
                root,
                category,
                subfolder,
                filename
        );
    }

    // Load a capture and match its features to the training structure
    private static Instances loadCaptureFile(
            File csvFile,
            Instances training,
            String actualClass) throws Exception {

        CSVLoader loader = new CSVLoader();
        loader.setSource(csvFile);

        Instances sourceData = loader.getDataSet();

        Instances converted = new Instances(training, 0);

        int expectedLabel =
                actualClass.equals("LIGHT_ATTACK")
                        ? LIGHT : HEAVY;

        // Verify every training feature exists in the CSV
        for (int j = 0; j < training.numAttributes(); j++) {

            if (j == training.classIndex()) {
                continue;
            }

            String featureName =
                    training.attribute(j).name();

            if (sourceData.attribute(featureName) == null) {

                throw new IllegalStateException(
                        "Feature missing from " + csvFile.getName()
                                + ": " + featureName
                );
            }
        }

        // Convert rows into the exact training format
        for (int i = 0; i < sourceData.numInstances(); i++) {

            Instance source = sourceData.instance(i);

            double[] values =
                    new double[training.numAttributes()];

            for (int j = 0; j < training.numAttributes(); j++) {

                Attribute target =
                        training.attribute(j);

                if (j == training.classIndex()) {

                    values[j] = expectedLabel;

                } else {

                    int sourceIndex =
                            sourceData.attribute(
                                    target.name()
                            ).index();

                    values[j] =
                            source.value(sourceIndex);
                }
            }

            converted.add(
                    new DenseInstance(source.weight(), values)
            );
        }

        return converted;
    }

    // Print capture-level results
    private static void printResults(
            Map<String, CaptureResult> results) {

        System.out.println("\nCAPTURE RESULTS");
        System.out.println("---------------------------------------------------------------");

        System.out.printf(
                "%-15s %-14s %10s %10s %12s%n",
                "Capture",
                "Actual Class",
                "Records",
                "Accuracy",
                "Correct"
        );

        System.out.println("---------------------------------------------------------------");

        for (CaptureResult result : results.values()) {

            double accuracy =
                    100.0 * result.correct / result.total;

            System.out.printf(
                    "%-15s %-14s %10d %9.2f%% %12d%n",
                    result.capture,
                    result.actualClass,
                    result.total,
                    accuracy,
                    result.correct
            );

            System.out.println(
                    "  Predicted LIGHT: "
                            + result.predictedLight
                            + " | Predicted HEAVY: "
                            + result.predictedHeavy
            );

            if (result.lightTotal > 0) {

                System.out.printf(
                        "  LIGHT recall: %.4f%n",
                        (double) result.lightCorrect
                                / result.lightTotal
                );
            }

            if (result.heavyTotal > 0) {

                System.out.printf(
                        "  HEAVY recall: %.4f%n",
                        (double) result.heavyCorrect
                                / result.heavyTotal
                );
            }
        }

        System.out.println("---------------------------------------------------------------");
    }
}
