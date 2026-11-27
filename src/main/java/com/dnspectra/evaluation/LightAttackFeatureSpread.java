
package com.dnspectra.evaluation;

import com.dnspectra.data.WekaDatasetBuilder;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.Attribute;

import java.util.Arrays;

public class LightAttackFeatureSpread {

    public static void main(String[] args) throws Exception {

        analyze("STATEFUL");
    }

    public static void analyze(String featureGroup)
            throws Exception {

        System.out.println("\n================================");
        System.out.println(
                featureGroup + " - FEATURE SPREAD ANALYSIS"
        );
        System.out.println("================================");

        // Load the datasets
        Instances[] datasets =
                WekaDatasetBuilder.loadExperiment(featureGroup);

        // Use training data only
        Instances training = datasets[0];

        int numberOfClasses = training.numClasses();

        // Process every numeric feature
        for (int attributeIndex = 0;
             attributeIndex < training.numAttributes();
             attributeIndex++) {

            Attribute attribute =
                    training.attribute(attributeIndex);

            // Skip the class label and non-numeric features
            if (attributeIndex == training.classIndex()
                    || !attribute.isNumeric()) {
                continue;
            }

            System.out.println("\nFEATURE: " + attribute.name());

            System.out.println("---------------------------------------------");

            System.out.printf(
                    "%-18s %14s %14s%n",
                    "Class", "Median", "Std Deviation"
            );

            for (int classIndex = 0;
                 classIndex < numberOfClasses;
                 classIndex++) {

                double[] values =
                        new double[training.numInstances()];

                int count = 0;

                // Collect feature values for this class
                for (int i = 0;
                     i < training.numInstances();
                     i++) {

                    Instance instance = training.instance(i);

                    if ((int) instance.classValue() == classIndex) {

                        double value =
                                instance.value(attributeIndex);

                        if (!Double.isNaN(value)
                                && !Double.isInfinite(value)) {

                            values[count++] = value;
                        }
                    }
                }

                // Copy only collected values
                values = Arrays.copyOf(values, count);

                if (count == 0) {
                    continue;
                }

                // Sort values to calculate the median
                Arrays.sort(values);

                double median;

                if (count % 2 == 0) {

                    median =
                            (values[count / 2 - 1]
                                    + values[count / 2]) / 2.0;

                } else {

                    median = values[count / 2];
                }

                // Calculate mean
                double sum = 0;

                for (double value : values) {
                    sum += value;
                }

                double mean = sum / count;

                // Calculate population standard deviation
                double squaredDifferenceSum = 0;

                for (double value : values) {

                    double difference = value - mean;

                    squaredDifferenceSum +=
                            difference * difference;
                }

                double standardDeviation =
                        Math.sqrt(squaredDifferenceSum / count);

                System.out.printf(
                        "%-18s %14.4f %14.4f%n",
                        training.classAttribute().value(classIndex),
                        median,
                        standardDeviation
                );
            }
        }

        System.out.println("\n================================");
        System.out.println("FEATURE SPREAD ANALYSIS COMPLETED");
        System.out.println("================================");
    }
}
