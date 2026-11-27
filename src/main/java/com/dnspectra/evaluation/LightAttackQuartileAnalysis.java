
package com.dnspectra.evaluation;

import com.dnspectra.data.WekaDatasetBuilder;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.Attribute;

import java.util.Arrays;

public class LightAttackQuartileAnalysis {

    public static void main(String[] args) throws Exception {

        analyze("STATEFUL");
    }

    public static void analyze(String featureGroup)
            throws Exception {

        System.out.println("\n================================");
        System.out.println(
                featureGroup + " - FEATURE QUARTILE ANALYSIS"
        );
        System.out.println("================================");

        // Load the existing datasets
        Instances[] datasets =
                WekaDatasetBuilder.loadExperiment(featureGroup);

        // Use training data only
        Instances training = datasets[0];

        // Process every numeric feature
        for (int attributeIndex = 0;
             attributeIndex < training.numAttributes();
             attributeIndex++) {

            Attribute attribute =
                    training.attribute(attributeIndex);

            // Skip class label and non-numeric features
            if (attributeIndex == training.classIndex()
                    || !attribute.isNumeric()) {
                continue;
            }

            System.out.println("\nFEATURE: " + attribute.name());

            System.out.println(
                    "-------------------------------------------------------------"
            );

            System.out.printf(
                    "%-18s %12s %12s %12s%n",
                    "Class", "Q1", "Median", "Q3"
            );

            // Calculate quartiles separately for each class
            for (int classIndex = 0;
                 classIndex < training.numClasses();
                 classIndex++) {

                double[] values =
                        new double[training.numInstances()];

                int count = 0;

                // Collect values for this class
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

                // Keep only collected values
                values = Arrays.copyOf(values, count);

                if (count == 0) {
                    continue;
                }

                // Sort values
                Arrays.sort(values);

                // Calculate quartiles using linear interpolation
                double q1 = percentile(values, 25);
                double median = percentile(values, 50);
                double q3 = percentile(values, 75);

                System.out.printf(
                        "%-18s %12.4f %12.4f %12.4f%n",
                        training.classAttribute().value(classIndex),
                        q1,
                        median,
                        q3
                );
            }
        }

        System.out.println("\n================================");
        System.out.println("QUARTILE ANALYSIS COMPLETED");
        System.out.println("================================");
    }

    // Calculate percentile using linear interpolation
    private static double percentile(
            double[] sortedValues,
            double percentile) {

        if (sortedValues.length == 1) {
            return sortedValues[0];
        }

        double position =
                (percentile / 100.0)
                        * (sortedValues.length - 1);

        int lowerIndex = (int) Math.floor(position);
        int upperIndex = (int) Math.ceil(position);

        if (lowerIndex == upperIndex) {
            return sortedValues[lowerIndex];
        }

        double fraction = position - lowerIndex;

        return sortedValues[lowerIndex]
                + fraction * (
                sortedValues[upperIndex]
                        - sortedValues[lowerIndex]
        );
    }
}
