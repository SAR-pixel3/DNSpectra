
package com.dnspectra.evaluation;

import com.dnspectra.data.WekaDatasetBuilder;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.Attribute;

public class LightAttackFeatureDistribution {

    public static void main(String[] args) throws Exception {

        analyze("STATEFUL");
    }

    public static void analyze(String featureGroup)
            throws Exception {

        System.out.println("\n================================");
        System.out.println(
                featureGroup + " - TRAINING FEATURE DISTRIBUTION"
        );
        System.out.println("================================");

        // Load the existing training and testing datasets
        Instances[] datasets =
                WekaDatasetBuilder.loadExperiment(featureGroup);

        // Use training data ONLY for this investigation
        Instances training = datasets[0];

        int classIndex =
                training.classAttribute()
                        .indexOfValue("LIGHT_ATTACK");

        if (classIndex < 0) {
            throw new IllegalStateException(
                    "LIGHT_ATTACK class not found."
            );
        }

        // Count records belonging to each class
        int[] classCounts = new int[training.numClasses()];

        for (int i = 0; i < training.numInstances(); i++) {

            Instance instance = training.instance(i);

            int label = (int) instance.classValue();

            classCounts[label]++;
        }

        System.out.println("\nTRAINING CLASS COUNTS");

        for (int i = 0; i < classCounts.length; i++) {

            System.out.println(
                    training.classAttribute().value(i)
                            + ": " + classCounts[i]
            );
        }

        System.out.println(
                "\nFEATURE MEANS BY CLASS (TRAINING DATA)"
        );

        System.out.println(
                "-------------------------------------------------------------"
        );

        System.out.printf(
                "%-24s %15s %15s %15s%n",
                "Feature",
                "BENIGN",
                "LIGHT_ATTACK",
                "HEAVY_ATTACK"
        );

        System.out.println(
                "-------------------------------------------------------------"
        );

        // Calculate the mean of every numeric feature
        for (int attributeIndex = 0;
             attributeIndex < training.numAttributes();
             attributeIndex++) {

            Attribute attribute =
                    training.attribute(attributeIndex);

            // Skip class attribute
            if (attributeIndex == training.classIndex()) {
                continue;
            }

            // Skip non-numeric attributes
            if (!attribute.isNumeric()) {
                continue;
            }

            double[] sums =
                    new double[training.numClasses()];

            int[] counts =
                    new int[training.numClasses()];

            for (int i = 0; i < training.numInstances(); i++) {

                Instance instance = training.instance(i);

                int label = (int) instance.classValue();

                double value =
                        instance.value(attributeIndex);

                if (!Double.isNaN(value)
                        && !Double.isInfinite(value)) {

                    sums[label] += value;
                    counts[label]++;
                }
            }

            System.out.printf(
                    "%-24s",
                    attribute.name()
            );

            for (int classNumber = 0;
                 classNumber < training.numClasses();
                 classNumber++) {

                double mean = counts[classNumber] > 0
                        ? sums[classNumber] / counts[classNumber]
                        : 0.0;

                System.out.printf(
                        " %15.4f",
                        mean
                );
            }

            System.out.println();
        }

        System.out.println(
                "-------------------------------------------------------------"
        );

        System.out.println(
                "\nAnalysis completed using training data only."
        );
    }
}
