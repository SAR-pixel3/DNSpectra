
package com.dnspectra.evaluation;

import com.dnspectra.data.WekaDatasetBuilder;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.Attribute;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class LightHeavyFeatureAUC {

    // Stores a feature value and its class
    private static class FeatureValue {
        double value;
        boolean isLight;

        FeatureValue(double value, boolean isLight) {
            this.value = value;
            this.isLight = isLight;
        }
    }

    // Stores the final AUC result
    private static class AUCResult {
        String feature;
        double auc;
        String direction;

        AUCResult(String feature, double auc, String direction) {
            this.feature = feature;
            this.auc = auc;
            this.direction = direction;
        }
    }

    public static void main(String[] args) throws Exception {

        analyze("STATEFUL");
    }

    public static void analyze(String featureGroup)
            throws Exception {

        System.out.println("\n================================");
        System.out.println(
                featureGroup + " - LIGHT VS HEAVY FEATURE AUC"
        );
        System.out.println("================================");

        // Load the existing capture-level split
        Instances[] datasets =
                WekaDatasetBuilder.loadExperiment(featureGroup);

        // Use training data only
        Instances training = datasets[0];

        // Find class indexes by their actual names
        int lightIndex =
                training.classAttribute()
                        .indexOfValue("LIGHT_ATTACK");

        int heavyIndex =
                training.classAttribute()
                        .indexOfValue("HEAVY_ATTACK");

        if (lightIndex < 0 || heavyIndex < 0) {
            throw new IllegalStateException(
                    "Required attack classes were not found."
            );
        }

        List<AUCResult> results = new ArrayList<>();

        // Examine every numeric feature
        for (int attributeIndex = 0;
             attributeIndex < training.numAttributes();
             attributeIndex++) {

            Attribute attribute =
                    training.attribute(attributeIndex);

            // Skip class label and non-numeric attributes
            if (attributeIndex == training.classIndex()
                    || !attribute.isNumeric()) {
                continue;
            }

            List<FeatureValue> values = new ArrayList<>();

            int lightCount = 0;
            int heavyCount = 0;

            // Collect only LIGHT and HEAVY training records
            for (int i = 0;
                 i < training.numInstances();
                 i++) {

                Instance instance = training.instance(i);

                int label = (int) instance.classValue();

                if (label != lightIndex && label != heavyIndex) {
                    continue;
                }

                double value = instance.value(attributeIndex);

                if (Double.isNaN(value)
                        || Double.isInfinite(value)) {
                    continue;
                }

                boolean isLight = (label == lightIndex);

                values.add(new FeatureValue(value, isLight));

                if (isLight) {
                    lightCount++;
                } else {
                    heavyCount++;
                }
            }

            if (lightCount == 0 || heavyCount == 0) {
                continue;
            }

            // Calculate AUC using average ranks for tied values
            double rawAUC = calculateAUC(
                    values,
                    lightCount,
                    heavyCount
            );

            // Choose the better direction
            // This measures the strongest single-feature separation
            double auc;
            String direction;

            if (rawAUC >= 0.5) {
                auc = rawAUC;
                direction = "LIGHT tends higher";
            } else {
                auc = 1.0 - rawAUC;
                direction = "LIGHT tends lower";
            }

            results.add(
                    new AUCResult(
                            attribute.name(),
                            auc,
                            direction
                    )
            );
        }

        // Sort features from highest AUC to lowest
        results.sort(
                Comparator.comparingDouble(
                        (AUCResult result) -> result.auc
                ).reversed()
        );

        // Display results
        System.out.println("\nLIGHT_ATTACK vs HEAVY_ATTACK");
        System.out.println("Training records only.");
        System.out.println("-----------------------------------------------");

        System.out.printf(
                "%-5s %-24s %-12s %s%n",
                "Rank", "Feature", "AUC", "Direction"
        );

        System.out.println("-----------------------------------------------");

        int rank = 1;

        for (AUCResult result : results) {

            System.out.printf(
                    "%-5d %-24s %-12.4f %s%n",
                    rank++,
                    result.feature,
                    result.auc,
                    result.direction
            );
        }

        System.out.println("-----------------------------------------------");

        System.out.println("\nAUC INTERPRETATION");
        System.out.println("0.50 = No separation");
        System.out.println("0.60-0.69 = Weak separation");
        System.out.println("0.70-0.79 = Moderate separation");
        System.out.println("0.80-0.89 = Good separation");
        System.out.println("0.90-1.00 = Excellent separation");

        System.out.println("\nAnalysis completed using training data only.");
    }

    // Calculate binary ROC-AUC using the Mann-Whitney rank method
    private static double calculateAUC(
            List<FeatureValue> values,
            int lightCount,
            int heavyCount) {

        // Sort feature values from smallest to largest
        Collections.sort(
                values,
                Comparator.comparingDouble(item -> item.value)
        );

        double lightRankSum = 0.0;

        int position = 0;

        while (position < values.size()) {

            int end = position + 1;

            // Find all records with the same value
            while (end < values.size()
                    && Double.compare(
                    values.get(position).value,
                    values.get(end).value
            ) == 0) {

                end++;
            }

            // Average rank for tied observations
            double averageRank =
                    ((position + 1) + end) / 2.0;

            // Add ranks belonging to LIGHT_ATTACK
            for (int i = position; i < end; i++) {

                if (values.get(i).isLight) {
                    lightRankSum += averageRank;
                }
            }

            position = end;
        }

        // Mann-Whitney U statistic converted to AUC
        double minimumRankSum =
                lightCount * (lightCount + 1) / 2.0;

        double uStatistic =
                lightRankSum - minimumRankSum;

        return uStatistic / ((double) lightCount * heavyCount);
    }
}
