
package com.dnspectra.evaluation;

import com.dnspectra.data.WekaDatasetBuilder;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class StatelessLightHeavyFeatureAUC {

    private static class FeatureScore {
        String name;
        double auc;
        double adjustedAuc;
        String direction;
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
        System.out.println("STATELESS LIGHT VS HEAVY FEATURE AUC");
        System.out.println("========================================");

        try {

            Instances[] datasets =
                    WekaDatasetBuilder.loadExperiment("STATELESS");

            if (datasets == null || datasets.length < 1
                    || datasets[0] == null) {
                System.out.println("Could not load training dataset.");
                return;
            }

            Instances training = datasets[0];

            int lightIndex =
                    training.classAttribute().indexOfValue("LIGHT_ATTACK");

            int heavyIndex =
                    training.classAttribute().indexOfValue("HEAVY_ATTACK");

            if (lightIndex < 0 || heavyIndex < 0) {
                System.out.println("Required classes were not found.");
                return;
            }

            List<FeatureScore> scores = new ArrayList<>();

            for (int a = 0; a < training.numAttributes(); a++) {

                if (a == training.classIndex()) {
                    continue;
                }

                Attribute attribute = training.attribute(a);

                if (!attribute.isNumeric()) {
                    continue;
                }

                List<ValueLabel> values = new ArrayList<>();

                int lightCount = 0;
                int heavyCount = 0;

                for (int i = 0; i < training.numInstances(); i++) {

                    Instance instance = training.instance(i);

                    if (instance.isMissing(a)) {
                        continue;
                    }

                    int classValue = (int) instance.classValue();

                    if (classValue == lightIndex) {

                        values.add(
                                new ValueLabel(instance.value(a), true)
                        );

                        lightCount++;

                    } else if (classValue == heavyIndex) {

                        values.add(
                                new ValueLabel(instance.value(a), false)
                        );

                        heavyCount++;
                    }
                }

                if (lightCount == 0 || heavyCount == 0) {
                    continue;
                }

                double auc = calculateAuc(values);

                FeatureScore score = new FeatureScore();

                score.name = attribute.name();
                score.auc = auc;
                score.adjustedAuc = Math.max(auc, 1.0 - auc);

                score.direction =
                        auc >= 0.5
                                ? "LIGHT tends higher"
                                : "LIGHT tends lower";

                scores.add(score);
            }

            scores.sort(
                    Comparator.comparingDouble(
                            (FeatureScore s) -> s.adjustedAuc
                    ).reversed()
            );

            System.out.println("\nTRAINING RECORDS");
            System.out.println("------------------------------");

            System.out.println(
                    "LIGHT_ATTACK: " + countClass(training, lightIndex)
            );

            System.out.println(
                    "HEAVY_ATTACK: " + countClass(training, heavyIndex)
            );

            System.out.println("\nFEATURE AUC RESULTS");
            System.out.println(
                    "---------------------------------------------------------------"
            );

            System.out.printf(
                    "%-23s %12s %15s%n",
                    "Feature",
                    "Raw AUC",
                    "Adjusted AUC"
            );

            System.out.println(
                    "---------------------------------------------------------------"
            );

            for (FeatureScore score : scores) {

                System.out.printf(
                        "%-23s %12.4f %15.4f%n",
                        score.name,
                        score.auc,
                        score.adjustedAuc
                );

                System.out.println(
                        "   " + score.direction
                );
            }

            System.out.println(
                    "\nInterpretation: adjusted AUC closer to 0.5 "
                            + "means weaker individual-feature separation."
            );

        } catch (Exception e) {

            System.out.println("Analysis failed:");
            e.printStackTrace();
        }
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

    private static double calculateAuc(
            List<ValueLabel> values) {

        Collections.sort(
                values,
                Comparator.comparingDouble(v -> v.value)
        );

        int lightCount = 0;
        int heavyCount = 0;

        for (ValueLabel value : values) {

            if (value.isLight) {
                lightCount++;
            } else {
                heavyCount++;
            }
        }

        double rankSumLight = 0.0;

        int i = 0;

        while (i < values.size()) {

            int start = i;

            double currentValue = values.get(i).value;

            while (i < values.size()
                    && Double.compare(
                    values.get(i).value,
                    currentValue
            ) == 0) {
                i++;
            }

            int end = i;

            double averageRank =
                    ((start + 1) + end) / 2.0;

            for (int j = start; j < end; j++) {

                if (values.get(j).isLight) {
                    rankSumLight += averageRank;
                }
            }
        }

        double uStatistic =
                rankSumLight
                        - (lightCount * (lightCount + 1)) / 2.0;

        return uStatistic / ((double) lightCount * heavyCount);
    }
}
