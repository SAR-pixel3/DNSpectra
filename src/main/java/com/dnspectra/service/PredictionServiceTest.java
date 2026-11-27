package com.dnspectra.service;

import java.util.List;

public class PredictionServiceTest {

    public static void main(String[] args) {

        String csvPath =
                "data/raw/heavy_attack/Attacks (1)/Attacks/stateful_features-heavy_video.pcap.csv";

        String featureGroup = "stateful";
        String modelName = "decision-tree";

        System.out.println("======================================");
        System.out.println("DNSPECTRA HEAVY ATTACK PREDICTION TEST");
        System.out.println("======================================");

        System.out.println("Model: Stateful Decision Tree");
        System.out.println("Input: " + csvPath);

        try {

            PredictionService service = new PredictionService();

            List<PredictionService.PredictionResult> results =
                    service.predictCsv(csvPath, featureGroup, modelName);

            int benign = 0;
            int lightAttack = 0;
            int heavyAttack = 0;

            for (PredictionService.PredictionResult result : results) {

                switch (result.getPredictedLabel()) {

                    case "BENIGN":
                        benign++;
                        break;

                    case "LIGHT_ATTACK":
                        lightAttack++;
                        break;

                    case "HEAVY_ATTACK":
                        heavyAttack++;
                        break;
                }
            }

            int total = results.size();

            if (total == 0) {
                System.out.println("No records found in the CSV file.");
                return;
            }

            System.out.println("\n======================================");
            System.out.println("PREDICTION SUMMARY");
            System.out.println("======================================");

            System.out.printf("BENIGN:       %d (%.2f%%)%n",
                    benign, benign * 100.0 / total);

            System.out.printf("LIGHT_ATTACK: %d (%.2f%%)%n",
                    lightAttack, lightAttack * 100.0 / total);

            System.out.printf("HEAVY_ATTACK: %d (%.2f%%)%n",
                    heavyAttack, heavyAttack * 100.0 / total);

            System.out.println("--------------------------------------");
            System.out.println("Total records: " + total);

            System.out.printf(
                    "HEAVY_ATTACK detection rate: %.2f%%%n",
                    heavyAttack * 100.0 / total
            );

            System.out.println("======================================");

        } catch (Exception e) {

            System.out.println("\nPREDICTION TEST FAILED");
            e.printStackTrace();
        }
    }
}