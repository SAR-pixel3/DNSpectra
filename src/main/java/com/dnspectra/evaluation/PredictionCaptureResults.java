package com.dnspectra.evaluation;

public class PredictionCaptureResults {

    public static void main(String[] args) {

        System.out.println("======================================");
        System.out.println("DNSPECTRA PREDICTION CAPTURE RESULTS");
        System.out.println("======================================");

        System.out.println("\nMODEL: Stateful Decision Tree");
        System.out.println("ACTUAL CLASS: LIGHT_ATTACK");

        System.out.println("\nCAPTURE 1: light_text");
        displayResults(921, 3, 36, 882);

        System.out.println("\nCAPTURE 2: light_video");
        displayResults(1245, 8, 8, 1229);

        System.out.println("\n======================================");
        System.out.println("OVERALL RESULTS");
        System.out.println("======================================");

        int total = 921 + 1245;
        int correctlyDetected = 36 + 8;

        System.out.println("Total records: " + total);
        System.out.println("Correct LIGHT_ATTACK predictions: "
                + correctlyDetected);

        System.out.printf(
                "Combined LIGHT_ATTACK detection rate: %.2f%%%n",
                correctlyDetected * 100.0 / total
        );
    }

    private static void displayResults(
            int total,
            int benign,
            int light,
            int heavy
    ) {

        System.out.println("Total records: " + total);
        System.out.println("BENIGN predictions: " + benign);
        System.out.println("LIGHT_ATTACK predictions: " + light);
        System.out.println("HEAVY_ATTACK predictions: " + heavy);

        System.out.printf(
                "LIGHT_ATTACK detection rate: %.2f%%%n",
                light * 100.0 / total
        );
    }
}