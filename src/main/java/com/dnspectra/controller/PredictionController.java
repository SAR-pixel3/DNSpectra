
package com.dnspectra.controller;

import com.dnspectra.service.PredictionService;
import com.dnspectra.service.PredictionService.PredictionResult;

import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

@RestController
@RequestMapping("/api")
public class PredictionController {

    private final PredictionService predictionService =
            new PredictionService();

    @PostMapping(
            value = "/predict",
            consumes = MediaType.MULTIPART_FORM_DATA_VALUE
    )
    public ResponseEntity<Map<String, Object>> predict(
            @RequestParam("file") MultipartFile file,
            @RequestParam("featureGroup") String featureGroup,
            @RequestParam("modelName") String modelName) {

        Path temporaryFile = null;

        try {
            // Validate uploaded file
            if (file == null || file.isEmpty()) {
                throw new IllegalArgumentException(
                        "Please upload a CSV file."
                );
            }

            String filename = file.getOriginalFilename();

            if (filename == null ||
                    !filename.toLowerCase(Locale.ROOT).endsWith(".csv")) {
                throw new IllegalArgumentException(
                        "Only CSV files are supported."
                );
            }

            // Validate feature group
            if (!featureGroup.equals("stateful") &&
                    !featureGroup.equals("stateless")) {
                throw new IllegalArgumentException(
                        "Invalid feature group selected."
                );
            }

            // Validate model
            List<String> validModels = Arrays.asList(
                    "decision-tree",
                    "random-forest",
                    "logistic-regression"
            );

            if (!validModels.contains(modelName)) {
                throw new IllegalArgumentException(
                        "Invalid machine learning model selected."
                );
            }

            // Save upload temporarily
            temporaryFile = Files.createTempFile(
                    "dnspectra-upload-",
                    ".csv"
            );

            file.transferTo(temporaryFile.toFile());

            // Run the existing prediction engine
            List<PredictionResult> results =
                    predictionService.predictCsv(
                            temporaryFile.toString(),
                            featureGroup,
                            modelName
                    );

            // Prepare prediction counts
            Map<String, Integer> counts = new LinkedHashMap<>();

            counts.put("BENIGN", 0);
            counts.put("LIGHT_ATTACK", 0);
            counts.put("HEAVY_ATTACK", 0);

            for (PredictionResult result : results) {

                String label = result.getPredictedLabel();

                counts.put(
                        label,
                        counts.getOrDefault(label, 0) + 1
                );
            }

            // Calculate percentages
            int totalRecords = results.size();

            Map<String, Double> percentages =
                    new LinkedHashMap<>();

            for (String label : counts.keySet()) {

                double percentage = totalRecords == 0
                        ? 0.0
                        : counts.get(label) * 100.0 / totalRecords;

                percentages.put(label, percentage);
            }

            // Prepare first 10 predictions as a preview
            List<Map<String, Object>> preview = new ArrayList<>();

            for (int i = 0; i < Math.min(10, results.size()); i++) {

                PredictionResult result = results.get(i);

                Map<String, Object> row = new LinkedHashMap<>();

                row.put("rowNumber", result.getRowNumber());
                row.put("predictedLabel", result.getPredictedLabel());
                row.put("confidence", result.getConfidence());
                row.put("probabilities", result.getProbabilities());

                preview.add(row);
            }

            // Build API response
            Map<String, Object> response = new LinkedHashMap<>();

            response.put("success", true);
            response.put("filename", filename);
            response.put("featureGroup", featureGroup);
            response.put("modelName", modelName);
            response.put("totalRecords", totalRecords);
            response.put("predictionCounts", counts);
            response.put("predictionPercentages", percentages);
            response.put("preview", preview);

            return ResponseEntity.ok(response);

        } catch (Exception e) {

            Map<String, Object> error = new LinkedHashMap<>();

            error.put("success", false);
            error.put(
                    "error",
                    e.getMessage() == null
                            ? "An unexpected error occurred."
                            : e.getMessage()
            );

            return ResponseEntity.badRequest().body(error);

        } finally {

            // Delete temporary upload after prediction
            if (temporaryFile != null) {
                try {
                    Files.deleteIfExists(temporaryFile);
                } catch (Exception ignored) {
                    // Temporary-file cleanup failed.
                }
            }
        }
    }
}
