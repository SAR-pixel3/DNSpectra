
package com.dnspectra.service;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;

import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class PredictionService {

    private static final List<String> CLASS_LABELS = Arrays.asList(
            "BENIGN", "LIGHT_ATTACK", "HEAVY_ATTACK"
    );

    private static final String[] STATEFUL_FEATURES = {
            "rr", "A_frequency", "NS_frequency", "CNAME_frequency",
            "SOA_frequency", "NULL_frequency", "PTR_frequency",
            "HINFO_frequency", "MX_frequency", "TXT_frequency",
            "AAAA_frequency", "SRV_frequency", "OPT_frequency",
            "rr_count", "rr_name_entropy", "rr_name_length",
            "distinct_ns", "a_records", "ttl_mean", "ttl_variance"
    };

    private static final String[] STATELESS_FEATURES = {
            "FQDN_count", "subdomain_length", "upper", "lower",
            "numeric", "entropy", "special", "labels",
            "labels_max", "labels_average", "len", "subdomain"
    };

    private static final String[] RR_TYPE_FEATURES = {
            "rr_type_A", "rr_type_AAAA", "rr_type_PTR",
            "rr_type_TXT", "rr_type_empty", "rr_type_missing"
    };

    public List<PredictionResult> predictCsv(
            String csvPath,
            String featureGroup,
            String modelName
    ) throws Exception {

        String group = featureGroup.trim().toLowerCase();
        String model = modelName.trim().toLowerCase();

        if (!group.equals("stateful") && !group.equals("stateless")) {
            throw new IllegalArgumentException(
                    "Feature group must be 'stateful' or 'stateless'."
            );
        }

        if (!model.equals("decision-tree")
                && !model.equals("random-forest")
                && !model.equals("logistic-regression")) {
            throw new IllegalArgumentException(
                    "Unsupported model name: " + modelName
            );
        }

        String modelPath = "models/" + model + "-" + group + ".model";

        if (!Files.exists(Paths.get(modelPath))) {
            throw new IllegalArgumentException(
                    "Model file not found: " + modelPath
            );
        }

        if (!Files.exists(Paths.get(csvPath))) {
            throw new IllegalArgumentException(
                    "CSV file not found: " + csvPath
            );
        }

        Classifier classifier = (Classifier)
                SerializationHelper.read(modelPath);

        Instances structure = createDatasetStructure(group);

        List<PredictionResult> results = new ArrayList<>();

        try (Reader reader = Files.newBufferedReader(
                Paths.get(csvPath), StandardCharsets.UTF_8)) {

            CSVFormat format = CSVFormat.DEFAULT.builder()
                    .setHeader()
                    .setSkipHeaderRecord(true)
                    .build();

            try (CSVParser parser = new CSVParser(reader, format)) {

                Map<String, Integer> headers = parser.getHeaderMap();

                if (headers == null) {
                    throw new IllegalArgumentException(
                            "CSV file does not contain a header."
                    );
                }

                validateHeaders(headers, group);

                int rowNumber = 1;

                for (CSVRecord record : parser) {

                    rowNumber++;

                    double[] values = new double[structure.numAttributes()];

                    if (group.equals("stateful")) {

                        for (int i = 0; i < STATEFUL_FEATURES.length; i++) {
                            values[i] = readNumber(
                                    record, STATEFUL_FEATURES[i], rowNumber
                            );
                        }

                        values[20] = Double.NaN;

                        double[] encoded = encodeRRType(
                                record.get("rr_type")
                        );

                        for (int i = 0; i < encoded.length; i++) {
                            values[21 + i] = encoded[i];
                        }

                    } else {

                        for (int i = 0; i < STATELESS_FEATURES.length; i++) {
                            values[i] = readNumber(
                                    record, STATELESS_FEATURES[i], rowNumber
                            );
                        }

                        values[12] = Double.NaN;
                    }

                    // Create the new Weka instance.
                    DenseInstance newInstance =
                            new DenseInstance(1.0, values);

                    // Add it to the dataset.
                    // Weka associates the stored copy with the dataset.
                    structure.add(newInstance);

                    // Retrieve the dataset-associated instance.
                    Instance predictionInstance =
                            structure.instance(structure.numInstances() - 1);

                    // Confirm that the instance has a dataset.
                    if (predictionInstance.dataset() == null) {
                        throw new IllegalStateException(
                                "Weka dataset was not assigned at row "
                                        + rowNumber
                        );
                    }

                    // Generate the prediction.
                    double[] probabilities =
                            classifier.distributionForInstance(
                                    predictionInstance
                            );

                    int predictedIndex = Utils.maxIndex(probabilities);

                    String predictedLabel =
                            structure.classAttribute().value(predictedIndex);

                    double confidence = probabilities[predictedIndex];

                    Map<String, Double> probabilityMap =
                            new LinkedHashMap<>();

                    for (int i = 0; i < CLASS_LABELS.size(); i++) {
                        probabilityMap.put(
                                CLASS_LABELS.get(i),
                                probabilities[i]
                        );
                    }

                    results.add(new PredictionResult(
                            rowNumber,
                            predictedLabel,
                            confidence,
                            probabilityMap
                    ));
                }
            }
        }

        return results;
    }

    private Instances createDatasetStructure(String group) {

        ArrayList<Attribute> attributes = new ArrayList<>();

        if (group.equals("stateful")) {

            for (String feature : STATEFUL_FEATURES) {
                attributes.add(new Attribute(feature));
            }

            attributes.add(new Attribute(
                    "label", new ArrayList<>(CLASS_LABELS)
            ));

            for (String feature : RR_TYPE_FEATURES) {
                attributes.add(new Attribute(feature));
            }

        } else {

            for (String feature : STATELESS_FEATURES) {
                attributes.add(new Attribute(feature));
            }

            attributes.add(new Attribute(
                    "label", new ArrayList<>(CLASS_LABELS)
            ));
        }

        Instances dataset = new Instances(
                "DNSpectraPrediction", attributes, 0
        );

        dataset.setClassIndex(dataset.attribute("label").index());

        return dataset;
    }

    private void validateHeaders(
            Map<String, Integer> headers,
            String group
    ) {

        List<String> requiredColumns = new ArrayList<>();

        if (group.equals("stateful")) {
            requiredColumns.addAll(Arrays.asList(STATEFUL_FEATURES));
            requiredColumns.add("rr_type");
        } else {
            requiredColumns.addAll(Arrays.asList(STATELESS_FEATURES));
        }

        for (String column : requiredColumns) {
            if (!headers.containsKey(column)) {
                throw new IllegalArgumentException(
                        "Required CSV column is missing: " + column
                );
            }
        }
    }

    private double readNumber(
            CSVRecord record,
            String column,
            int rowNumber
    ) {

        String value = record.get(column).trim();

        if (value.isEmpty()) {
            throw new IllegalArgumentException(
                    "Empty value in column '" + column
                            + "' at CSV row " + rowNumber
            );
        }

        try {
            double number = Double.parseDouble(value);

            if (!Double.isFinite(number)) {
                throw new NumberFormatException();
            }

            return number;

        } catch (NumberFormatException exception) {
            throw new IllegalArgumentException(
                    "Invalid number '" + value
                            + "' in column '" + column
                            + "' at CSV row " + rowNumber
            );
        }
    }

    private double[] encodeRRType(String rrType) {

        double[] encoded = new double[6];

        if (rrType == null || rrType.trim().isEmpty()
                || rrType.trim().equals("{None}")) {
            encoded[5] = 1.0;
            return encoded;
        }

        String value = rrType.trim();

        if (value.equals("set()")) {
            encoded[4] = 1.0;
            return encoded;
        }

        if (value.contains("'A'")) encoded[0] = 1.0;
        if (value.contains("'AAAA'")) encoded[1] = 1.0;
        if (value.contains("'PTR'")) encoded[2] = 1.0;
        if (value.contains("'TXT'")) encoded[3] = 1.0;

        return encoded;
    }

    public static class PredictionResult {

        private final int rowNumber;
        private final String predictedLabel;
        private final double confidence;
        private final Map<String, Double> probabilities;

        public PredictionResult(
                int rowNumber,
                String predictedLabel,
                double confidence,
                Map<String, Double> probabilities
        ) {
            this.rowNumber = rowNumber;
            this.predictedLabel = predictedLabel;
            this.confidence = confidence;
            this.probabilities = probabilities;
        }

        public int getRowNumber() {
            return rowNumber;
        }

        public String getPredictedLabel() {
            return predictedLabel;
        }

        public double getConfidence() {
            return confidence;
        }

        public Map<String, Double> getProbabilities() {
            return probabilities;
        }
    }
}
