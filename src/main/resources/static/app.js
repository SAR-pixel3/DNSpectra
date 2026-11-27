
// =====================================================
// DNSPECTRA - MAIN DASHBOARD JAVASCRIPT
// Prediction + Model Comparison + Confusion Matrices
// =====================================================

const analyseButton = document.getElementById("analyseButton");
const csvFileInput = document.getElementById("csvFile");
const statusMessage = document.getElementById("status");

// =====================================================
// 1. HELPER FUNCTION
// =====================================================

function createTextElement(tag, text, styles = "") {
    const element = document.createElement(tag);
    element.textContent = text;
    element.style.cssText = styles;
    return element;
}

// =====================================================
// 2. PREDICTION RESULTS SECTION
// =====================================================

const resultsPanel = document.createElement("section");

resultsPanel.id = "resultsPanel";

resultsPanel.style.cssText = `
    display: none;
    margin-top: 25px;
    padding: 25px;
    border-radius: 15px;
    background: rgba(17, 25, 45, 0.92);
    border: 1px solid rgba(34, 211, 238, 0.25);
    color: #f1f5f9;
`;

resultsPanel.innerHTML = `
    <h2 style="margin-bottom: 8px; font-size: 20px;">
        ◈ Analysis Results
    </h2>

    <p id="resultDetails"
       style="color:#94a3b8; font-size:12px; margin-bottom:20px;">
    </p>

    <div id="resultCards"
         style="display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px;">
    </div>

    <h3 style="margin:25px 0 12px; font-size:15px;">
        Prediction Preview
    </h3>

    <div style="overflow-x:auto;">
        <table id="previewTable"
               style="width:100%; border-collapse:collapse; font-size:12px;">
            <thead>
                <tr style="color:#22d3ee; text-align:left;">
                    <th style="padding:10px;">Row</th>
                    <th style="padding:10px;">Prediction</th>
                    <th style="padding:10px;">Confidence</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
    </div>
`;

document.querySelector("main").appendChild(resultsPanel);

// =====================================================
// 3. MODEL PERFORMANCE DATA
// Actual experimental results
// =====================================================

const modelPerformance = [
    {
        featureGroup: "Stateful",
        model: "Decision Tree",
        accuracy: 80.83,
        macroF1: 56.13
    },
    {
        featureGroup: "Stateful",
        model: "Random Forest",
        accuracy: 80.75,
        macroF1: 56.07
    },
    {
        featureGroup: "Stateful",
        model: "Logistic Regression",
        accuracy: 81.89,
        macroF1: 55.51
    },
    {
        featureGroup: "Stateless",
        model: "Decision Tree",
        accuracy: 79.60,
        macroF1: 53.36
    },
    {
        featureGroup: "Stateless",
        model: "Random Forest",
        accuracy: 79.62,
        macroF1: 53.38
    },
    {
        featureGroup: "Stateless",
        model: "Logistic Regression",
        accuracy: 79.23,
        macroF1: 53.12
    }
];

// =====================================================
// 4. MODEL COMPARISON SECTION
// =====================================================

const comparisonPanel = document.createElement("section");

comparisonPanel.id = "comparisonPanel";

comparisonPanel.style.cssText = `
    margin-top: 25px;
    padding: 25px;
    border-radius: 15px;
    background: rgba(17, 25, 45, 0.92);
    border: 1px solid rgba(167, 139, 250, 0.25);
    color: #f1f5f9;
`;

comparisonPanel.innerHTML = `
    <h2 style="font-size:20px; margin-bottom:8px;">
        ◈ Model Performance Comparison
    </h2>

    <p style="font-size:12px; color:#94a3b8; margin-bottom:25px;">
        Comparison of the six trained machine-learning configurations
        using the held-out experimental test dataset.
    </p>

    <div id="performanceCharts"
         style="display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:25px;">
    </div>

    <h3 style="font-size:15px; margin:30px 0 15px;">
        Detailed Performance Results
    </h3>

    <div style="overflow-x:auto;">
        <table style="width:100%; border-collapse:collapse; font-size:12px;">
            <thead>
                <tr style="color:#22d3ee; text-align:left;">
                    <th style="padding:12px;">Feature Group</th>
                    <th style="padding:12px;">Model</th>
                    <th style="padding:12px;">Accuracy</th>
                    <th style="padding:12px;">Macro F1</th>
                </tr>
            </thead>
            <tbody id="performanceTableBody"></tbody>
        </table>
    </div>

    <div style="
        margin-top:22px;
        padding:15px;
        border-radius:10px;
        background:rgba(34,211,238,0.05);
        border:1px solid rgba(34,211,238,0.15);
    ">
        <p style="font-size:12px; color:#cbd5e1; line-height:1.8;">
            <strong style="color:#22d3ee;">Research observation:</strong>
            Stateful features achieved higher macro F1-scores
            across all three algorithms. Logistic Regression achieved
            the highest accuracy (81.89%), while Decision Tree
            achieved the highest macro F1-score (56.13%).
            All three models experienced difficulty detecting
            LIGHT_ATTACK records.
        </p>
    </div>

    <p style="font-size:10px; color:#64748b; margin-top:15px;">
        Note: These are experimental evaluation results, not the
        predictions from an individual uploaded CSV file.
    </p>
`;

document.querySelector("main").appendChild(comparisonPanel);

// =====================================================
// 5. DRAW PERFORMANCE BAR CHARTS
// =====================================================

function createPerformanceCharts() {

    const chartContainer =
        document.getElementById("performanceCharts");

    chartContainer.replaceChildren();

    const metrics = [
        {
            name: "Accuracy",
            property: "accuracy",
            color: "#22d3ee"
        },
        {
            name: "Macro F1-Score",
            property: "macroF1",
            color: "#a78bfa"
        }
    ];

    metrics.forEach(metric => {

        const chartCard = document.createElement("div");

        chartCard.style.cssText = `
            padding:20px;
            border-radius:12px;
            background:rgba(5,10,23,0.75);
            border:1px solid rgba(148,163,184,0.12);
        `;

        chartCard.appendChild(
            createTextElement(
                "h3",
                metric.name,
                "font-size:15px; margin-bottom:20px;"
            )
        );

        modelPerformance.forEach(result => {

            const item = document.createElement("div");
            item.style.marginBottom = "17px";

            const label = document.createElement("div");

            label.style.cssText = `
                display:flex;
                justify-content:space-between;
                gap:10px;
                margin-bottom:7px;
                font-size:11px;
                color:#cbd5e1;
            `;

            const modelLabel = document.createElement("span");

            modelLabel.textContent =
                `${result.featureGroup} - ${result.model}`;

            const valueLabel = document.createElement("strong");

            const value = result[metric.property];

            valueLabel.textContent = `${value.toFixed(2)}%`;
            valueLabel.style.color = metric.color;

            label.appendChild(modelLabel);
            label.appendChild(valueLabel);

            const track = document.createElement("div");

            track.style.cssText = `
                width:100%;
                height:9px;
                border-radius:20px;
                background:rgba(148,163,184,0.12);
                overflow:hidden;
            `;

            const bar = document.createElement("div");

            bar.style.cssText = `
                width:${value}%;
                height:100%;
                border-radius:20px;
                background:${metric.color};
                transition:width 0.5s ease;
            `;

            track.appendChild(bar);
            item.appendChild(label);
            item.appendChild(track);
            chartCard.appendChild(item);
        });

        chartContainer.appendChild(chartCard);
    });
}

// =====================================================
// 6. DISPLAY PERFORMANCE TABLE
// =====================================================

function displayPerformanceTable() {

    const tableBody =
        document.getElementById("performanceTableBody");

    tableBody.replaceChildren();

    modelPerformance.forEach(result => {

        const row = document.createElement("tr");

        row.style.borderTop =
            "1px solid rgba(148,163,184,0.12)";

        const values = [
            result.featureGroup,
            result.model,
            `${result.accuracy.toFixed(2)}%`,
            `${result.macroF1.toFixed(2)}%`
        ];

        values.forEach(value => {

            const cell = document.createElement("td");

            cell.textContent = value;
            cell.style.padding = "13px 12px";

            row.appendChild(cell);
        });

        tableBody.appendChild(row);
    });
}

// =====================================================
// 7. CONFUSION MATRIX DATA
// Actual Weka evaluation results
//
// Rows = actual classes
// Columns = predicted classes
//
// Class order:
// BENIGN, LIGHT_ATTACK, HEAVY_ATTACK
// =====================================================

const confusionMatrices = {

    stateful: {

        "decision-tree": [
            [24904, 89, 9567],
            [11, 44, 2111],
            [92, 883, 28838]
        ],

        "random-forest": [
            [24859, 99, 9602],
            [15, 44, 2107],
            [98, 887, 28828]
        ],

        "logistic-regression": [
            [25007, 0, 9553],
            [18, 0, 2148],
            [327, 1, 29485]
        ]
    },

    stateless: {

        "decision-tree": [
            [54562, 0, 34012],
            [2, 0, 7848],
            [67, 0, 109047]
        ],

        "random-forest": [
            [54609, 4, 33961],
            [4, 0, 7846],
            [66, 6, 109042]
        ],

        "logistic-regression": [
            [54817, 0, 33757],
            [64, 0, 7786],
            [1092, 0, 108022]
        ]
    }
};

// =====================================================
// 8. CONFUSION MATRIX VISUALISATION SECTION
// =====================================================

const confusionPanel = document.createElement("section");

confusionPanel.id = "confusionPanel";

confusionPanel.style.cssText = `
    margin-top:25px;
    padding:25px;
    border-radius:15px;
    background:rgba(17,25,45,0.92);
    border:1px solid rgba(34,211,238,0.25);
    color:#f1f5f9;
`;

confusionPanel.innerHTML = `
    <h2 style="font-size:20px; margin-bottom:8px;">
        ◈ Confusion Matrix Analysis
    </h2>

    <p style="font-size:12px; color:#94a3b8; line-height:1.8; margin-bottom:22px;">
        Examine how accurately each model classifies BENIGN,
        LIGHT_ATTACK and HEAVY_ATTACK DNS records.
        Select a feature group and machine-learning model below.
    </p>

    <div style="
        display:grid;
        grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
        gap:15px;
        margin-bottom:25px;
    ">
        <div>
            <label for="matrixFeatureGroup"
                   style="display:block; font-size:12px; color:#94a3b8; margin-bottom:8px;">
                Feature Group
            </label>

            <select id="matrixFeatureGroup"
                    style="
                        width:100%;
                        padding:12px;
                        border-radius:8px;
                        background:#0b1224;
                        color:#f1f5f9;
                        border:1px solid rgba(34,211,238,0.4);
                    ">
                <option value="stateful">Stateful Features</option>
                <option value="stateless">Stateless Features</option>
            </select>
        </div>

        <div>
            <label for="matrixModel"
                   style="display:block; font-size:12px; color:#94a3b8; margin-bottom:8px;">
                Machine Learning Model
            </label>

            <select id="matrixModel"
                    style="
                        width:100%;
                        padding:12px;
                        border-radius:8px;
                        background:#0b1224;
                        color:#f1f5f9;
                        border:1px solid rgba(167,139,250,0.4);
                    ">
                <option value="decision-tree">Decision Tree</option>
                <option value="random-forest">Random Forest</option>
                <option value="logistic-regression">Logistic Regression</option>
            </select>
        </div>
    </div>

    <div id="matrixDetails"
         style="font-size:12px; color:#22d3ee; margin-bottom:18px;">
    </div>

    <div style="overflow-x:auto;">
        <div id="confusionMatrixDisplay"></div>
    </div>

    <div style="
        display:flex;
        flex-wrap:wrap;
        gap:15px;
        margin-top:20px;
        font-size:11px;
        color:#94a3b8;
    ">
        <span>■ Darker cells = larger counts</span>
        <span>■ Diagonal cells = correct predictions</span>
        <span>■ Other cells = incorrect predictions</span>
    </div>

    <div style="
        margin-top:22px;
        padding:15px;
        border-radius:10px;
        background:rgba(167,139,250,0.06);
        border:1px solid rgba(167,139,250,0.2);
    ">
        <p id="matrixObservation"
           style="font-size:12px; color:#cbd5e1; line-height:1.8;">
        </p>
    </div>

    <p style="font-size:10px; color:#64748b; margin-top:15px;">
        Source: DNSpectra held-out test-set evaluation using Weka.
        Values represent record counts, not percentages.
    </p>
`;

document.querySelector("main").appendChild(confusionPanel);

// =====================================================
// 9. DRAW SELECTED CONFUSION MATRIX
// =====================================================

function displayConfusionMatrix() {

    const featureGroup =
        document.getElementById("matrixFeatureGroup").value;

    const model =
        document.getElementById("matrixModel").value;

    const matrix = confusionMatrices[featureGroup][model];

    const classNames = [
        "BENIGN",
        "LIGHT_ATTACK",
        "HEAVY_ATTACK"
    ];

    const display = document.getElementById(
        "confusionMatrixDisplay"
    );

    const total = matrix.flat().reduce(
        (sum, value) => sum + value,
        0
    );

    const maxValue = Math.max(...matrix.flat());

    const modelLabels = {
        "decision-tree": "Decision Tree",
        "random-forest": "Random Forest",
        "logistic-regression": "Logistic Regression"
    };

    document.getElementById("matrixDetails").textContent =
        `${featureGroup.toUpperCase()} FEATURES | ` +
        `${modelLabels[model]} | ${total.toLocaleString()} TEST RECORDS`;

    let html = `
        <table style="
            width:100%;
            min-width:520px;
            border-collapse:separate;
            border-spacing:6px;
            text-align:center;
            font-size:12px;
        ">
            <thead>
                <tr>
                    <th style="padding:12px; color:#94a3b8;">
                        Actual ↓ / Predicted →
                    </th>
    `;

    classNames.forEach(name => {
        html += `
            <th style="
                padding:12px 8px;
                color:#22d3ee;
                font-size:11px;
            ">
                ${name}
            </th>
        `;
    });

    html += `</tr></thead><tbody>`;

    matrix.forEach((row, rowIndex) => {

        html += `
            <tr>
                <th style="
                    padding:12px 8px;
                    text-align:left;
                    color:#22d3ee;
                    font-size:11px;
                    white-space:nowrap;
                ">
                    ${classNames[rowIndex]}
                </th>
        `;

        row.forEach((value, columnIndex) => {

            const intensity = value / maxValue;

            const isCorrect = rowIndex === columnIndex;

            const background = isCorrect
                ? `rgba(34,211,238,${0.12 + intensity * 0.65})`
                : `rgba(167,139,250,${0.08 + intensity * 0.55})`;

            const borderColor = isCorrect
                ? "rgba(34,211,238,0.45)"
                : "rgba(167,139,250,0.3)";

            html += `
                <td style="
                    padding:22px 8px;
                    min-width:110px;
                    border-radius:9px;
                    background:${background};
                    border:1px solid ${borderColor};
                    color:#ffffff;
                    font-size:15px;
                    font-weight:700;
                ">
                    ${value.toLocaleString()}
                    <div style="
                        margin-top:5px;
                        font-size:10px;
                        color:#d1d5db;
                        font-weight:400;
                    ">
                        ${isCorrect ? "Correct" : "Incorrect"}
                    </div>
                </td>
            `;
        });

        html += `</tr>`;
    });

    html += `</tbody></table>`;

    display.innerHTML = html;

    // Research observation for selected model

    const lightCorrect = matrix[1][1];
    const lightTotal = matrix[1].reduce(
        (sum, value) => sum + value,
        0
    );

    const lightRecall = lightTotal > 0
        ? (lightCorrect / lightTotal) * 100
        : 0;

    const observation = document.getElementById(
        "matrixObservation"
    );

    observation.innerHTML = `
        <strong style="color:#a78bfa;">
            Model observation:
        </strong>
        This model correctly identifies
        <strong>${lightCorrect.toLocaleString()}</strong>
        out of
        <strong>${lightTotal.toLocaleString()}</strong>
        LIGHT_ATTACK records.

        Its LIGHT_ATTACK recall is
        <strong>${lightRecall.toFixed(2)}%</strong>.
        This demonstrates the difficulty of detecting
        LIGHT_ATTACK traffic in the current test dataset.
    `;
}

// =====================================================
// 10. CONFUSION MATRIX SELECTOR EVENTS
// =====================================================

document.getElementById("matrixFeatureGroup")
    .addEventListener("change", displayConfusionMatrix);

document.getElementById("matrixModel")
    .addEventListener("change", displayConfusionMatrix);

// =====================================================
// 11. INITIALISE DASHBOARD SECTIONS
// =====================================================

createPerformanceCharts();
displayPerformanceTable();
displayConfusionMatrix();

// =====================================================
// 12. HANDLE ANALYSE BUTTON
// =====================================================

analyseButton.addEventListener("click", async function () {

    const file = csvFileInput.files[0];

    if (!file) {

        statusMessage.textContent =
            "Please select a CSV file first.";

        statusMessage.style.color = "#fbbf24";

        return;
    }

    const featureGroup =
        document.getElementById("featureGroup").value;

    const modelName =
        document.getElementById("modelName").value;

    const formData = new FormData();

    formData.append("file", file);
    formData.append("featureGroup", featureGroup);
    formData.append("modelName", modelName);

    analyseButton.disabled = true;
    analyseButton.textContent = "ANALYSING DNS TRAFFIC...";

    statusMessage.textContent =
        "Uploading dataset and running machine-learning predictions...";

    statusMessage.style.color = "#22d3ee";

    resultsPanel.style.display = "none";

    try {

        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(
                data.error || "Prediction request failed."
            );
        }

        displayResults(data);

        statusMessage.textContent =
            "Analysis completed successfully.";

        statusMessage.style.color = "#4ade80";

    } catch (error) {

        statusMessage.textContent =
            "Analysis failed: " + error.message;

        statusMessage.style.color = "#f87171";

    } finally {

        analyseButton.disabled = false;

        analyseButton.textContent =
            "◈  ANALYSE DNS TRAFFIC";
    }
});

// =====================================================
// 13. DISPLAY UPLOADED CSV PREDICTION RESULTS
// =====================================================

function displayResults(data) {

    const counts = data.predictionCounts;
    const percentages = data.predictionPercentages;

    document.getElementById("resultDetails").textContent =
        `${data.filename} | ${data.featureGroup} features | ` +
        `${data.modelName} | ${data.totalRecords.toLocaleString()} records`;

    const cards = document.getElementById("resultCards");

    cards.replaceChildren();

    const categories = [
        {
            name: "BENIGN",
            key: "BENIGN",
            color: "#34d399"
        },
        {
            name: "LIGHT ATTACK",
            key: "LIGHT_ATTACK",
            color: "#fbbf24"
        },
        {
            name: "HEAVY ATTACK",
            key: "HEAVY_ATTACK",
            color: "#fb7185"
        }
    ];

    categories.forEach(category => {

        const key = category.key;

        const card = document.createElement("div");

        card.style.cssText = `
            padding:18px;
            border-radius:12px;
            background:rgba(5,10,23,0.75);
            border:1px solid ${category.color}44;
        `;

        card.appendChild(
            createTextElement(
                "p",
                category.name,
                `font-size:11px; color:${category.color}; font-weight:700;`
            )
        );

        card.appendChild(
            createTextElement(
                "h2",
                (counts[key] || 0).toLocaleString(),
                "font-size:25px; margin:10px 0;"
            )
        );

        card.appendChild(
            createTextElement(
                "p",
                `${(percentages[key] || 0).toFixed(2)}% of records`,
                "font-size:11px; color:#94a3b8;"
            )
        );

        cards.appendChild(card);
    });

    // Display first 10 predictions

    const tableBody =
        document.querySelector("#previewTable tbody");

    tableBody.replaceChildren();

    data.preview.forEach(result => {

        const row = document.createElement("tr");

        row.style.borderTop =
            "1px solid rgba(148,163,184,0.12)";

        const values = [
            result.rowNumber,
            result.predictedLabel,
            `${(result.confidence * 100).toFixed(2)}%`
        ];

        values.forEach(value => {

            const cell = document.createElement("td");

            cell.textContent = value;
            cell.style.padding = "12px 10px";

            row.appendChild(cell);
        });

        tableBody.appendChild(row);
    });

    resultsPanel.style.display = "block";

    resultsPanel.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}
