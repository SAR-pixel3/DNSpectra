
// =====================================================
// DNSPECTRA - FEATURE IMPORTANCE VISUALISATION
// Based on actual Weka Random Forest experiments
// =====================================================

// 1. ACTUAL EXPERIMENTAL RESULTS

const featureImportanceData = {
    stateful: [
        { name: "rr_type_A", value: 0.28 },
        { name: "rr_name_entropy", value: 0.12 },
        { name: "rr", value: 0.08 },
        { name: "A_frequency", value: 0.07 },
        { name: "rr_type_PTR", value: 0.07 },
        { name: "ttl_mean", value: 0.06 },
        { name: "rr_name_length", value: 0.05 },
        { name: "rr_type_AAAA", value: 0.05 },
        { name: "AAAA_frequency", value: 0.03 },
        { name: "rr_type_empty", value: 0.03 },
        { name: "TXT_frequency", value: 0.03 },
        { name: "ttl_variance", value: 0.02 },
        { name: "rr_type_missing", value: 0.02 },
        { name: "PTR_frequency", value: 0.02 },
        { name: "rr_count", value: 0.02 },
        { name: "rr_type_TXT", value: 0.01 },
        { name: "distinct_ns", value: 0.01 },
        { name: "CNAME_frequency", value: 0 },
        { name: "NS_frequency", value: 0 },
        { name: "a_records", value: 0 },
        { name: "SOA_frequency", value: 0 },
        { name: "SRV_frequency", value: 0 },
        { name: "OPT_frequency", value: 0 },
        { name: "MX_frequency", value: 0 },
        { name: "HINFO_frequency", value: 0 },
        { name: "NULL_frequency", value: 0 }
    ],

    stateless: [
        { name: "subdomain_length", value: 0.16 },
        { name: "labels", value: 0.14 },
        { name: "labels_max", value: 0.13 },
        { name: "upper", value: 0.13 },
        { name: "entropy", value: 0.12 },
        { name: "lower", value: 0.11 },
        { name: "FQDN_count", value: 0.11 },
        { name: "labels_average", value: 0.09 },
        { name: "special", value: 0.09 },
        { name: "len", value: 0.08 },
        { name: "numeric", value: 0.08 },
        { name: "subdomain", value: 0.02 }
    ]
};

// 2. CREATE DASHBOARD SECTION

const importancePanel = document.createElement("section");

importancePanel.id = "featureImportancePanel";

importancePanel.style.cssText = `
    margin-top: 25px;
    padding: 25px;
    border-radius: 15px;
    background: rgba(17, 25, 45, 0.92);
    border: 1px solid rgba(167, 139, 250, 0.25);
    color: #f1f5f9;
`;

importancePanel.innerHTML = `
    <h2 style="font-size:20px; margin-bottom:8px;">
        ◈ Feature Importance Analysis
    </h2>

    <p style="
        font-size:12px;
        color:#94a3b8;
        line-height:1.8;
        margin-bottom:22px;
    ">
        Explore which DNS traffic features contributed most
        to the Random Forest model's classification decisions.
        Switch between the two feature representations.
    </p>

    <div style="margin-bottom:22px;">
        <label for="importanceFeatureGroup"
               style="
                   display:block;
                   font-size:12px;
                   color:#94a3b8;
                   margin-bottom:8px;
               ">
            FEATURE REPRESENTATION
        </label>

        <select id="importanceFeatureGroup"
                style="
                    width:100%;
                    max-width:350px;
                    padding:12px;
                    border-radius:8px;
                    background:#0b1224;
                    color:#f1f5f9;
                    border:1px solid rgba(167,139,250,0.4);
                ">
            <option value="stateful">Stateful Features</option>
            <option value="stateless">Stateless Features</option>
        </select>
    </div>

    <div id="importanceSummary"
         style="
             display:grid;
             grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
             gap:12px;
             margin-bottom:25px;
         ">
    </div>

    <div style="
        padding:20px;
        border-radius:12px;
        background:rgba(5,10,23,0.75);
        border:1px solid rgba(148,163,184,0.12);
    ">
        <h3 style="font-size:15px; margin-bottom:20px;">
            Feature Ranking
        </h3>

        <div id="importanceChart"></div>
    </div>

    <div style="
        margin-top:22px;
        padding:15px;
        border-radius:10px;
        background:rgba(34,211,238,0.05);
        border:1px solid rgba(34,211,238,0.15);
    ">
        <p id="importanceObservation"
           style="
               font-size:12px;
               color:#cbd5e1;
               line-height:1.8;
           ">
        </p>
    </div>

    <p style="
        font-size:10px;
        color:#64748b;
        line-height:1.8;
        margin-top:15px;
    ">
        Method: Random Forest attribute importance based on
        average impurity decrease, calculated using Weka.
        Bar lengths are scaled relative to the highest score
        within the selected feature group. Displayed values
        are the original experimental importance scores.
    </p>
`;

// Add section to dashboard

document.querySelector("main").appendChild(importancePanel);


// 3. DISPLAY FEATURE IMPORTANCE

function displayFeatureImportance() {

    const featureGroup =
        document.getElementById("importanceFeatureGroup").value;

    const features = featureImportanceData[featureGroup];

    const chart =
        document.getElementById("importanceChart");

    const summary =
        document.getElementById("importanceSummary");

    const observation =
        document.getElementById("importanceObservation");

    chart.replaceChildren();
    summary.replaceChildren();

    // Sort features from highest to lowest importance

    const sortedFeatures = [...features].sort(
        (a, b) => b.value - a.value
    );

    const highestFeature = sortedFeatures[0];

    const maxValue = highestFeature.value;

    // Summary cards

    const summaryItems = [
        {
            title: "Total Features",
            value: features.length
        },
        {
            title: "Highest Importance",
            value: highestFeature.value.toFixed(2)
        },
        {
            title: "Top Feature",
            value: highestFeature.name
        }
    ];

    summaryItems.forEach(item => {

        const card = document.createElement("div");

        card.style.cssText = `
            padding:17px;
            border-radius:11px;
            background:rgba(5,10,23,0.75);
            border:1px solid rgba(34,211,238,0.15);
        `;

        const heading = createTextElement(
            "p",
            item.title,
            "font-size:11px; color:#94a3b8; margin-bottom:9px;"
        );

        const value = createTextElement(
            "h3",
            item.value,
            "font-size:17px; color:#22d3ee; overflow-wrap:anywhere;"
        );

        card.appendChild(heading);
        card.appendChild(value);

        summary.appendChild(card);
    });

    // Draw horizontal bars

    sortedFeatures.forEach((feature, index) => {

        const item = document.createElement("div");

        item.style.cssText = `
            margin-bottom:17px;
        `;

        const labelRow = document.createElement("div");

        labelRow.style.cssText = `
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:12px;
            margin-bottom:7px;
            font-size:11px;
        `;

        const name = document.createElement("span");

        name.textContent =
            `${index + 1}. ${feature.name}`;

        name.style.cssText = `
            color:#cbd5e1;
            overflow-wrap:anywhere;
        `;

        const value = document.createElement("strong");

        value.textContent = feature.value.toFixed(2);

        value.style.cssText = `
            color:#22d3ee;
            font-variant-numeric:tabular-nums;
        `;

        labelRow.appendChild(name);
        labelRow.appendChild(value);

        const track = document.createElement("div");

        track.style.cssText = `
            width:100%;
            height:9px;
            border-radius:20px;
            background:rgba(148,163,184,0.12);
            overflow:hidden;
        `;

        const bar = document.createElement("div");

        const width = maxValue > 0
            ? (feature.value / maxValue) * 100
            : 0;

        bar.style.cssText = `
            width:${width}%;
            height:100%;
            border-radius:20px;
            background:linear-gradient(90deg,#22d3ee,#a78bfa);
            transition:width 0.4s ease;
        `;

        track.appendChild(bar);

        item.appendChild(labelRow);
        item.appendChild(track);

        chart.appendChild(item);
    });

    // Research observation

    const groupName =
        featureGroup === "stateful"
            ? "Stateful"
            : "Stateless";

    observation.innerHTML = `
        <strong style="color:#a78bfa;">
            Research observation:
        </strong>

        The most influential ${groupName.toLowerCase()}
        feature in this Random Forest experiment was

        <strong style="color:#22d3ee;">
            ${highestFeature.name}
        </strong>

        with an importance score of

        <strong>
            ${highestFeature.value.toFixed(2)}
        </strong>.

        This score indicates the feature's contribution
        to the model's impurity reduction. It does not
        establish that the feature causes DNS tunnelling.
    `;
}


// 4. FEATURE GROUP SELECTOR

document.getElementById("importanceFeatureGroup")
    .addEventListener("change", displayFeatureImportance);


// 5. INITIALISE

displayFeatureImportance();
