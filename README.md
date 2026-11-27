# DNSpectra
## AI-Powered DNS Tunnelling Detection and Threat Analysis

DNSpectra is a Java-based machine-learning application that analyses DNS traffic records from CSV files and classifies them as **BENIGN**, **LIGHT_ATTACK**, or **HEAVY_ATTACK**. It provides an interactive dashboard for inspecting predictions, comparing models, viewing confusion matrices, and exploring feature importance.

> **Important:** A model prediction is an indication based on learned patterns, not proof that a device or network is compromised. Results should be reviewed alongside other security evidence.

## Project objectives

- Analyse DNS traffic records using extracted DNS features.
- Classify records into benign and two attack categories.
- Compare multiple machine-learning algorithms.
- Compare Stateful and Stateless feature groups.
- Present predictions and model-analysis results through a web dashboard.
- Make the model's behaviour easier to inspect through confusion matrices and feature-importance visualisations.

## Main features

- CSV upload and input validation.
- Three-class prediction: `BENIGN`, `LIGHT_ATTACK`, and `HEAVY_ATTACK`.
- Three machine-learning models:
  - Decision Tree
  - Random Forest
  - Logistic Regression
- Stateful and Stateless feature-group analysis.
- Prediction counts, percentages, and a prediction preview.
- Model comparison visualisation.
- Confusion matrix visualisation.
- Feature-importance visualisation.
- Java Spring Boot backend and browser-based dashboard.

## Technology stack

| Area | Technology |
|---|---|
| Programming language | Java |
| Backend | Spring Boot |
| Machine learning | Weka |
| Build and dependency management | Maven |
| Frontend | HTML, CSS, JavaScript |
| Model storage | Serialized Weka models |
| Version control | Git and GitHub |
| Deployment preparation | Docker |

## System overview

1. The user uploads a DNS traffic CSV file through the dashboard.
2. The backend validates the file and checks that the required input features are available.
3. The selected feature group and trained model are used to generate predictions.
4. The application groups the predictions into BENIGN, LIGHT_ATTACK, and HEAVY_ATTACK.
5. The dashboard displays prediction summaries and visual analysis.

DNSpectra currently analyses uploaded CSV files. It is not a live DNS monitor and does not capture network traffic directly.

## Machine-learning models

DNSpectra compares three supervised learning algorithms:

- **Decision Tree:** learns a sequence of feature-based decision rules.
- **Random Forest:** combines predictions from multiple decision trees.
- **Logistic Regression:** estimates class probabilities using a learned statistical model.

The application supports Stateful and Stateless feature groups. These represent different sets of extracted DNS traffic characteristics.

## Dataset

The project uses the **CIC-Bell-DNS-EXF-2021** dataset.

Dataset page:  
https://www.unb.ca/cic/datasets/dns-exf-2021.html

Please consult the dataset provider's page for the original dataset description, terms, and citation requirements.

Raw and processed dataset files are intentionally not included in this repository. The trained model files are included so the application can load its models without committing the full dataset.

## Experimental evaluation

The current baseline evaluation produced the following results:

| Feature group | Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| Stateful | Decision Tree | 80.83% | 0.5613 |
| Stateful | Random Forest | 80.75% | 0.5607 |
| Stateful | Logistic Regression | 81.89% | 0.5551 |
| Stateless | Decision Tree | 79.60% | 0.5336 |
| Stateless | Random Forest | 79.62% | 0.5338 |
| Stateless | Logistic Regression | 79.23% | 0.5312 |

These figures are the recorded baseline evaluation results. They should not be interpreted as a guarantee of performance on other networks or datasets.

### Interpretation

- The Stateful Logistic Regression model achieved the highest recorded accuracy (81.89%) among the six configurations.
- The Stateful Decision Tree achieved the highest recorded macro F1-score (0.5613), although the difference from the Stateful Random Forest was small.
- The Stateful configurations performed slightly better than the Stateless configurations in this evaluation.
- Macro F1 is important because it gives each class equal weight, helping reveal performance on less frequent or harder-to-detect classes.
- The baseline models had very low F1 performance for the LIGHT_ATTACK class. This indicates that the models struggled to distinguish this category reliably.
- Some benign test traffic was also classified as suspicious, showing that false positives remain an issue.

### Evaluation limitations

The evaluation used a capture-level holdout: captures were separated into training and testing groups rather than randomly splitting individual rows from the same capture. This helps reduce the risk of overly optimistic results caused by near-duplicate records appearing in both sets.

However, the test set contained only one benign capture, alongside a limited selection of held-out light- and heavy-attack captures. The results therefore represent performance on these selected captures and should not be generalised to all DNS environments.

The LIGHT_ATTACK results and false positives on benign traffic are important limitations. Further work should investigate class imbalance, feature quality, decision thresholds, additional capture diversity, and validation on independent DNS traffic.

## Feature importance

DNSpectra includes feature-importance visualisations to show which input features most influenced a model's decisions.

In the recorded experiments:

- Stateful features included `rr_type_A`, `rr_name_entropy`, and `rr`.
- Stateless features included `subdomain_length`, `labels`, and `labels_max`.

Feature importance describes how the trained model used features. It does **not** establish that a feature causes DNS tunnelling.

## Running the application

### Requirements

- Java Development Kit (JDK) compatible with the project configuration.
- IntelliJ IDEA or another Java IDE.
- Maven support.
- The project source code and the `models/` directory.

### Run using IntelliJ IDEA

1. Clone or download this repository.
2. Open the project folder containing `pom.xml` in IntelliJ IDEA.
3. Allow IntelliJ to load and resolve the Maven dependencies.
4. Confirm that the trained model files are present in the `models/` directory.
5. Open the Spring Boot application entry point: `DNSpectraApplication.java`.
6. Run the application.
7. Open `http://localhost:8080` in a browser.

If Maven is available in your terminal, the application can also be packaged with:

```bash
mvn clean package
```

Then run the generated Spring Boot JAR from the `target/` directory:

```bash
java -jar target/<generated-jar-name>.jar
```

Use the actual JAR filename generated by Maven.

## Repository contents

- `src/` — Java backend and web dashboard source files.
- `models/` — trained Stateful and Stateless model files.
- `pom.xml` — Maven project configuration.
- `Dockerfile` — container build and runtime configuration.
- `.gitignore` — excludes local and dataset files from version control.

The exact source folders may contain additional packages and resources.

## Responsible interpretation

DNSpectra is an academic analysis and demonstration tool. It should not be used as the sole basis for incident response or security decisions. A suspicious classification is a signal for further investigation, not confirmation of malicious activity.

## Future improvements

- Improve detection of LIGHT_ATTACK traffic.
- Reduce false positives on benign DNS traffic.
- Evaluate using more diverse and independent DNS captures.
- Explore class-imbalance handling and additional feature engineering.
- Expand model explainability and per-class evaluation.
- Complete and validate public cloud deployment.

## Acknowledgements

Dataset: Canadian Institute for Cybersecurity, CIC-Bell-DNS-EXF-2021. Please follow the dataset provider's citation and usage requirements.

---

**Project:** DNSpectra — AI-Powered DNS Tunnelling Detection and Threat Analysis  
**Repository:** https://github.com/SAR-pixel3/DNSpectra
