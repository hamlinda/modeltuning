# Hamlin Software | AI Model Tuner & Evaluator - Agents Architecture & Solution View

## 1. Detailed Architecture

The system utilizes a client-server architecture engineered for local execution, ensuring low latency and complete privacy for dataset processing. The architecture is modularly separated into a Flask backend for model computation and a Vanilla JS/HTML/CSS frontend for interactive visualization.

### 1.1 High-Level Component Diagram

```mermaid
graph TD
    Client[Browser Frontend (Vanilla JS / HTML / CSS)] <-->|HTTP JSON APIs| Backend[Flask Server app.py]
    
    subgraph "Backend System"
        Backend <-->|In-Memory Cache| DataStore[Pandas DataStore Dictionary]
        Backend <-->|Scikit-Learn / Custom Pipelines| MLEngine[ML Engine & Preprocessing]
    end
    
    subgraph "Frontend System"
        Client -->|Event Listeners| UILogic[DOM Mutation & Event Logic]
        UILogic -->|Data Binding| Forms[Hyperparameter & Feature Tables]
        UILogic -->|Rendering| Charts[Chart.js Visualization & Confusion Matrix]
    end
```

### 1.2 Backend Components (`app.py`)
- **Flask Framework Engine**: Exposes RESTful endpoints for model metadata, dataset uploads, dataset sampling, and evaluation.
- **In-Memory Data Store (`DATA_CACHE`)**: Caches processed `pandas` DataFrames using UUIDs (`dataset_token`). This architectural decision guarantees that sensitive CSV data resides only in memory, mitigating disk storage vulnerabilities and disk I/O bottlenecks.
- **Dynamic ML Pipeline Builder**: Translates JSON requests from the UI into fully configured Scikit-Learn pipelines. This component intelligently binds preprocessing transforms (e.g., `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `OrdinalEncoder`) to individual columns based on the UI's selections.
- **Evaluation Engine**: Responsible for splitting data, fitting the model, and extracting performance metrics (Accuracy, F1, RMSE, R2, etc.) along with feature importances and confusion matrices.

### 1.3 Communication Protocol (API)
- **`GET /`**: Serves the main SPA interface.
- **`GET /api/models`**: Provides configuration schemas (algorithms and hyperparameter bounds) to dynamically construct the tuning UI.
- **`GET /api/sample-data`**: Injects synthetic datasets (e.g., Housing, Churn) directly into the in-memory cache and returns schema characteristics.
- **`POST /api/analyze-csv`**: Handles multi-part file uploads, strictly restricting payload size (5MB limit) and caching it securely.
- **`POST /api/evaluate`**: Receives hyperparameter tuning state and feature scaling rules, executing model training synchronously and returning metrics/visualization coordinates.

---

## 2. Solution View

The solution is an integrated, single-page application (SPA) focused on removing the friction of iterative model tuning. Instead of writing repetitive boilerplate for pandas, scikit-learn, and matplotlib, a user can instantly load a CSV, dynamically assemble a pipeline with graphical controls, and observe the immediate effect on metrics and fit curves.

- **Data Privacy by Default**: Zero persistent storage of data payloads.
- **Safety First UI Constraints**: Hyperparameter inputs dynamically clamp to acceptable bounds as defined by the backend schema, ensuring execution stability.
- **Type-Aware Guidance**: The frontend logic proactively disables incompatible ML algorithms (e.g., preventing linear regression from targeting categorical strings).

---

## 3. Comprehensive UI Element Identification

The user interface follows a three-panel "Glassmorphism" grid design, segregating the workflow into logical steps.

### 3.1 Global Elements
- **App Header (`.app-header`)**: Contains the brand logo, a "Tuning Guide" modal trigger (`#tuning-guide-link`), and a live `Server: Idle`/`Server: Computing` status indicator (`#status-text`, `#status-pulse`).
- **Tuning Guide Modal (`#tuning-modal`)**: A popup containing educational tables for tuning Regressors/Classifiers, Computer Vision models, and LLMs.
- **Error Toast Notification (`#error-toast`)**: A floating element for non-blocking HTTP or validation error alerts.

### 3.2 Panel 1: Dataset Configuration (`#data-panel`)
This section manages data ingestion and basic splitting.
- **CSV Dropzone (`#file-dropzone`)**: A drag-and-drop region encompassing a hidden `<input type="file" id="csv-file-input">`. Includes SVG icons and textual prompts.
- **File Info Box (`#dropzone-file-info`)**: Appears after successful upload, displaying the filename (`#selected-file-name`) and a clear button (`#btn-clear-file`).
- **Demo Data Triggers**: Quick-load buttons for predefined data. Includes `#btn-demo-housing` (Regression), `#btn-demo-churn` (Classification), `#btn-demo-cv`, and `#btn-demo-llm`.
- **Target Selector (`#target-select`)**: A `<select>` dropdown populated dynamically with column headers, designating the `y` variable.
- **Train/Test Split Slider (`#split-slider`)**: An HTML range input (50% to 95%) determining the holdout subset, with an active label (`#split-value-label`).

### 3.3 Panel 2: Model & Features (`#config-panel`)
This section allows technical tuning of algorithms and data transformations.
- **Algorithm Selector (`#model-select`)**: Grouped by classification (`#model-optgroup-clf`), regression (`#model-optgroup-reg`), CV, and LLM via `<optgroup>`.
- **Optimization Goal / Experience (`#experience-select`)**: Populated dynamically, allows users to choose parameter presets. Its description appears in `#experience-description`.
- **Hyperparameter Grid (`#hyperparams-container`)**: An empty container dynamically injected with form controls (sliders, inputs) representing model-specific kwargs (e.g., `n_estimators`, `max_depth`).
- **Feature Configuration Table (`.features-table`)**:
  - `col-active`: Checkboxes to include/exclude features.
  - `col-name`: Name of the feature.
  - `col-type`: Display of inferred data type.
  - `col-transform`: `<select>` element per feature to assign scalers/encoders.

### 3.4 Panel 3: Results Dashboard (`#results-panel`)
The execution and visualization zone.
- **Execution Button (`#btn-run-evaluation`)**: Primary action trigger to POST the configuration. Contains a glowing effect span.
- **Console Output Logs (`#console-box`)**: A scrolling terminal-like `div` (`#console-log`) simulating execution progress (e.g., pipeline building, fitting, scoring).
- **Performance Metrics Grid (`#metrics-grid`)**: Dynamically populated container for metric cards (Accuracy, R2, RMSE).
- **Chart Tab Controls (`.chart-tabs`)**: Toggle buttons switching the view context:
  - Model Fit (`#tab-primary`)
  - Feature Importance (`#tab-secondary`)
  - Confusion Matrix (`#tab-tertiary`)
- **Chart Viewbox (`.charts-viewbox`)**:
  - Primary Canvas (`#chart-primary`): Rendered via Chart.js for predictions or fit curves.
  - Secondary Canvas (`#chart-secondary`): Rendered via Chart.js for feature importance bar charts.
  - Confusion Matrix Grid (`#confusion-matrix-grid`): A CSS-grid based interactive heatmap generated purely via secure DOM manipulation.
