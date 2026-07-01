/**
 * ANTIGRAVITY MODEL TUNER - CLIENT SIDE CONTROLLER
 * 
 * UI-TO-CODE ASSOCIATION MAP:
 * =============================================================================
 * HTML Element ID / Class       ->  JS Event Handler / State Variable  ->  Backend Endpoint
 * -----------------------------------------------------------------------------
 * #csv-file-input              ->  handleFileUpload()                 ->  POST /api/analyze-csv
 * #btn-demo-housing            ->  loadSampleDataset('housing')        ->  GET /api/sample-data
 * #btn-demo-churn              ->  loadSampleDataset('churn')          ->  GET /api/sample-data
 * #target-select               ->  handleTargetSelection()             ->  Updates state
 * #split-slider                ->  Updates split label                 ->  Passed in payload
 * #model-select                ->  handleModelSelection()              ->  GET /api/models schema
 * .hyperparam-input            ->  Collected in evaluateModel()        ->  POST /api/evaluate
 * .feature-row checkbox/select ->  Collected in evaluateModel()        ->  POST /api/evaluate
 * #btn-run-evaluation          ->  evaluateModel()                     ->  POST /api/evaluate
 * =============================================================================
 */

// Application Global State
let appState = {
    datasetToken: null,
    columns: [],
    targetColumn: null,
    targetType: null,
    activeModel: null,
    modelsSchema: {},
    primaryChart: null,
    secondaryChart: null
};

// Console Log Helper
// Safety: Writes message strictly using textContent to prevent script injection
function logConsole(message, type = "info") {
    const consoleLog = document.getElementById("console-log");
    if (!consoleLog) return;

    const line = document.createElement("div");
    line.className = "log-line";
    if (type === "system") {
        line.classList.add("system-msg");
    } else if (type === "error") {
        line.classList.add("error-msg");
    }
    
    // Safety check: always use textContent over innerHTML
    line.textContent = `> [${new Date().toLocaleTimeString()}] ${message}`;
    consoleLog.appendChild(line);
    consoleLog.scrollTop = consoleLog.scrollHeight;
}

// Show/Hide toast messages
function showToast(message, type = "error") {
    const toast = document.getElementById("error-toast");
    const toastMsg = document.getElementById("toast-message");
    if (!toast || !toastMsg) return;

    toastMsg.textContent = message;
    toast.classList.remove("hidden");

    // Auto close after 5 seconds
    const timeout = setTimeout(() => {
        toast.classList.add("hidden");
    }, 5000);

    const closeBtn = document.getElementById("toast-close");
    if (closeBtn) {
        closeBtn.onclick = () => {
            clearTimeout(timeout);
            toast.classList.add("hidden");
        };
    }
}

// Set status indicator
function updateServerStatus(text, busy = false, error = false) {
    const statusPulse = document.getElementById("status-pulse");
    const statusText = document.getElementById("status-text");
    if (!statusPulse || !statusText) return;

    statusText.textContent = `Server: ${text}`;
    
    statusPulse.className = "status-pulse";
    if (busy) {
        statusPulse.classList.add("busy");
    } else if (error) {
        statusPulse.classList.add("error");
    }
}

// Initial Loading Handlers
document.addEventListener("DOMContentLoaded", () => {
    logConsole("Initializing Model Tuner workspace...", "system");
    loadModelsSchema();
    setupDropzone();
    setupEventListeners();
});

// Load supported models and their schema from Flask backend API
function loadModelsSchema() {
    updateServerStatus("Querying models schema...", true);
    fetch("/api/models")
        .then(res => {
            if (!res.ok) throw new Error("Failed to load models list");
            return res.json();
        })
        .then(data => {
            appState.modelsSchema = data;
            populateModelDropdown(data);
            updateServerStatus("Idle");
            logConsole("Supported machine learning algorithms cached successfully.", "system");
        })
        .catch(err => {
            updateServerStatus("Failed to load schema", false, true);
            logConsole(`Connection error: ${err.message}`, "error");
            showToast("Failed to initialize models from backend. Verify local server execution.");
        });
}

// Set up event listeners for main dashboard inputs
function setupEventListeners() {
    // Train/Test Split Slider
    const splitSlider = document.getElementById("split-slider");
    const splitLabel = document.getElementById("split-value-label");
    if (splitSlider && splitLabel) {
        splitSlider.addEventListener("input", (e) => {
            const trainPercent = e.target.value;
            const testPercent = 100 - trainPercent;
            splitLabel.textContent = `${trainPercent}% / ${testPercent}%`;
        });
    }

    // Demo Datasets loading
    const demoHousing = document.getElementById("btn-demo-housing");
    const demoChurn = document.getElementById("btn-demo-churn");
    const demoCV = document.getElementById("btn-demo-cv");
    const demoLLM = document.getElementById("btn-demo-llm");
    if (demoHousing) demoHousing.addEventListener("click", () => loadSampleDataset("housing"));
    if (demoChurn) demoChurn.addEventListener("click", () => loadSampleDataset("churn"));
    if (demoCV) demoCV.addEventListener("click", () => loadSampleDataset("cv_images"));
    if (demoLLM) demoLLM.addEventListener("click", () => loadSampleDataset("llm_text"));

    // Target Column Dropdown Selection
    const targetSelect = document.getElementById("target-select");
    if (targetSelect) {
        targetSelect.addEventListener("change", (e) => {
            handleTargetSelection(e.target.value);
        });
    }

    // Model Selection Dropdown
    const modelSelect = document.getElementById("model-select");
    if (modelSelect) {
        modelSelect.addEventListener("change", (e) => {
            handleModelSelection(e.target.value);
        });
    }

    // Execute button trigger
    const runBtn = document.getElementById("btn-run-evaluation");
    if (runBtn) {
        runBtn.addEventListener("click", evaluateModel);
    }

    // Chart Tabs setup
    const tabs = document.querySelectorAll(".chart-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", (e) => {
            // Remove active states
            tabs.forEach(t => t.classList.remove("active"));
            
            // Set current active
            e.target.classList.add("active");
            
            // Toggle wrapper wrappers visibility
            const targetId = e.target.getAttribute("data-target");
            const wrappers = document.querySelectorAll(".chart-wrapper");
            wrappers.forEach(w => {
                if (w.id === targetId) {
                    w.classList.remove("hidden");
                } else {
                    w.classList.add("hidden");
                }
            });
        });
    });

    // Tuning Modal Logic
    const tuningLink = document.getElementById('tuning-guide-link');
    const tuningModal = document.getElementById('tuning-modal');
    const modalCloseBtn = document.getElementById('modal-close-btn');

    if (tuningLink && tuningModal && modalCloseBtn) {
        tuningLink.addEventListener('click', (e) => {
            e.preventDefault();
            tuningModal.classList.add('active');
        });

        modalCloseBtn.addEventListener('click', () => {
            tuningModal.classList.remove('active');
        });

        // Close on outside click
        tuningModal.addEventListener('click', (e) => {
            if (e.target === tuningModal) {
                tuningModal.classList.remove('active');
            }
        });
    }
}

// Setup Drag & Drop File Upload listeners
function setupDropzone() {
    const dropzone = document.getElementById("file-dropzone");
    const fileInput = document.getElementById("csv-file-input");
    const clearFileBtn = document.getElementById("btn-clear-file");
    const selectedFileName = document.getElementById("selected-file-name");
    const dropzonePrompt = dropzone ? dropzone.querySelector(".dropzone-prompt") : null;
    const dropzoneInfo = document.getElementById("dropzone-file-info");

    if (!dropzone || !fileInput) return;

    // Trigger file dialog on dropzone click
    dropzone.addEventListener("click", (e) => {
        // Only trigger if clicking dropzone prompt areas (ignore input itself or clear button)
        if (e.target !== clearFileBtn && !clearFileBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    // Handle drag styling highlights
    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.style.borderColor = "var(--primary)";
            dropzone.style.background = "rgba(59, 130, 246, 0.08)";
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.style.borderColor = "hsla(217, 91%, 60%, 0.25)";
            dropzone.style.background = "rgba(59, 130, 246, 0.02)";
        }, false);
    });

    // Drop file handler
    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileUpload(files[0]);
        }
    });

    // Input selection handler
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Clear file handler
    if (clearFileBtn) {
        clearFileBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            fileInput.value = "";
            if (dropzonePrompt) dropzonePrompt.classList.remove("hidden");
            if (dropzoneInfo) dropzoneInfo.classList.add("hidden");
            resetWorkspace();
            logConsole("CSV dataset unloaded.", "system");
        });
    }
}

// Process CSV uploading to local backend API
function handleFileUpload(file) {
    const dropzonePrompt = document.querySelector(".dropzone-prompt");
    const dropzoneInfo = document.getElementById("dropzone-file-info");
    const selectedFileName = document.getElementById("selected-file-name");
    
    if (!file) return;

    if (dropzonePrompt) dropzonePrompt.classList.add("hidden");
    if (dropzoneInfo) dropzoneInfo.classList.remove("hidden");
    if (selectedFileName) selectedFileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;

    logConsole(`Uploading dataset: ${file.name}...`, "system");
    updateServerStatus("Processing CSV upload...", true);

    const formData = new FormData();
    formData.append("file", file);

    fetch("/api/analyze-csv", {
        method: "POST",
        body: formData
    })
    .then(res => {
        if (!res.ok) {
            return res.json().then(json => { throw new Error(json.error || "Upload failed"); });
        }
        return res.json();
    })
    .then(data => {
        updateServerStatus("Idle");
        logConsole(`Dataset loaded successfully. Assigned session ID: ${data.dataset_token.slice(0, 8)}...`, "system");
        logConsole(`Columns detected: ${data.columns.length}. Custom dataset ready for configuration.`, "info");
        
        initializeDataState(data);
    })
    .catch(err => {
        updateServerStatus("Upload failed", false, true);
        logConsole(`Upload failed: ${err.message}`, "error");
        showToast(err.message || "Failed to upload file. Make sure file is a valid CSV less than 5MB.");
        
        // Reset dropzone
        if (dropzonePrompt) dropzonePrompt.classList.remove("hidden");
        if (dropzoneInfo) dropzoneInfo.classList.add("hidden");
        fileInput.value = "";
        resetWorkspace();
    });
}

// Load a sample/demo dataset from the local mock API
function loadSampleDataset(type) {
    logConsole(`Requesting sample dataset '${type}'...`, "system");
    updateServerStatus("Fetching sample dataset...", true);

    fetch(`/api/sample-data?type=${type}`)
        .then(res => {
            if (!res.ok) throw new Error("Failed to load sample dataset");
            return res.json();
        })
        .then(data => {
            updateServerStatus("Idle");
            logConsole(`Demo dataset loaded successfully: sample_${type}`, "system");
            logConsole(`Features: ${data.columns.length - 1} | Samples: 500+`, "info");
            
            // Update UI dropzone display with fake file indicator
            const dropzonePrompt = document.querySelector(".dropzone-prompt");
            const dropzoneInfo = document.getElementById("dropzone-file-info");
            const selectedFileName = document.getElementById("selected-file-name");
            
            if (dropzonePrompt) dropzonePrompt.classList.add("hidden");
            if (dropzoneInfo) dropzoneInfo.classList.remove("hidden");
            if (selectedFileName) selectedFileName.textContent = `Demo: ${type === 'housing' ? 'Housing Prices' : 'Customer Churn'} (Synthesized)`;

            initializeDataState(data);
        })
        .catch(err => {
            updateServerStatus("Error loading dataset", false, true);
            logConsole(`Error: ${err.message}`, "error");
            showToast("Failed to retrieve sample dataset from server.");
        });
}

// Initialize dataset configurations when loaded
function initializeDataState(data) {
    appState.datasetToken = data.dataset_token;
    appState.columns = data.columns;
    
    // Enable target selection and train split inputs
    const settingsGroup = document.getElementById("data-settings-group");
    if (settingsGroup) settingsGroup.classList.remove("disabled-state");

    populateTargetDropdown(data.columns);

    // Auto-select a recommended target variable if present
    const targetSelect = document.getElementById("target-select");
    if (targetSelect) {
        let defaultTarget = "";
        // Look for common target keywords
        const targetKeywords = ["price", "churned", "churn", "target", "label", "class", "outcome", "y"];
        for (const col of data.columns) {
            if (targetKeywords.includes(col.name.toLowerCase())) {
                defaultTarget = col.name;
                break;
            }
        }
        
        // If not found, select last column by default
        if (!defaultTarget && data.columns.length > 0) {
            defaultTarget = data.columns[data.columns.length - 1].name;
        }

        if (defaultTarget) {
            targetSelect.value = defaultTarget;
            handleTargetSelection(defaultTarget);
        }
    }
}

// Clean and reset the dashboard when datasets are cleared
function resetWorkspace() {
    appState.datasetToken = null;
    appState.columns = [];
    appState.targetColumn = null;
    appState.targetType = null;
    appState.activeModel = null;

    // Reset target dropdown
    const targetSelect = document.getElementById("target-select");
    if (targetSelect) {
        targetSelect.replaceChildren();
        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.disabled = true;
        placeholder.selected = true;
        placeholder.textContent = "Select target column...";
        targetSelect.appendChild(placeholder);
    }

    // Disable panels
    document.getElementById("data-settings-group").classList.add("disabled-state");
    document.getElementById("config-panel").classList.add("disabled-state");
    document.getElementById("btn-run-evaluation").classList.add("disabled-state");

    // Clear tables
    document.getElementById("features-table-body").replaceChildren();
    
    // Reset models
    document.getElementById("model-select").value = "";
    document.getElementById("hyperparams-section").classList.add("hidden");
    document.getElementById("hyperparams-container").replaceChildren();

    // Clear metrics and charts
    document.getElementById("metrics-container").classList.add("hidden");
    document.getElementById("charts-container").classList.add("hidden");

    if (appState.primaryChart) { appState.primaryChart.destroy(); appState.primaryChart = null; }
    if (appState.secondaryChart) { appState.secondaryChart.destroy(); appState.secondaryChart = null; }
}

// Populate the Target Selection dropdown list
function populateTargetDropdown(columns) {
    const targetSelect = document.getElementById("target-select");
    if (!targetSelect) return;

    // Clear existing
    targetSelect.replaceChildren();

    // Add prompt
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.disabled = true;
    placeholder.selected = true;
    placeholder.textContent = "Select target column...";
    targetSelect.appendChild(placeholder);

    columns.forEach(col => {
        const option = document.createElement("option");
        option.value = col.name;
        option.textContent = `${col.name} (${col.type === "numerical" ? "Numeric" : "Categorical"})`;
        targetSelect.appendChild(option);
    });
}

// Populate the model options grouped by type
function populateModelDropdown(schema) {
    const clfGroup = document.getElementById("model-optgroup-clf");
    const regGroup = document.getElementById("model-optgroup-reg");
    const cvGroup = document.getElementById("model-optgroup-cv");
    const llmGroup = document.getElementById("model-optgroup-llm");

    if (clfGroup) clfGroup.replaceChildren();
    if (regGroup) regGroup.replaceChildren();
    if (cvGroup) cvGroup.replaceChildren();
    if (llmGroup) llmGroup.replaceChildren();

    Object.entries(schema).forEach(([key, model]) => {
        const option = document.createElement("option");
        option.value = key;
        option.textContent = model.name;

        if (model.type === "classification" && clfGroup) {
            clfGroup.appendChild(option);
        } else if (model.type === "regression" && regGroup) {
            regGroup.appendChild(option);
        } else if (model.type === "cv" && cvGroup) {
            cvGroup.appendChild(option);
        } else if (model.type === "llm" && llmGroup) {
            llmGroup.appendChild(option);
        }
    });
}

// Triggered when a target variable is selected
function handleTargetSelection(targetName) {
    appState.targetColumn = targetName;
    
    const selectedCol = appState.columns.find(c => c.name === targetName);
    if (!selectedCol) return;

    let taskName = selectedCol.type === "numerical" ? "Regression Task" : "Classification Task";
    let internalTargetType = selectedCol.type;

    if (targetName === "Class" && appState.datasetToken && appState.datasetToken.includes("cv_images")) {
        taskName = "Computer Vision Task";
        internalTargetType = "cv";
    } else if (targetName === "Response" && appState.datasetToken && appState.datasetToken.includes("llm_text")) {
        taskName = "LLM Fine-Tuning Task";
        internalTargetType = "llm";
    }

    appState.targetType = internalTargetType;
    logConsole(`Target column set to: ${targetName} (${taskName})`, "system");

    // Toggle corresponding model algorithm availability in dropdown to assist selection
    updateModelOptions(internalTargetType);

    // Build the features listing table
    populateFeaturesTable(appState.columns, targetName);

    // Enable configurations panel
    document.getElementById("config-panel").classList.remove("disabled-state");
}

// Enable/Disable algorithms in selection dropdown based on target variable task type
function updateModelOptions(targetType) {
    const modelSelect = document.getElementById("model-select");
    if (!modelSelect) return;

    const clfGroup = document.getElementById("model-optgroup-clf");
    const regGroup = document.getElementById("model-optgroup-reg");
    const cvGroup = document.getElementById("model-optgroup-cv");
    const llmGroup = document.getElementById("model-optgroup-llm");

    // Enable all first
    if (clfGroup) clfGroup.disabled = false;
    if (regGroup) regGroup.disabled = false;
    if (cvGroup) cvGroup.disabled = false;
    if (llmGroup) llmGroup.disabled = false;

    // Reset selection if changing tasks
    if (appState.activeModel) {
        const selectedModelType = appState.modelsSchema[appState.activeModel].type;
        // 'categorical' maps to 'classification', 'numerical' to 'regression' for legacy models
        let expectedTargetType = selectedModelType;
        if (selectedModelType === "classification") expectedTargetType = "categorical";
        if (selectedModelType === "regression") expectedTargetType = "numerical";
        
        if (targetType !== expectedTargetType) {
            modelSelect.value = "";
            appState.activeModel = null;
            const hc = document.getElementById("hyperparams-section");
            if (hc) hc.classList.add("hidden");
            const runBtn = document.getElementById("btn-run-evaluation");
            if (runBtn) runBtn.classList.add("disabled-state");
        }
    }

    // Disable non-matching group
    if (clfGroup) clfGroup.disabled = (targetType !== "categorical");
    if (regGroup) regGroup.disabled = (targetType !== "numerical");
    if (cvGroup) cvGroup.disabled = (targetType !== "cv");
    if (llmGroup) llmGroup.disabled = (targetType !== "llm");
    
    // Recommend a model
    if (!modelSelect.value) {
        if (targetType === "numerical") modelSelect.value = "random_forest_regressor";
        if (targetType === "categorical") modelSelect.value = "random_forest_classifier";
        if (targetType === "cv") modelSelect.value = "resnet50_finetune";
        if (targetType === "llm") modelSelect.value = "llama3_lora";
        if (modelSelect.value) handleModelSelection(modelSelect.value);
    }
}

// Build features rows dynamically with custom transformations selectors
function populateFeaturesTable(columns, selectedTarget) {
    const tableBody = document.getElementById("features-table-body");
    if (!tableBody) return;

    tableBody.replaceChildren();

    columns.forEach(col => {
        // Target column itself cannot be a feature (prevent label leak)
        if (col.name === selectedTarget) return;

        const row = document.createElement("tr");
        row.className = "feature-row";
        row.setAttribute("data-feature", col.name);

        // Col 1: Checkbox to Toggle Active State
        const tdActive = document.createElement("td");
        tdActive.className = "col-active";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = true;
        checkbox.className = "custom-checkbox feature-active-toggle";
        tdActive.appendChild(checkbox);

        // Col 2: Feature Name
        const tdName = document.createElement("td");
        tdName.className = "col-name";
        tdName.textContent = col.name;

        // Col 3: Datatype badge
        const tdType = document.createElement("td");
        tdType.className = "col-type";
        const badge = document.createElement("span");
        badge.className = `feature-tag ${col.type === "numerical" ? "tag-num" : "tag-cat"}`;
        badge.textContent = col.type === "numerical" ? "Numeric" : "Categoric";
        tdType.appendChild(badge);

        // Col 4: Preprocessing Transform Selector
        const tdTransform = document.createElement("td");
        tdTransform.className = "col-transform";
        const select = document.createElement("select");
        select.className = "form-control feature-transform-select";
        select.style.padding = "0.3rem 0.5rem";
        select.style.fontSize = "0.75rem";

        if (col.type === "numerical") {
            // Numeric Transformation options
            const optNone = document.createElement("option");
            optNone.value = "none";
            optNone.textContent = "None (Raw)";
            const optStandard = document.createElement("option");
            optStandard.value = "standard";
            optStandard.textContent = "Standard Scaler";
            const optMinMax = document.createElement("option");
            optMinMax.value = "minmax";
            optMinMax.textContent = "MinMax Scaler";
            
            select.appendChild(optNone);
            select.appendChild(optStandard);
            select.appendChild(optMinMax);
            
            // Default recommend Standard scaling
            select.value = "standard";
        } else {
            // Categorical options
            const optNone = document.createElement("option");
            optNone.value = "none";
            optNone.textContent = "None (Raw String)";
            const optOneHot = document.createElement("option");
            optOneHot.value = "onehot";
            optOneHot.textContent = "One-Hot Encode";
            const optOrdinal = document.createElement("option");
            optOrdinal.value = "ordinal";
            optOrdinal.textContent = "Ordinal Encode";
            
            select.appendChild(optNone);
            select.appendChild(optOneHot);
            select.appendChild(optOrdinal);
            
            // Default recommend OneHot
            select.value = "onehot";
        }
        
        tdTransform.appendChild(select);

        // Enable/Disable select based on checkbox
        checkbox.addEventListener("change", (e) => {
            select.disabled = !e.target.checked;
            row.style.opacity = e.target.checked ? "1" : "0.5";
        });

        row.appendChild(tdActive);
        row.appendChild(tdName);
        row.appendChild(tdType);
        row.appendChild(tdTransform);

        tableBody.appendChild(row);
    });
}

// Triggered when an algorithm is chosen in dropdown
function handleModelSelection(modelKey) {
    appState.activeModel = modelKey;
    logConsole(`Model selection updated: ${modelKey}`, "system");

    // Dynamic hyperparameter form generation
    renderHyperparameters(modelKey);
    renderExperiences(modelKey);

    // Show hyperparams panel
    document.getElementById("hyperparams-section").classList.remove("hidden");

    // Enable Run Trigger
    document.getElementById("btn-run-evaluation").classList.remove("disabled-state");
}

// Dynamic rendering of experience presets
function renderExperiences(modelKey) {
    const experienceSelect = document.getElementById("experience-select");
    if (!experienceSelect) return;

    experienceSelect.replaceChildren();

    const schema = appState.modelsSchema[modelKey];
    if (!schema || !schema.experiences) {
        document.getElementById("experience-group").classList.add("hidden");
        return;
    }

    document.getElementById("experience-group").classList.remove("hidden");
    const descElement = document.getElementById("experience-description");
    if (descElement) {
        descElement.classList.add("hidden");
        descElement.textContent = "";
    }
    
    // Add default unselected option
    const defaultOpt = document.createElement("option");
    defaultOpt.value = "";
    defaultOpt.disabled = true;
    defaultOpt.selected = true;
    defaultOpt.textContent = "Select an experience preset...";
    experienceSelect.appendChild(defaultOpt);

    Object.entries(schema.experiences).forEach(([expKey, expData]) => {
        const opt = document.createElement("option");
        opt.value = expKey;
        opt.textContent = expData.name;
        experienceSelect.appendChild(opt);
    });

    // Remove old listeners to avoid duplicates by replacing the node
    const newSelect = experienceSelect.cloneNode(true);
    experienceSelect.parentNode.replaceChild(newSelect, experienceSelect);
    
    newSelect.addEventListener("change", (e) => {
        const expKey = e.target.value;
        const expData = schema.experiences[expKey];
        if (!expData) return;
        
        logConsole(`Applied experience preset: ${expData.name}`, "info");

        const descElement = document.getElementById("experience-description");
        if (descElement) {
            // Extract the human-readable changes
            const changedParams = [];
            Object.entries(expData.params).forEach(([paramName, paramValue]) => {
                const paramSchema = schema.params.find(p => p.name === paramName);
                if (paramSchema) {
                    changedParams.push(`${paramSchema.label} to ${paramValue}`);
                }
            });
            
            const changesText = changedParams.length > 0 
                ? ` Adjusts settings: ${changedParams.join(", ")}.` 
                : "";

            descElement.textContent = (expData.description || "") + changesText;
            descElement.classList.remove("hidden");
            // Add a style to make it look decent
            descElement.style.paddingTop = "0.5rem";
            descElement.style.fontSize = "0.75rem";
            descElement.style.color = "var(--text-muted)";
            descElement.style.fontStyle = "italic";
        }

        // Update each hyperparameter input
        Object.entries(expData.params).forEach(([paramName, paramValue]) => {
            const input = document.getElementById(`param-${paramName}`);
            if (input) {
                if (input.type === "checkbox") {
                    input.checked = paramValue;
                } else {
                    input.value = paramValue;
                    // Trigger input event to update display labels (if any)
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }
        });
    });
}

// Dynamic construction of hyperparameter controls (sliders, drop downs, switches)
function renderHyperparameters(modelKey) {
    const container = document.getElementById("hyperparams-container");
    if (!container) return;

    container.replaceChildren();

    const schema = appState.modelsSchema[modelKey];
    if (!schema || !schema.params) return;

    schema.params.forEach(param => {
        const group = document.createElement("div");
        group.className = "form-group";
        group.style.marginBottom = "0.6rem";

        // Label
        const label = document.createElement("label");
        label.className = "field-label";
        label.style.fontSize = "0.75rem";
        label.textContent = param.label;

        // Add dynamic tooltip if description is configured in backend schema
        if (param.description) {
            const tooltip = document.createElement("span");
            tooltip.className = "tooltip-container tooltip-bottom";

            const icon = document.createElement("span");
            icon.className = "tooltip-icon";
            icon.textContent = "?";

            const text = document.createElement("span");
            text.className = "tooltip-text";
            text.textContent = param.description;

            tooltip.appendChild(icon);
            tooltip.appendChild(text);
            label.appendChild(tooltip);
        }

        group.appendChild(label);

        // Value Input
        if (param.type === "int" || param.type === "float" || param.type === "int_or_none") {
            const inputWrapper = document.createElement("div");
            inputWrapper.style.display = "flex";
            inputWrapper.style.alignItems = "center";
            inputWrapper.style.gap = "0.6rem";

            const slider = document.createElement("input");
            slider.type = "range";
            slider.className = "range-slider hyperparam-input";
            slider.style.flex = "1";
            slider.id = `param-${param.name}`;
            slider.min = param.min;
            slider.max = param.max;
            slider.step = param.step;
            slider.value = param.default;

            const valDisplay = document.createElement("span");
            valDisplay.style.fontSize = "0.8rem";
            valDisplay.style.fontWeight = "700";
            valDisplay.style.minWidth = "30px";
            valDisplay.style.color = "var(--primary)";
            valDisplay.textContent = param.default;

            slider.addEventListener("input", (e) => {
                valDisplay.textContent = e.target.value;
            });

            inputWrapper.appendChild(slider);
            inputWrapper.appendChild(valDisplay);
            group.appendChild(inputWrapper);

        } else if (param.type === "choice") {
            const select = document.createElement("select");
            select.className = "form-control hyperparam-input";
            select.id = `param-${param.name}`;
            select.style.padding = "0.4rem 0.6rem";
            select.style.fontSize = "0.8rem";

            param.choices.forEach(choice => {
                const opt = document.createElement("option");
                opt.value = choice;
                opt.textContent = choice;
                if (choice === param.default) opt.selected = true;
                select.appendChild(opt);
            });
            group.appendChild(select);

        } else if (param.type === "bool") {
            const select = document.createElement("select");
            select.className = "form-control hyperparam-input";
            select.id = `param-${param.name}`;
            select.style.padding = "0.4rem 0.6rem";
            select.style.fontSize = "0.8rem";

            const optTrue = document.createElement("option");
            optTrue.value = "True";
            optTrue.textContent = "Yes";
            if (param.default) optTrue.selected = true;

            const optFalse = document.createElement("option");
            optFalse.value = "False";
            optFalse.textContent = "No";
            if (!param.default) optFalse.selected = true;

            select.appendChild(optTrue);
            select.appendChild(optFalse);
            group.appendChild(select);
        }

        container.appendChild(group);
    });
}

// Gathers state and calls model training & evaluation API
function evaluateModel() {
    if (!appState.datasetToken || !appState.activeModel || !appState.targetColumn) {
        showToast("Invalid configuration status. Please reload dataset and configurations.");
        return;
    }

    logConsole("Initializing model training pipeline...", "system");
    updateServerStatus("Fitting estimator...", true);

    // Gathers dynamic Hyperparameter values
    const hyperparams = {};
    const schema = appState.modelsSchema[appState.activeModel];
    if (schema && schema.params) {
        schema.params.forEach(param => {
            const input = document.getElementById(`param-${param.name}`);
            if (input) {
                let val = input.value;
                // Parse boolean values
                if (val === "True") val = true;
                if (val === "False") val = false;
                hyperparams[param.name] = val;
            }
        });
    }

    // Gathers active features & transforms
    const features = [];
    const rows = document.querySelectorAll(".feature-row");
    rows.forEach(row => {
        const name = row.getAttribute("data-feature");
        const active = row.querySelector(".feature-active-toggle").checked;
        const transform = row.querySelector(".feature-transform-select").value;
        features.push({ name, active, transform });
    });

    const activeFeaturesCount = features.filter(f => f.active).length;
    if (activeFeaturesCount === 0) {
        updateServerStatus("Idle");
        logConsole("Evaluation halted: Select at least one active feature.", "error");
        showToast("Training aborted: No active features selected.");
        return;
    }

    const trainSplit = document.getElementById("split-slider").value / 100.0;

    // Build Payload
    const payload = {
        dataset_token: appState.datasetToken,
        model_type: appState.activeModel,
        hyperparameters: hyperparams,
        features: features,
        target_column: appState.targetColumn,
        train_split: trainSplit
    };

    logConsole(`Training ${schema.name} on ${activeFeaturesCount} features...`, "info");
    logConsole("Preprocessing data matrices (Imputing + Scaling/Encoding)...", "info");

    // POST request to evaluate model
    fetch("/api/evaluate", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: jsonSafeStringify(payload)
    })
    .then(res => {
        if (!res.ok) {
            return res.json().then(json => { throw new Error(json.error || "Evaluation failed"); });
        }
        return res.json();
    })
    .then(data => {
        updateServerStatus("Idle");
        logConsole("Fitting completed successfully.", "system");
        logConsole("Evaluating test predictions against validation labels...", "info");

        // Display results
        displayEvaluationResults(data);
    })
    .catch(err => {
        updateServerStatus("Training failed", false, true);
        logConsole(`Training failed: ${err.message}`, "error");
        showToast(err.message || "An error occurred during training. Verify column configurations.");
    });
}

// Display metrics and draw Chart.js results
function displayEvaluationResults(data) {
    // Show sections
    document.getElementById("metrics-container").classList.remove("hidden");
    document.getElementById("charts-container").classList.remove("hidden");

    // 1. Populate Metrics Cards Grid
    const metricsGrid = document.getElementById("metrics-grid");
    metricsGrid.replaceChildren();

    const isClassification = appState.modelsSchema[appState.activeModel].type === "classification";

    Object.entries(data.metrics).forEach(([key, val]) => {
        const card = document.createElement("div");
        card.className = "metric-card";
        if (!isClassification) card.classList.add("reg-metric");

        const title = document.createElement("div");
        title.className = "metric-title";
        title.textContent = key;

        const value = document.createElement("div");
        value.className = "metric-value";
        value.textContent = val;

        card.appendChild(title);
        card.appendChild(value);
        metricsGrid.appendChild(card);
    });

    logConsole(`Evaluation complete. Key Metric: ${Object.keys(data.metrics)[0]} = ${Object.values(data.metrics)[0]}`, "system");

    // 2. Render Charts
    renderPrimaryVisualization(data.visualization);
    renderSecondaryVisualization(data.visualization);
    
    // Confusion Matrix Tab configuration
    const cmTab = document.getElementById("tab-tertiary");
    if (isClassification && data.visualization.confusion_matrix) {
        cmTab.classList.remove("hidden");
        renderConfusionMatrix(data.visualization.confusion_matrix);
    } else {
        cmTab.classList.add("hidden");
        // Ensure tertiary wrapper wrapper is hidden if it was active
        if (cmTab.classList.contains("active")) {
            document.getElementById("tab-primary").click();
        }
    }
}

// Draw the model fit visualization (ROC for classifier, Scatter for regressor)
function renderPrimaryVisualization(vis) {
    const ctx = document.getElementById("chart-primary");
    if (!ctx) return;

    if (appState.primaryChart) {
        appState.primaryChart.destroy();
    }

    const modelType = appState.modelsSchema[appState.activeModel].type;

    if (modelType === "classification") {
        // Render ROC Curve
        const roc = vis.roc_curve;
        if (!roc) {
            logConsole("ROC curve unavailable (needs probabilities output or binary task).", "system");
            // Render text alternative inside canvas parent
            drawFallbackChart(ctx, "ROC Curve Unavailable for Multi-class / SVM without probabilities");
            return;
        }

        // Map FPR/TPR pairs
        const rocData = roc.fpr.map((fprVal, idx) => ({ x: fprVal, y: roc.tpr[idx] }));

        appState.primaryChart = new Chart(ctx, {
            type: "line",
            data: {
                datasets: [
                    {
                        label: `Model ROC (AUC = ${roc.auc})`,
                        data: rocData,
                        borderColor: "#3b82f6",
                        backgroundColor: "rgba(59, 130, 246, 0.1)",
                        fill: true,
                        tension: 0.1,
                        borderWidth: 2,
                        pointRadius: 1
                    },
                    {
                        label: "Random Classifier",
                        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                        borderColor: "rgba(255, 255, 255, 0.2)",
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: "linear",
                        title: { display: true, text: "False Positive Rate (FPR)", color: "#9ca3af" },
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#9ca3af" },
                        min: 0,
                        max: 1
                    },
                    y: {
                        type: "linear",
                        title: { display: true, text: "True Positive Rate (TPR)", color: "#9ca3af" },
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#9ca3af" },
                        min: 0,
                        max: 1
                    }
                },
                plugins: {
                    legend: { labels: { color: "#f3f4f6" } }
                }
            }
        });
    } else if (modelType === "regression") {
        // Render Regression Scatter Plot (Predictions vs Actuals)
        const reg = vis.regression_results;
        if (!reg) return;

        const scatterData = reg.actuals.map((act, idx) => ({ x: act, y: reg.predictions[idx] }));
        
        // Find bounds for line of equality
        const minVal = Math.min(...reg.actuals, ...reg.predictions);
        const maxVal = Math.max(...reg.actuals, ...reg.predictions);

        appState.primaryChart = new Chart(ctx, {
            type: "scatter",
            data: {
                datasets: [
                    {
                        label: "Predictions vs. Actuals",
                        data: scatterData,
                        backgroundColor: "rgba(6, 182, 212, 0.6)",
                        borderColor: "#06b6d4",
                        borderWidth: 1,
                        pointRadius: 4
                    },
                    {
                        label: "Ideal Fit (y = x)",
                        data: [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }],
                        borderColor: "rgba(16, 185, 129, 0.6)",
                        borderDash: [4, 4],
                        type: "line",
                        fill: false,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: "Actual Values", color: "#9ca3af" },
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#9ca3af" }
                    },
                    y: {
                        title: { display: true, text: "Predicted Values", color: "#9ca3af" },
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#9ca3af" }
                    }
                },
                plugins: {
                    legend: { labels: { color: "#f3f4f6" } }
                }
            }
        });
    } else if (modelType === "cv" || modelType === "llm") {
        // Render Training Curve
        const tc = vis.training_curve;
        if (!tc) return;

        appState.primaryChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: tc.epochs,
                datasets: [
                    {
                        label: "Training Loss",
                        data: tc.loss,
                        borderColor: "#8b5cf6",
                        backgroundColor: "rgba(139, 92, 246, 0.1)",
                        fill: true,
                        tension: 0.3,
                        borderWidth: 2,
                        pointRadius: 3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: "Epoch", color: "#9ca3af" },
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#9ca3af" }
                    },
                    y: {
                        title: { display: true, text: "Loss", color: "#9ca3af" },
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#9ca3af" }
                    }
                },
                plugins: {
                    legend: { labels: { color: "#f3f4f6" } }
                }
            }
        });
    }
}

// Draw Feature Importance chart
function renderSecondaryVisualization(vis) {
    const ctx = document.getElementById("chart-secondary");
    if (!ctx) return;

    if (appState.secondaryChart) {
        appState.secondaryChart.destroy();
    }

    const fi = vis.feature_importance;
    if (!fi || !fi.labels || fi.labels.length === 0) {
        drawFallbackChart(ctx, "Feature Importance unavailable for this model structure", true);
        return;
    }

    appState.secondaryChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: fi.labels,
            datasets: [{
                label: "Relative Coefficient Magnitude / Feature Importance",
                data: fi.values,
                backgroundColor: "rgba(139, 92, 246, 0.65)",
                borderColor: "#8b5cf6",
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: "Importance / Absolute Weight", color: "#9ca3af" },
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#9ca3af" }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: "#f3f4f6", font: { size: 10 } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// Dynamic construction of Confusion Matrix grid
function renderConfusionMatrix(cm) {
    const container = document.getElementById("confusion-matrix-grid");
    if (!container) return;

    container.replaceChildren();

    const classes = cm.classes;
    const matrix = cm.matrix;
    const numClasses = classes.length;

    // Header label
    const header = document.createElement("div");
    header.className = "cm-header-label";
    header.textContent = "Predicted Class Labels";
    container.appendChild(header);

    // Grid row header (Top class names)
    const topRow = document.createElement("div");
    topRow.className = "cm-row";
    // Corner block spacing
    const cornerSpacer = document.createElement("div");
    cornerSpacer.className = "cm-label-left";
    topRow.appendChild(cornerSpacer);

    classes.forEach(cls => {
        const topLabel = document.createElement("div");
        topLabel.className = "cm-cell cm-label-top";
        topLabel.style.border = "none";
        topLabel.style.background = "none";
        topLabel.textContent = cls;
        topRow.appendChild(topLabel);
    });
    container.appendChild(topRow);

    // Dynamic rendering of rows containing left class labels and values
    // Calculates total samples per actual label for percentages
    for (let r = 0; r < numClasses; r++) {
        const row = document.createElement("div");
        row.className = "cm-row";

        // Left Label (Actual class name)
        const leftLabel = document.createElement("div");
        leftLabel.className = "cm-label-left";
        leftLabel.textContent = classes[r];
        leftLabel.title = classes[r];
        row.appendChild(leftLabel);

        const rowTotal = matrix[r].reduce((a, b) => a + b, 0);

        for (let c = 0; c < numClasses; c++) {
            const cellVal = matrix[r][c];
            const pct = rowTotal > 0 ? ((cellVal / rowTotal) * 100).toFixed(1) : "0.0";

            const cell = document.createElement("div");
            cell.className = "cm-cell";
            
            // HSL Heatmap styling: match color density to prediction accuracy (diagonals = true class prediction)
            const isCorrect = r === c;
            const intensity = rowTotal > 0 ? (cellVal / rowTotal) : 0;
            if (isCorrect) {
                // Success Emerald Green tint
                cell.style.background = `hsla(160, 84%, 39%, ${0.1 + intensity * 0.7})`;
                cell.style.borderColor = `hsla(160, 84%, 39%, ${0.3 + intensity * 0.4})`;
            } else if (cellVal > 0) {
                // Error Red tint
                cell.style.background = `hsla(350, 89%, 60%, ${0.05 + intensity * 0.5})`;
                cell.style.borderColor = `hsla(350, 89%, 60%, ${0.1 + intensity * 0.3})`;
            }

            const valSpan = document.createElement("span");
            valSpan.className = "cm-cell-val";
            valSpan.textContent = cellVal;

            const pctSpan = document.createElement("span");
            pctSpan.className = "cm-cell-pct";
            pctSpan.textContent = `${pct}%`;

            cell.appendChild(valSpan);
            cell.appendChild(pctSpan);
            row.appendChild(cell);
        }
        container.appendChild(row);
    }
}

// Fallback visual in canvas wrapper if charts cannot render
function drawFallbackChart(canvasElement, message, isSecondary = false) {
    const parent = canvasElement.parentNode;
    if (!parent) return;

    // Destroy chart reference
    if (isSecondary && appState.secondaryChart) {
        appState.secondaryChart.destroy();
        appState.secondaryChart = null;
    } else if (!isSecondary && appState.primaryChart) {
        appState.primaryChart.destroy();
        appState.primaryChart = null;
    }

    // Canvas drawing workaround for plain text display
    const ctx = canvasElement.getContext("2d");
    if (!ctx) return;
    
    // Clear and draw text helper
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    ctx.fillStyle = "#9ca3af";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(message, canvasElement.width / 2, canvasElement.height / 2);
}

// Safe json stringifier to prevent injection
function jsonSafeStringify(obj) {
    return JSON.stringify(obj);
}
