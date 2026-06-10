import io
import uuid
import logging
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)

# Individual model classes
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Flask Application Setup
# TODO(security): Binding to 0.0.0.0 (all interfaces) to allow local network discoverability and management, overriding standard loopback restriction.
app = Flask(__name__)

# Size limit: 5MB maximum file upload to prevent DoS attacks.
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024

# Global in-memory cache to store loaded DataFrames.
# This prevents directory traversal vulnerabilities by avoiding writing/reading files from disk.
DATA_CACHE = {}

# ==============================================================================
# UI MAPPING REFERENCE:
# Below is a detailed directory of how UI elements trigger routes and parameters:
# 
# 1. UI Action: Select Sample Dataset -> main.js loads sample data via GET /api/sample-data
# 2. UI Action: File Upload -> main.js sends CSV multipart file to POST /api/analyze-csv
# 3. UI Action: Select Model Type -> main.js reads parameters from GET /api/models schema
# 4. UI Action: Click "Run Evaluation" -> main.js gathers:
#    - '#target-select' selection (y)
#    - '#split-slider' position (train/test split ratio)
#    - '#feature-table' checkboxes and transforms (list of features to include and their scalers)
#    - '#model-select' chosen model type
#    - Hyperparameter values from dynamic input boxes (e.g. '#param-n_estimators', etc.)
#    Then triggers POST /api/evaluate.
# ==============================================================================

MODELS_SCHEMA = {
    # CLASSIFICATION MODELS
    "random_forest_classifier": {
        "name": "Random Forest Classifier",
        "type": "classification",
        "params": [
            {"name": "n_estimators", "label": "Number of Trees", "type": "int", "default": 100, "min": 10, "max": 500, "step": 10, "description": "The number of trees in the forest. More trees increase accuracy and generalization but slow down training and prediction speed."},
            {"name": "max_depth", "label": "Max Depth", "type": "int_or_none", "default": 10, "min": 1, "max": 50, "step": 1, "description": "The maximum depth of the individual trees. Deeper trees capture complex patterns but run a high risk of overfitting."},
            {"name": "min_samples_split", "label": "Min Samples Split", "type": "int", "default": 2, "min": 2, "max": 20, "step": 1, "description": "The minimum number of samples required to split an internal node. Higher values act as regularization by preventing small, highly-specific leaves."}
        ]
    },
    "logistic_regression": {
        "name": "Logistic Regression",
        "type": "classification",
        "params": [
            {"name": "C", "label": "Regularization Strength (C)", "type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01, "description": "Inverse of regularization strength. Smaller values specify stronger regularization, which keeps weights small to prevent overfitting."},
            {"name": "penalty", "label": "Penalty", "type": "choice", "default": "l2", "choices": ["l2", "none"], "description": "The type of norm regularization applied. L2 shrinks weights, while 'none' applies standard maximum likelihood estimation."},
            {"name": "solver", "label": "Solver", "type": "choice", "default": "lbfgs", "choices": ["lbfgs", "liblinear", "saga"], "description": "Optimization algorithm used to solve for model weights. 'liblinear' is good for small data; 'saga' is faster for large datasets; 'lbfgs' is standard."}
        ]
    },
    "svc": {
        "name": "Support Vector Classifier (SVC)",
        "type": "classification",
        "params": [
            {"name": "C", "label": "Regularization Parameter (C)", "type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01, "description": "Regularization parameter. Margin trade-off: larger values try to classify all training points correctly (overfitting risk), smaller values allow a wider margin (generalizes better)."},
            {"name": "kernel", "label": "Kernel Type", "type": "choice", "default": "rbf", "choices": ["rbf", "linear", "poly", "sigmoid"], "description": "Determines how data is projected into high-dimensional space. 'linear' keeps it flat; 'rbf' maps non-linear circular shapes; 'poly' handles curves."},
            {"name": "gamma", "label": "Gamma scale", "type": "choice", "default": "scale", "choices": ["scale", "auto"], "description": "Kernel coefficient for non-linear kernels. 'scale' uses 1/(n_features * X.var()), adjusting dynamically; 'auto' uses 1/n_features."}
        ]
    },
    "decision_tree_classifier": {
        "name": "Decision Tree Classifier",
        "type": "classification",
        "params": [
            {"name": "criterion", "label": "Splitting Criterion", "type": "choice", "default": "gini", "choices": ["gini", "entropy", "log_loss"], "description": "Mathematical function measuring split quality. 'gini' measures general impurity; 'entropy' and 'log_loss' calculate information gain."},
            {"name": "max_depth", "label": "Max Depth", "type": "int_or_none", "default": 10, "min": 1, "max": 50, "step": 1, "description": "The maximum depth of the decision tree. Deep trees memorize training data (overfit); shallow trees are fast and robust."}
        ]
    },
    "gradient_boosting_classifier": {
        "name": "Gradient Boosting Classifier",
        "type": "classification",
        "params": [
            {"name": "n_estimators", "label": "Number of Estimators", "type": "int", "default": 100, "min": 10, "max": 300, "step": 10, "description": "Number of boosting stages (weak trees) to perform. Gradient boosting is robust to overfitting but too many trees will eventually overfit."},
            {"name": "learning_rate", "label": "Learning Rate", "type": "float", "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01, "description": "Shrinkage factor of each weak tree's contribution. Smaller values require higher estimators for same fit, but lead to better generalization."},
            {"name": "max_depth", "label": "Max Depth", "type": "int", "default": 3, "min": 1, "max": 15, "step": 1, "description": "Maximum depth of the individual regression trees. Limits the degree of interaction between features."}
        ]
    },

    # REGRESSION MODELS
    "linear_regression": {
        "name": "Linear Regression",
        "type": "regression",
        "params": [
            {"name": "fit_intercept", "label": "Fit Intercept", "type": "bool", "default": True, "description": "Whether to calculate the constant intercept offset. Uncheck if your data is already centered around zero."}
        ]
    },
    "ridge_regression": {
        "name": "Ridge Regression",
        "type": "regression",
        "params": [
            {"name": "alpha", "label": "Regularization Strength (Alpha)", "type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01, "description": "L2 regularization strength. Larger values force model weights closer to zero, reducing multicollinearity impact and overfitting."}
        ]
    },
    "random_forest_regressor": {
        "name": "Random Forest Regressor",
        "type": "regression",
        "params": [
            {"name": "n_estimators", "label": "Number of Trees", "type": "int", "default": 100, "min": 10, "max": 500, "step": 10, "description": "Number of trees in the forest. A higher count yields smoother predictions but uses more RAM and compute time."},
            {"name": "max_depth", "label": "Max Depth", "type": "int_or_none", "default": 10, "min": 1, "max": 50, "step": 1, "description": "Maximum depth of the trees. Deeper trees capture fine details but run a higher risk of model overfitting."},
            {"name": "min_samples_split", "label": "Min Samples Split", "type": "int", "default": 2, "min": 2, "max": 20, "step": 1, "description": "Minimum samples needed to split an internal node. Acts as regularizer: higher values prevent modeling small-sample anomalies."}
        ]
    },
    "svr": {
        "name": "Support Vector Regressor (SVR)",
        "type": "regression",
        "params": [
            {"name": "C", "label": "Regularization Parameter (C)", "type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01, "description": "Regularization parameter. Trade-off: larger values prioritize fitting all training samples closely (overfitting danger), smaller values allow softer boundaries."},
            {"name": "kernel", "label": "Kernel Type", "type": "choice", "default": "rbf", "choices": ["rbf", "linear", "poly", "sigmoid"], "description": "Determines the high-dimensional projection algorithm. 'rbf' handles non-linear distributions; 'linear' is suitable for simple flat trends."},
            {"name": "gamma", "label": "Gamma scale", "type": "choice", "default": "scale", "choices": ["scale", "auto"], "description": "Kernel scale parameter. Controls the range of influence of a single training point. 'scale' uses 1/(n_features * X.var()) and adapts to variance."}
        ]
    },
    "gradient_boosting_regressor": {
        "name": "Gradient Boosting Regressor",
        "type": "regression",
        "params": [
            {"name": "n_estimators", "label": "Number of Estimators", "type": "int", "default": 100, "min": 10, "max": 300, "step": 10, "description": "Number of sequential trees to fit. Higher numbers improve fit quality but extend execution times."},
            {"name": "learning_rate", "label": "Learning Rate", "type": "float", "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01, "description": "Shrinkage factor scaling each tree's output. Best practice is to lower learning rate and raise estimators to reduce overfitting."},
            {"name": "max_depth", "label": "Max Depth", "type": "int", "default": 3, "min": 1, "max": 15, "step": 1, "description": "Maximum depth of the individual regression trees. Restricts the number of interaction terms modeled."}
        ]
    }
}


@app.route("/")
def index():
    """Serves the primary Single Page Application interface."""
    return render_template("index.html")


@app.route("/api/models", methods=["GET"])
def get_models():
    """
    Returns the schema defining supported models and their hyperparameters.
    Used by the UI to dynamically build configuration panels.
    """
    return jsonify(MODELS_SCHEMA)


@app.route("/api/sample-data", methods=["GET"])
def load_sample():
    """
    Generates synthetic datasets for demonstration.
    Avoids loading files from disk, keeping execution fully local and self-contained.
    Parameters:
      - type: 'housing' (regression) or 'churn' (classification)
    """
    dataset_type = request.args.get("type", "housing")
    dataset_token = f"sample_{dataset_type}"
    np.random.seed(42)

    if dataset_type == "housing":
        # Generate regression dataset: House Prices
        n_samples = 500
        sq_feet = np.random.normal(1800, 500, n_samples).clip(500, 5000).astype(int)
        bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
        bathrooms = (bedrooms * np.random.choice([0.75, 1.0, 1.25], n_samples)).round(1).clip(1, 4)
        age = np.random.randint(0, 80, n_samples)
        neighborhood = np.random.choice(["Urban", "Suburban", "Rural"], n_samples, p=[0.4, 0.4, 0.2])
        has_pool = np.random.choice(["Yes", "No"], n_samples, p=[0.15, 0.85])

        # Generate target value based on features + noise
        neighborhood_mult = {"Urban": 1.2, "Suburban": 1.0, "Rural": 0.8}
        base_price = sq_feet * 150 + bedrooms * 15000 + bathrooms * 10000 - age * 800
        pool_add = (has_pool == "Yes") * 25000
        neighborhood_factor = np.array([neighborhood_mult[n] for n in neighborhood])
        price = (base_price + pool_add) * neighborhood_factor + np.random.normal(0, 15000, n_samples)
        price = price.round(-3).clip(50000).astype(int)

        df = pd.DataFrame({
            "SquareFeet": sq_feet,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Age": age,
            "Neighborhood": neighborhood,
            "HasPool": has_pool,
            "Price": price
        })

    elif dataset_type == "churn":
        # Generate classification dataset: Customer Churn
        n_samples = 600
        tenure = np.random.randint(1, 72, n_samples)
        contract = np.random.choice(["Month-to-month", "One year", "Two year"], n_samples, p=[0.5, 0.25, 0.25])
        monthly_charges = np.random.normal(65, 30, n_samples).clip(19, 120).round(2)
        total_charges = (tenure * monthly_charges + np.random.normal(0, 50, n_samples)).clip(19).round(2)
        tech_support = np.random.choice(["Yes", "No"], n_samples, p=[0.3, 0.7])
        paperless = np.random.choice(["Yes", "No"], n_samples, p=[0.6, 0.4])

        # Latent churn probability
        logit = -0.5 - 0.05 * tenure + 0.015 * monthly_charges
        logit += np.where(contract == "Month-to-month", 0.8, -0.6)
        logit += np.where(tech_support == "No", 0.4, -0.4)
        logit += np.where(paperless == "Yes", 0.2, -0.2)
        
        prob = 1 / (1 + np.exp(-logit))
        churned = np.random.binomial(1, prob, n_samples)
        churn_labels = np.where(churned == 1, "Yes", "No")

        df = pd.DataFrame({
            "TenureMonths": tenure,
            "Contract": contract,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "TechSupport": tech_support,
            "PaperlessBilling": paperless,
            "Churned": churn_labels
        })
    else:
        return jsonify({"error": "Unknown sample type"}), 400

    # Cache DataFrame in memory
    DATA_CACHE[dataset_token] = df

    # Prepare metadata to send to the UI
    columns_info = []
    for col in df.columns:
        dtype = "numerical" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
        columns_info.append({
            "name": col,
            "type": dtype,
            "unique_count": int(df[col].nunique()),
            "sample_values": [str(x) for x in df[col].head(3).tolist()]
        })

    return jsonify({
        "dataset_token": dataset_token,
        "columns": columns_info,
        "sample_rows": df.head(5).to_dict(orient="records")
    })


@app.route("/api/analyze-csv", methods=["POST"])
def analyze_csv():
    """
    Accepts an uploaded CSV file, parses it entirely in-memory,
    caches the dataset, and returns column specifications to populate the UI.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded in query request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename provided"}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Unsupported file format. Please upload a CSV file."}), 400

    try:
        # Check file size before loading fully into memory
        file.seek(0, io.SEEK_END)
        size = file.tell()
        file.seek(0)

        if size > MAX_FILE_SIZE_BYTES:
            return jsonify({"error": "File size exceeds the 5MB limit. Please upload a smaller file."}), 400

        # Read CSV directly in memory
        content = file.read().decode("utf-8", errors="ignore")
        df = pd.read_csv(io.StringIO(content))

        if df.empty:
            return jsonify({"error": "The uploaded CSV file is empty."}), 400

        dataset_token = str(uuid.uuid4())
        DATA_CACHE[dataset_token] = df

        columns_info = []
        for col in df.columns:
            dtype = "numerical" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
            columns_info.append({
                "name": col,
                "type": dtype,
                "unique_count": int(df[col].nunique()),
                "sample_values": [str(x) for x in df[col].head(3).tolist()]
            })

        return jsonify({
            "dataset_token": dataset_token,
            "columns": columns_info,
            "sample_rows": df.head(5).to_dict(orient="records")
        })

    except Exception as e:
        logging.error(f"Error parsing uploaded file: {str(e)}")
        # Provide generic error for client while logging detail
        return jsonify({"error": "Failed to parse the CSV file. Ensure it is correctly formatted."}), 500


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """
    Processes model training and evaluation using parameters sent from UI.
    Extracts features, runs pipeline, scores, and outputs visualizations.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No parameters provided"}), 400

    dataset_token = data.get("dataset_token")
    model_key = data.get("model_type")
    hyperparams = data.get("hyperparameters", {})
    features_config = data.get("features", [])
    target_col = data.get("target_column")
    train_split = float(data.get("train_split", 0.8))

    # Input validation
    if not dataset_token or not model_key or not target_col or not features_config:
        return jsonify({"error": "Missing required configuration fields"}), 400

    if dataset_token not in DATA_CACHE:
        # Check if it is a sample key
        if dataset_token.startswith("sample_"):
            # Recreate sample
            type_name = dataset_token.replace("sample_", "")
            # Calling helper internally to load and cache
            with app.test_client() as client:
                res = client.get(f"/api/sample-data?type={type_name}")
                if res.status_code != 200:
                    return jsonify({"error": "Failed to load cached sample dataset"}), 400
        else:
            return jsonify({"error": "Dataset session expired. Please reload the dataset."}), 400

    df = DATA_CACHE[dataset_token]

    if target_col not in df.columns:
        return jsonify({"error": f"Target column '{target_col}' not found in dataset."}), 400

    # Build active features lists and transformations
    active_features = []
    num_transforms = {}
    cat_transforms = {}

    for fc in features_config:
        name = fc.get("name")
        active = fc.get("active", True)
        transform = fc.get("transform", "none")
        
        if not active:
            continue
            
        if name not in df.columns:
            return jsonify({"error": f"Feature column '{name}' not found in dataset."}), 400
            
        if name == target_col:
            continue  # Prevent leak
            
        active_features.append(name)
        
        # Classify transformations
        is_numeric = pd.api.types.is_numeric_dtype(df[name])
        if is_numeric:
            num_transforms[name] = transform
        else:
            cat_transforms[name] = transform

    if not active_features:
        return jsonify({"error": "Please select at least one active feature for training."}), 400

    # Construct X and y
    # Drop rows where target or selected features are missing for training stability
    clean_df = df[[target_col] + active_features].dropna(subset=[target_col])

    X = clean_df[active_features]
    y = clean_df[target_col]

    if len(X) < 10:
        return jsonify({"error": "Insufficient data samples remaining after removing missing values."}), 400

    # 1. Resolve model config and construct estimator
    model_spec = MODELS_SCHEMA.get(model_key)
    if not model_spec:
        return jsonify({"error": "Invalid model type selected."}), 400

    # Convert parameter values to proper types
    parsed_params = {}
    for param_def in model_spec["params"]:
        p_name = param_def["name"]
        val = hyperparams.get(p_name, param_def["default"])
        p_type = param_def["type"]

        if val is None or val == "None":
            parsed_params[p_name] = None
        elif p_type == "int":
            parsed_params[p_name] = int(val)
        elif p_type == "int_or_none":
            parsed_params[p_name] = int(val) if val is not None else None
        elif p_type == "float":
            parsed_params[p_name] = float(val)
        elif p_type == "bool":
            parsed_params[p_name] = bool(val)
        else:
            parsed_params[p_name] = str(val)

    # Instantiate the estimator
    is_classification = model_spec["type"] == "classification"
    try:
        estimator = None
        if model_key == "random_forest_classifier":
            estimator = RandomForestClassifier(random_state=42, **parsed_params)
        elif model_key == "logistic_regression":
            estimator = LogisticRegression(random_state=42, max_iter=2000, **parsed_params)
        elif model_key == "svc":
            estimator = SVC(probability=True, random_state=42, **parsed_params)
        elif model_key == "decision_tree_classifier":
            estimator = DecisionTreeClassifier(random_state=42, **parsed_params)
        elif model_key == "gradient_boosting_classifier":
            estimator = GradientBoostingClassifier(random_state=42, **parsed_params)
        elif model_key == "linear_regression":
            estimator = LinearRegression(**parsed_params)
        elif model_key == "ridge_regression":
            estimator = Ridge(**parsed_params)
        elif model_key == "random_forest_regressor":
            estimator = RandomForestRegressor(random_state=42, **parsed_params)
        elif model_key == "svr":
            estimator = SVR(**parsed_params)
        elif model_key == "gradient_boosting_regressor":
            estimator = GradientBoostingRegressor(random_state=42, **parsed_params)
    except Exception as e:
        logging.error(f"Error instantiating model {model_key}: {str(e)}")
        return jsonify({"error": f"Failed to configure model: {str(e)}"}), 400

    # 2. Build preprocessing transformers dynamically based on user feature configs
    numeric_features = [f for f in active_features if f in num_transforms]
    categorical_features = [f for f in active_features if f in cat_transforms]

    # Sub-transformers lists
    transformers = []

    # Numeric pipeline: Impute + optional Scale
    for nf in numeric_features:
        t_type = num_transforms[nf]
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if t_type == "standard":
            steps.append(("scaler", StandardScaler()))
        elif t_type == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        transformers.append((f"num_{nf}", Pipeline(steps), [nf]))

    # Categorical pipeline: Impute + optional Encode
    for cf in categorical_features:
        t_type = cat_transforms[cf]
        steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
        if t_type == "onehot":
            steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        elif t_type == "ordinal":
            steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
        transformers.append((f"cat_{cf}", Pipeline(steps), [cf]))

    preprocessor = ColumnTransformer(transformers=transformers)

    # Full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    # 3. Train/Test Split
    test_size = 1.0 - train_split
    # Target value encoding if target is categorical and model is classifier
    y_encoded = y.copy()
    target_mapping = None

    if is_classification:
        # Map target classes to integers if string
        if not pd.api.types.is_numeric_dtype(y):
            unique_classes = sorted(y.unique())
            target_mapping = {val: idx for idx, val in enumerate(unique_classes)}
            y_encoded = y.map(target_mapping)
        else:
            # Numeric labels
            unique_classes = sorted(y.unique())
            target_mapping = {val: val for val in unique_classes}
            
        # Stratified split if possible
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
        except Exception:
            # Fallback to normal split if stratify fails (e.g. single sample classes)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42
            )
    else:
        # Regression
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42
        )

    # 4. Train pipeline
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        return jsonify({"error": f"Error occurred during model fitting: {str(e)}"}), 500

    # 5. Predict & score
    try:
        y_pred = pipeline.predict(X_test)
        
        metrics = {}
        visualization_data = {}

        if is_classification:
            # Classification scoring
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            metrics = {
                "Accuracy": round(acc, 4),
                "F1-Score": round(f1, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4)
            }

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            # Match back to label names
            classes_list = [str(k) for k in (target_mapping.keys() if target_mapping else sorted(y_encoded.unique()))]
            visualization_data["confusion_matrix"] = {
                "classes": classes_list,
                "matrix": cm.tolist()
            }

            # ROC Curve (support for binary classification)
            unique_classes_count = len(np.unique(y_test))
            if unique_classes_count == 2 and hasattr(estimator, "predict_proba"):
                try:
                    # Probabilities of the positive class
                    y_probs = pipeline.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_probs)
                    roc_auc = auc(fpr, tpr)
                    visualization_data["roc_curve"] = {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "auc": round(roc_auc, 4)
                    }
                except Exception as ex:
                    logging.warning(f"Failed to calculate ROC curve: {str(ex)}")

        else:
            # Regression scoring
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            metrics = {
                "R-squared (R2)": round(r2, 4),
                "Root MSE (RMSE)": round(rmse, 4),
                "Mean Absolute Error (MAE)": round(mae, 4),
                "Mean Squared Error (MSE)": round(mse, 4)
            }

            # Sort predictions and actuals for alignment plotting
            test_indices = y_test.index
            visualization_data["regression_results"] = {
                "actuals": y_test.tolist(),
                "predictions": y_pred.tolist(),
                "index": list(range(len(y_test)))
            }

        # Feature Importance calculation (if available)
        try:
            # Retrieve processed features names out
            # We reconstruct the names since sklearn transforms generate new features
            feat_names = []
            for name, trans, cols in preprocessor.transformers_:
                if name == "remainder":
                    continue
                # If step encoder exists, fetch names, else use original column
                if "encoder" in trans.named_steps:
                    enc = trans.named_steps["encoder"]
                    if isinstance(enc, OneHotEncoder):
                        names_out = enc.get_feature_names_out(cols)
                        feat_names.extend(names_out)
                    else:
                        feat_names.extend(cols)
                else:
                    feat_names.extend(cols)

            importances = None
            model = pipeline.named_steps["model"]
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_.tolist()
            elif hasattr(model, "coef_"):
                # Coefficients might be multi-dimensional for classification, average magnitude
                coef = model.coef_
                if coef.ndim > 1:
                    importances = np.mean(np.abs(coef), axis=0).tolist()
                else:
                    importances = np.abs(coef).tolist()

            if importances is not None and len(importances) == len(feat_names):
                # Pair and sort
                sorted_feat_imp = sorted(
                    zip(feat_names, importances),
                    key=lambda x: x[1],
                    reverse=True
                )
                visualization_data["feature_importance"] = {
                    "labels": [item[0] for item in sorted_feat_imp[:15]],  # limit to top 15
                    "values": [round(item[1], 4) for item in sorted_feat_imp[:15]]
                }
        except Exception as fe:
            logging.warning(f"Could not compute feature importances: {str(fe)}")

        return jsonify({
            "status": "success",
            "model_type": model_spec["name"],
            "metrics": metrics,
            "visualization": visualization_data
        })

    except Exception as e:
        logging.error(f"Evaluation error: {str(e)}")
        return jsonify({"error": f"An error occurred during scoring: {str(e)}"}), 500


if __name__ == "__main__":
    # TODO(security): Binding to 0.0.0.0 (all interfaces) to enable discoverability and management by locally connected systems.
    # Ensure local network firewall restricts unauthorized incoming connections in staging/production deployments.
    logging.info("Starting discoverable model tuning application server on all interfaces...")
    app.run(host="0.0.0.0", port=5000, debug=True)
