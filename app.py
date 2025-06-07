from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer # Import MultiLabelBinarizer

# Initialize the Flask application
app = Flask(__name__)

# Define the directory where your first-stage .joblib models are stored (drug usage classification)
FIRST_STAGE_MODELS_DIR = 'saved_models_usage'

# Define the directory where your second-stage .joblib models are stored (Impulsive, SS regression)
SECOND_STAGE_MODELS_DIR = 'trained_models' # As per your training script's MODEL_SAVE_DIR

# --- NEW: Define the directory for your THIRD stage model (psychoactive classification) ---
THIRD_MODEL_DIR = 'third_model' # Ensure this directory exists and contains the models and preprocessors

# --- Mappings for Input Parameters (Keep these as they are used by the first/second stage) ---
AGE_MAPPING = {
    "18-24": -0.95197, "25-34": -0.07854, "35-44": 0.49788, "45-54": 1.09449,
    "55-64": 1.82213, "65+": 2.59171
}

GENDER_MAPPING = {
    "Female": 0.48246, "Male": -0.48246
}

EDUCATION_MAPPING = {
    "Left school before 16 years": -2.43591, "Left school at 16 years": -1.73790,
    "Left school at 17 years": -1.43719, "Left school at 18 years": -1.22751,
    "Some college or university, no certificate or degree": -0.61113,
    "Professional certificate/ diploma": -0.05921, "University degree": 0.45468,
    "Masters degree": 1.16365, "Doctorate degree": 1.98437
}

COUNTRY_MAPPING = {
    "Australia": -0.09765, "Canada": 0.24923, "New Zealand": -0.46841,
    "Other": -0.28519, "Republic of Ireland": 0.21128, "UK": 0.96082,
    "USA": -0.57009
}

ETHINICITY_MAPPING = {
    "Asian": -0.50212, "Black": -1.10702, "Mixed-Black/Asian": 1.90725,
    "Mixed-White/Asian": 0.12600, "Mixed-White/Black": -0.22166,
    "Other": 0.11440, "White": -0.31685
}

# Combine all mappings for easy access in the template
ALL_MAPPINGS = {
    'age': AGE_MAPPING,
    'gender': GENDER_MAPPING,
    'education': EDUCATION_MAPPING,
    'country': COUNTRY_MAPPING,
    'ethinicity': ETHINICITY_MAPPING
}

# Define the list of substances for consistent naming and extraction
substances_to_process_list = [
    'Alcohol', 'Amphete', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Coke',
    'Crack', 'Ecstacy', 'Heroin', 'Ketamine', 'LegalH', 'LSD', 'Meth',
    'Mushrooms', 'Nicotine', 'VSA'
]

# Define the feature names for the FIRST STAGE models in the exact order expected.
EXPECTED_FEATURE_ORDER_FIRST_STAGE = [
    'Nscore', 'Escore', 'Ascore', 'Oscore', 'Cscore', # Numerical personality features
    'Age', 'Gender', 'Education', 'Country', 'Ethinicity' # Categorical demographic features (after mapping)
]

# Define the feature names for the SECOND STAGE models (Impulsive, SS) in the exact order expected.
EXPECTED_FEATURE_ORDER_SECOND_STAGE = [
    'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', # Personality scores (numerical)
    'Education', 'Age', 'Ethinicity' # Demographic features (Age is numerical, Education/Ethinicity are STRINGS)
] + substances_to_process_list # Plus the 17 drug usage predictions (numerical)

# --- NEW: Define the BASE numerical features for the THIRD STAGE model ---
BASE_NUMERICAL_FEATURES_THIRD_STAGE = [
    'MolecularWeight', 'TPSA', 'RotatableBondCount', 'HydrogenBondDonors', 'HydrogenBondAcceptors'
]

# EXPECTED_FEATURE_ORDER_THIRD_STAGE will be dynamically generated after MLB is loaded.
# Initialize as empty; populated during asset loading.
EXPECTED_FEATURE_ORDER_THIRD_STAGE = []


# --- Model Loading Functions ---

def load_first_stage_models(models_dir):
    """
    Loads all .joblib model pipeline files for drug usage classification.
    """
    models = {}
    if not os.path.exists(models_dir):
        print(f"Error: First-stage models directory '{models_dir}' not found. Please create it and place your .joblib files inside.")
        return models

    for filename in os.listdir(models_dir):
        if filename.endswith('.joblib'):
            drug_name = None
            try:
                parts = filename.replace('.joblib', '').split('_')
                if len(parts) == 3 and parts[0] == 'model' and parts[1] == 'Drug':
                    drug_index = int(parts[2]) - 1
                    if 0 <= drug_index < len(substances_to_process_list):
                        drug_name = substances_to_process_list[drug_index]
                    else:
                        print(f"Warning: Drug index {drug_index + 1} from '{filename}' is out of bounds for substances list. Skipping.")
                        continue
                else:
                    print(f"Warning: First-stage filename '{filename}' does not match expected 'model_Drug_X.joblib' format. Skipping.")
                    continue
            except ValueError:
                print(f"Warning: Could not parse drug index from first-stage filename '{filename}'. Skipping.")
                continue
            except Exception as e:
                print(f"Error processing first-stage filename '{filename}': {e}. Skipping.")
                continue

            if drug_name:
                try:
                    filepath = os.path.join(models_dir, filename)
                    models[drug_name] = joblib.load(filepath)
                    print(f"Successfully loaded first-stage pipeline for: {drug_name} from {filepath}")
                except Exception as e:
                    print(f"Error loading first-stage model pipeline '{filename}' for drug '{drug_name}': {e}")
    
    print(f"Finished loading first-stage models. Total models loaded: {len(models)}")
    return models

def load_second_stage_models(models_dir):
    """
    Loads .joblib model pipeline files for Impulsive and SS prediction.
    """
    models = {}
    if not os.path.exists(models_dir):
        print(f"Error: Second-stage models directory '{models_dir}' not found. Please create it and place your .joblib files inside.")
        return models

    expected_targets = ['Impulsive', 'SS']
    for filename in os.listdir(models_dir):
        if filename.endswith('.joblib'):
            target_name = None
            try:
                # Expecting filenames like 'best_model_Impulsive_Random_Forest.joblib' or 'best_model_SS_XGBoost.joblib'
                parts = filename.replace('.joblib', '').split('_')
                if len(parts) >= 3 and parts[0] == 'best' and parts[1] == 'model':
                    target_name = parts[2] # e.g., 'Impulsive' or 'SS'
                    if target_name not in expected_targets:
                        print(f"Warning: Second-stage filename '{filename}' contains unexpected target '{target_name}'. Skipping.")
                        continue
                else:
                    print(f"Warning: Second-stage filename '{filename}' does not match expected 'best_model_TARGET_MODELNAME.joblib' format. Skipping.")
                    continue
            except Exception as e:
                print(f"Error processing second-stage filename '{filename}': {e}. Skipping.")
                continue

            if target_name:
                try:
                    filepath = os.path.join(models_dir, filename)
                    models[target_name] = joblib.load(filepath)
                    print(f"Successfully loaded second-stage pipeline for: {target_name} from {filepath}")
                except Exception as e:
                    print(f"Error loading second-stage model pipeline '{filename}' for target '{target_name}': {e}")
    
    print(f"Finished loading second-stage models. Total models loaded: {len(models)}")
    return models

# --- NEW: Function to load the psychoactive classification model and its preprocessors ---
def load_third_stage_assets(models_dir, model_filename='psychoactive_drug_model.joblib',
                           mlb_filename='mlb_psychoactive.joblib',
                           scaler_filename='scaler_psychoactive.joblib',
                           selector_filename='selector_psychoactive.joblib'):
    """
    Loads the specific psychoactive drug classification model and its associated preprocessors.
    """
    model_path = os.path.join(models_dir, model_filename)
    mlb_path = os.path.join(models_dir, mlb_filename)
    scaler_path = os.path.join(models_dir, scaler_filename)
    selector_path = os.path.join(models_dir, selector_filename)

    model = None
    mlb_obj = None
    scaler_obj = None
    selector_obj = None

    try:
        model = joblib.load(model_path)
        print(f"Successfully loaded third-stage model: {model_filename}")
    except Exception as e:
        print(f"Error loading third-stage model '{model_filename}': {e}")

    try:
        mlb_obj = joblib.load(mlb_path)
        print(f"Successfully loaded MultiLabelBinarizer: {mlb_filename}")
    except Exception as e:
        print(f"Error loading MultiLabelBinarizer '{mlb_filename}': {e}")

    try:
        scaler_obj = joblib.load(scaler_path)
        print(f"Successfully loaded StandardScaler: {scaler_filename}")
    except Exception as e:
        print(f"Error loading StandardScaler '{scaler_filename}': {e}")

    try:
        selector_obj = joblib.load(selector_path)
        print(f"Successfully loaded SelectKBest: {selector_filename}")
    except Exception as e:
        print(f"Error loading SelectKBest '{selector_filename}': {e}")

    return model, mlb_obj, scaler_obj, selector_obj


# Load all models and preprocessors once when the Flask application starts.
first_stage_models = load_first_stage_models(FIRST_STAGE_MODELS_DIR)
second_stage_models = load_second_stage_models(SECOND_STAGE_MODELS_DIR)
third_stage_model, third_stage_mlb, third_stage_scaler, third_stage_selector = load_third_stage_assets(THIRD_MODEL_DIR)

# Dynamically set EXPECTED_FEATURE_ORDER_THIRD_STAGE after mlb is loaded
if third_stage_mlb:
    # Combine base numerical features with the classes learned by MultiLabelBinarizer
    # This ensures the order matches the training script's X.columns.tolist()
    # Removed 'global' keyword as it's not needed in global scope for initial assignment/modification
    EXPECTED_FEATURE_ORDER_THIRD_STAGE = BASE_NUMERICAL_FEATURES_THIRD_STAGE + list(third_stage_mlb.classes_)
    print(f"Dynamically set EXPECTED_FEATURE_ORDER_THIRD_STAGE: {EXPECTED_FEATURE_ORDER_THIRD_STAGE}")
else:
    print("WARNING: third_stage_mlb not loaded. Cannot set EXPECTED_FEATURE_ORDER_THIRD_STAGE dynamically. Prediction for psychoactive drugs may fail.")
    # Fallback or error handling for the route itself will catch this if assets are missing.


# --- Flask Routes ---

@app.route('/')
def home():
    """
    Renders the main home page (home.html) for navigation.
    """
    return render_template('home.html')

@app.route('/prediction_app')
def prediction_app_page():
    """
    Renders the drug usage and personality prediction application page (prediction_app.html).
    """
    first_stage_loaded = bool(first_stage_models)
    second_stage_loaded = bool(second_stage_models)

    if not first_stage_loaded or not second_stage_loaded:
        error_msg = []
        if not first_stage_loaded:
            error_msg.append(f"No first-stage models found or loaded from '{FIRST_STAGE_MODELS_DIR}'.")
        if not second_stage_loaded:
            error_msg.append(f"No second-stage models found or loaded from '{SECOND_STAGE_MODELS_DIR}'.")
        
        return render_template(
            'prediction_app.html', # Corrected to render prediction_app.html
            error=" ".join(error_msg) + " Please ensure the correct directories exist and contain your .joblib pipeline files.",
            models_loaded=False,
            mappings=ALL_MAPPINGS
        )
    return render_template(
        'prediction_app.html', # Corrected to render prediction_app.html
        models_loaded=True,
        model_names=sorted(first_stage_models.keys()), # Display first stage model names
        mappings=ALL_MAPPINGS
    )

@app.route('/psychoactive_classification')
def psychoactive_classification_page():
    """
    Renders the psychoactive drug classification page (regression_app.html) with chemical inputs.
    """
    if not third_stage_model or not third_stage_mlb or not third_stage_scaler or not third_stage_selector:
        return render_template(
            'regression_app.html',
            error="Psychoactive Classification model or its preprocessors not loaded. Please check server logs.",
            model_loaded=False
        )
    return render_template('regression_app.html', model_loaded=True)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the drug usage prediction form (prediction_app.html).
    It retrieves the input scores and categorical selections,
    maps categorical selections to numerical values for first stage,
    and passes string categorical values for second stage if needed.
    Always returns JSON response.
    """
    if not first_stage_models or not second_stage_models:
        error_msg = []
        if not first_stage_models:
            error_msg.append(f"Prediction failed: No first-stage models are loaded from '{FIRST_STAGE_MODELS_DIR}'.")
        if not second_stage_models:
            error_msg.append(f"Prediction failed: No second-stage models are loaded from '{SECOND_STAGE_MODELS_DIR}'.")
        # Ensure error is returned as JSON
        return jsonify({
            "error": " ".join(error_msg),
            "models_loaded": False,
            "predictions": {}, # Empty predictions for error state
            "impulsive_prediction": "Error",
            "ss_prediction": "Error",
            "second_stage_errors": ["Models not loaded on server."]
        }), 500

    try:
        # Retrieve JSON data from the request body
        data = request.get_json(force=True)

        # Retrieve numerical input scores using .get() for safety
        n_score = float(data.get('n_score'))
        e_score = float(data.get('e_score'))
        a_score = float(data.get('a_score'))
        o_score = float(data.get('o_score'))
        c_score = float(data.get('c_score'))

        # Retrieve categorical input strings using .get()
        age_str = data.get('age')
        gender_str = data.get('gender')
        education_str = data.get('education')
        country_str = data.get('country')
        ethinicity_str = data.get('ethinicity')

        # Map categorical strings to their numerical 'Real' values for FIRST STAGE models
        age_val = AGE_MAPPING.get(age_str)
        gender_val = GENDER_MAPPING.get(gender_str)
        country_val = COUNTRY_MAPPING.get(country_str)
        
        # Validate if mappings returned None (i.e., invalid selection)
        if any(val is None for val in [age_val, gender_val, country_val]) or \
           education_str not in EDUCATION_MAPPING or ethinicity_str not in ETHINICITY_MAPPING:
            raise ValueError("One or more categorical inputs are invalid or missing. Please select from dropdowns.")

    except (ValueError, TypeError) as e:
        # Ensure error is returned as JSON
        return jsonify({
            "error": f"Invalid input or missing field: {e}. Please ensure all scores are numeric and selections are valid.",
            "predictions": {},
            "impulsive_prediction": "Error",
            "ss_prediction": "Error",
            "second_stage_errors": [f"Input validation failed: {e}"],
            "input_values": { # Return inputs for re-population on error
                "n_score": data.get('n_score', ''), "e_score": data.get('e_score', ''),
                "a_score": data.get('a_score', ''), "o_score": data.get('o_score', ''),
                "c_score": data.get('c_score', ''), "age_selected": data.get('age', ''),
                "gender_selected": data.get('gender', ''), "education_selected": data.get('education', ''),
                "country_selected": data.get('country', ''), "ethinicity_selected": data.get('ethinicity', '')
            }
        }), 400
    except KeyError as e:
        # Ensure error is returned as JSON
        return jsonify({
            "error": f"Missing input field: {e}. All required fields must be provided.",
            "predictions": {},
            "impulsive_prediction": "Error",
            "ss_prediction": "Error",
            "second_stage_errors": [f"Missing input field: {e}"],
            "input_values": { # Return inputs for re-population on error
                "n_score": data.get('n_score', ''), "e_score": data.get('e_score', ''),
                "a_score": data.get('a_score', ''), "o_score": data.get('o_score', ''),
                "c_score": data.get('c_score', ''), "age_selected": data.get('age', ''),
                "gender_selected": data.get('gender', ''), "education_selected": data.get('education', ''),
                "country_selected": data.get('country', ''), "ethinicity_selected": data.get('ethinicity', '')
            }
        }), 400

    # --- FIRST STAGE: Predict Drug Usage ---
    input_data_first_stage = {
        'Nscore': [float(n_score)], 'Escore': [float(e_score)], 'Ascore': [float(a_score)], 'Oscore': [float(o_score)], 'Cscore': [float(c_score)],
        'Age': [float(age_val)], 'Gender': [float(gender_val)], 'Education': [float(EDUCATION_MAPPING.get(education_str))],
        'Country': [float(country_val)], 'Ethinicity': [float(ETHINICITY_MAPPING.get(ethinicity_str))]
    }
    try:
        input_df_first_stage = pd.DataFrame(input_data_first_stage, columns=EXPECTED_FEATURE_ORDER_FIRST_STAGE)
    except Exception as e:
        return jsonify({
            "error": f"Error creating first-stage input DataFrame: {e}. Ensure feature order and names match.",
            "predictions": {},
            "impulsive_prediction": "Error",
            "ss_prediction": "Error",
            "second_stage_errors": [f"Internal error preparing first stage input: {e}"],
            "input_values": {
                "n_score": n_score, "e_score": e_score, "a_score": a_score, "o_score": o_score, "c_score": c_score,
                "age_selected": age_str, "gender_selected": gender_str, "education_selected": education_str,
                "country_selected": country_str, "ethinicity_selected": ethinicity_str
            }
        }), 500

    predictions = {}
    numerical_drug_predictions = {} # To store numerical predictions (0 or 1) for next stage

    for drug_name, pipeline_model in first_stage_models.items():
        try:
            prediction = pipeline_model.predict(input_df_first_stage)[0]
            numerical_drug_predictions[drug_name] = int(prediction)
            predictions[drug_name] = "Takes Drug" if prediction == 1 else "Does Not Take Drug"
        except Exception as e:
            predictions[drug_name] = f"Prediction Error: {e}"
            numerical_drug_predictions[drug_name] = -1 # Indicate error for numerical output
            print(f"Error predicting for {drug_name} (first stage): {e}")

    print("\n--- Numerical Drug Predictions (for next model input) ---")
    print(numerical_drug_predictions)
    print("----------------------------------------------------------\n")

    # --- SECOND STAGE: Predict Impulsive and SS ---
    impulsive_prediction_val = None
    ss_prediction_val = None
    second_stage_prediction_errors = []

    # Prepare input for the second stage model
    input_data_second_stage = {
        'Nscore': [float(n_score)],
        'Escore': [float(e_score)],
        'Oscore': [float(o_score)],
        'Ascore': [float(a_score)],
        'Cscore': [float(c_score)],
        'Education': [education_str], # Passed as string
        'Age': [float(age_val)], # Passed as numerical
        'Ethinicity': [ethinicity_str] # Passed as string
    }
    
    for drug in substances_to_process_list:
        input_data_second_stage[drug] = [float(numerical_drug_predictions.get(drug, 0))]

    try:
        input_df_second_stage = pd.DataFrame(input_data_second_stage, columns=EXPECTED_FEATURE_ORDER_SECOND_STAGE)
        # --- DIAGNOSTIC PRINTS FOR SECOND STAGE INPUT ---
        print("\n--- Second Stage Input DataFrame Info ---")
        print(input_df_second_stage.info())
        print("\nSecond Stage Input DataFrame Content:")
        print(input_df_second_stage)
        if input_df_second_stage.isnull().any().any():
            print("\nWARNING: NaNs found in second-stage input DataFrame!")
            print(input_df_second_stage.isnull().sum())
        print("-----------------------------------------\n")
        # --- END DIAGNOSTIC PRINTS ---

    except Exception as e:
        second_stage_prediction_errors.append(f"Error creating second-stage input DataFrame: {e}. Ensure feature order and names match.")
        print(second_stage_prediction_errors[-1])

    if not second_stage_prediction_errors:
        if 'Impulsive' in second_stage_models:
            try:
                impulsive_prediction_val = second_stage_models['Impulsive'].predict(input_df_second_stage)[0]
                print(f"Impulsive prediction: {impulsive_prediction_val:.4f}")
            except Exception as e:
                second_stage_prediction_errors.append(f"Error predicting Impulsive score: {e}")
                print(f"Error predicting Impulsive score: {e}") # Print to console for debugging
        else:
            second_stage_prediction_errors.append("Impulsive model not loaded.")
            print("Impulsive model not loaded.") # Print to console for debugging

        if 'SS' in second_stage_models:
            try:
                ss_prediction_val = second_stage_models['SS'].predict(input_df_second_stage)[0]
                print(f"SS prediction: {ss_prediction_val:.4f}")
            except Exception as e:
                second_stage_prediction_errors.append(f"Error predicting SS score: {e}")
                print(f"Error predicting SS score: {e}") # Print to console for debugging
        else:
            second_stage_prediction_errors.append("SS model not loaded.")
            print("SS model not loaded.") # Print to console for debugging

    # Return JSON response with results and input values
    return jsonify({
        "predictions": predictions, # First stage predictions
        "impulsive_prediction": f"{impulsive_prediction_val:.4f}" if impulsive_prediction_val is not None else "N/A",
        "ss_prediction": f"{ss_prediction_val:.4f}" if ss_prediction_val is not None else "N/A",
        "second_stage_errors": second_stage_prediction_errors, # Pass errors to UI
        "input_values": {
            "n_score": n_score, "e_score": e_score, "a_score": a_score, "o_score": o_score, "c_score": c_score,
            "age_selected": age_str, "gender_selected": gender_str, "education_selected": education_str,
            "country_selected": country_str, "ethinicity_selected": ethinicity_str
        }
    })

# --- NEW/MODIFIED ROUTE for Psychoactive Classification (Chemical Inputs) ---
@app.route('/predict_psychoactive', methods=['POST'])
def predict_psychoactive():
    """
    Handles the prediction request specifically for the psychoactive drug classification
    model (psychoactive_drug_model.joblib), taking chemical inputs from regression_app.html.
    """
    # Check if all necessary third-stage assets are loaded
    if not all([third_stage_model, third_stage_mlb, third_stage_scaler, third_stage_selector, EXPECTED_FEATURE_ORDER_THIRD_STAGE]):
        return jsonify({
            "error": "Psychoactive Classification model or its preprocessors not fully loaded on server. Please check server logs.",
            "predicted_effects": "Error",
            "predicted_psychoactivity": "Error",
            "predicted_donors": "Error",
            "predicted_acceptors": "Error",
            "predicted_smiles": "Error"
        }), 500

    try:
        data = request.get_json(force=True)

        # Extracting user inputs
        drug_name_input = data.get('drug_name', 'N/A')
        molecular_formula_input = data.get('molecular_formula', 'N/A')
        molecular_weight = float(data.get('molecular_weight'))
        tpsa = float(data.get('tpsa'))
        rotatable_bond_count = int(data.get('rotatable_bond_count'))
        hydrogen_bond_donors = int(data.get('hydrogen_bond_donors'))
        hydrogen_bond_acceptors = int(data.get('hydrogen_bond_acceptors'))
        functional_groups_str = data.get('functional_groups', '')

        # --- Process Functional Groups String using loaded MLB ---
        # Convert comma-separated string to a list of lists (required by MLB.transform)
        if functional_groups_str:
            input_fgs_list = [fg.strip() for fg in functional_groups_str.split(',') if fg.strip()]
        else:
            input_fgs_list = [] # Handle empty functional groups

        # Use the loaded mlb to transform the input list of functional groups.
        fg_encoded_array = third_stage_mlb.transform([input_fgs_list])[0]
        
        # Create a dictionary for the functional group features, ensuring all expected FG columns are present
        # and initialized to 0.0, then update with the actual encoded input.
        functional_group_features_dict = {cls: 0.0 for cls in third_stage_mlb.classes_}
        
        # Update with the actual encoded features based on the input
        for i, fg_col_name in enumerate(third_stage_mlb.classes_):
            functional_group_features_dict[fg_col_name] = float(fg_encoded_array[i])

        # --- Prepare the input DataFrame for the THIRD STAGE MODEL ---
        input_data_raw_features = {
            'MolecularWeight': molecular_weight,
            'TPSA': tpsa,
            'RotatableBondCount': rotatable_bond_count,
            'HydrogenBondDonors': hydrogen_bond_donors,
            'HydrogenBondAcceptors': hydrogen_bond_acceptors,
            **functional_group_features_dict # Unpack functional group features
        }

        # Create DataFrame ensuring the column order matches the dynamically generated order
        ordered_input_values = [input_data_raw_features[col] for col in EXPECTED_FEATURE_ORDER_THIRD_STAGE]
        input_df_third_stage = pd.DataFrame([ordered_input_values], columns=EXPECTED_FEATURE_ORDER_THIRD_STAGE)

        # --- Apply Preprocessing steps: Scaling and Feature Selection ---
        X_scaled_predict = third_stage_scaler.transform(input_df_third_stage)
        X_final_predict = third_stage_selector.transform(X_scaled_predict)

        # --- Make Prediction with the Third Stage Model ---
        final_prediction_raw = third_stage_model.predict(X_final_predict)[0]

        # --- Interpret and format the prediction ---
        predicted_psychoactivity_text = "Psychoactive" if final_prediction_raw == 1 else "Non-Psychoactive"
        
        predicted_effects = "Effects correlated with chemical structure and predicted psychoactivity."
        predicted_donors = str(hydrogen_bond_donors)
        predicted_acceptors = str(hydrogen_bond_acceptors)

        predicted_smiles = molecular_formula_input # Still a placeholder based on input

        return jsonify({
            "predicted_effects": predicted_effects,
            "predicted_psychoactivity": predicted_psychoactivity_text,
            "predicted_donors": predicted_donors,
            "predicted_acceptors": predicted_acceptors,
            "predicted_smiles": predicted_smiles
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid numerical input: {e}. Please check your values."}), 400
    except KeyError as e:
        return jsonify({"error": f"Missing input field: {e}. All fields must be provided."}), 400
    except Exception as e:
        print(f"Error during psychoactive prediction: {e}")
        return jsonify({"error": f"An unexpected error occurred during psychoactive prediction: {e}"}), 500


# --- Main execution block ---
if __name__ == '__main__':
    # IMPORTANT: You MUST run your provided training scripts to generate the actual .joblib model files.
    # First-stage models (e.g., model_Drug_1.joblib) must be placed in the 'saved_models_usage' directory.
    # Second-stage models (e.g., best_model_Impulsive_Random_Forest.joblib) must be placed in the 'trained_models' directory.
    # Third-stage model and preprocessors must be in 'third_model' directory.

    print("\n" + "="*80)
    print("CRITICAL: Ensure your model directories are correctly set up and contain your .joblib files.")
    print(f"First-stage models expected in: '{FIRST_STAGE_MODELS_DIR}'")
    print(f"Second-stage models expected in: '{SECOND_STAGE_MODELS_DIR}'")
    print(f"Third-stage model assets expected in: '{THIRD_MODEL_DIR}'")
    print("="*80 + "\n")

    # Run the Flask application in debug mode.
    app.run(debug=True)
