# app.py

from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Define the directory where your first-stage .joblib models are stored (drug usage classification)
FIRST_STAGE_MODELS_DIR = 'saved_models_usage'

# Define the directory where your second-stage .joblib models are stored (Impulsive, SS regression)
SECOND_STAGE_MODELS_DIR = 'trained_models'

THIRD_MODEL_DIR = 'third_model'  # Directory for the third model

# --- Mappings for Input Parameters ---
# These dictionaries map the user-friendly string inputs to their numerical 'Real' values.
AGE_MAPPING = {
    "18-24": -0.95197,
    "25-34": -0.07854,
    "35-44": 0.49788,
    "45-54": 1.09449,
    "55-64": 1.82213,
    "65+": 2.59171
}

GENDER_MAPPING = {
    "Female": 0.48246,
    "Male": -0.48246
}

EDUCATION_MAPPING = {
    "Left school before 16 years": -2.43591,
    "Left school at 16 years": -1.73790,
    "Left school at 17 years": -1.43719,
    "Left school at 18 years": -1.22751,
    "Some college or university, no certificate or degree": -0.61113,
    "Professional certificate/ diploma": -0.05921,
    "University degree": 0.45468,
    "Masters degree": 1.16365,
    "Doctorate degree": 1.98437
}

COUNTRY_MAPPING = {
    "Australia": -0.09765,
    "Canada": 0.24923,
    "New Zealand": -0.46841,
    "Other": -0.28519,
    "Republic of Ireland": 0.21128,
    "UK": 0.96082,
    "USA": -0.57009
}

ETHINICITY_MAPPING = {
    "Asian": -0.50212,
    "Black": -1.10702,
    "Mixed-Black/Asian": 1.90725,
    "Mixed-White/Asian": 0.12600,
    "Mixed-White/Black": -0.22166,
    "Other": 0.11440,
    "White": -0.31685
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
    'Nscore', 'Escore', 'Ascore', 'Oscore', 'Cscore',
    'Age', 'Gender', 'Education', 'Country', 'Ethinicity'
]

# Define the feature names for the SECOND STAGE models (Impulsive, SS) in the exact order expected.
EXPECTED_FEATURE_ORDER_SECOND_STAGE = [
    'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore',
    'Education', 'Age', 'Ethinicity'
] + substances_to_process_list

# --- Model Loading Functions ---

def load_first_stage_models(models_dir):
    """Loads all .joblib model pipeline files for drug usage classification."""
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
    """Loads .joblib model pipeline files for Impulsive and SS prediction."""
    models = {}
    if not os.path.exists(models_dir):
        print(f"Error: Second-stage models directory '{models_dir}' not found. Please create it and place your .joblib files inside.")
        return models

    expected_targets = ['Impulsive', 'SS']
    for filename in os.listdir(models_dir):
        if filename.endswith('.joblib'):
            target_name = None
            try:
                parts = filename.replace('.joblib', '').split('_')
                if len(parts) >= 3 and parts[0] == 'best' and parts[1] == 'model':
                    target_name = parts[2]
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


# Load all models once when the Flask application starts.
first_stage_models = load_first_stage_models(FIRST_STAGE_MODELS_DIR)
second_stage_models = load_second_stage_models(SECOND_STAGE_MODELS_DIR)

# --- Flask Routes ---

@app.route('/')
def home():
    """
    Renders the main home page.
    """
    return render_template('home.html')

@app.route('/prediction_app')
def prediction_app_page():
    """
    Renders the prediction application page.
    It passes information about whether models were loaded successfully,
    the names of the loaded models, and the mappings for dropdowns to the HTML template.
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
            'prediction_app.html',
            error=" ".join(error_msg) + " Please ensure the correct directories exist and contain your .joblib pipeline files.",
            models_loaded=False,
            mappings=ALL_MAPPINGS
        )
    return render_template(
        'prediction_app.html',
        models_loaded=True,
        model_names=sorted(first_stage_models.keys()), # Display first stage model names
        mappings=ALL_MAPPINGS
    )

@app.route('/psychoactive_classification') # Renamed route
def psychoactive_classification_page(): # Renamed function
    """
    Renders the psychoactive drug classification page (formerly empty_page).
    """
    # Assuming the user renamed empty_page.html to regression_app.html
    return render_template('regression_app.html') # Corrected template name to match user's rename


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the prediction application form.
    It retrieves the input scores and categorical selections,
    maps categorical selections to numerical values,
    creates a Pandas DataFrame with appropriate column names,
    and then passes this DataFrame to the loaded ML pipelines for prediction.
    """
    if not first_stage_models or not second_stage_models:
        error_msg = []
        if not first_stage_models:
            error_msg.append(f"Prediction failed: No first-stage models are loaded from '{FIRST_STAGE_MODELS_DIR}'.")
        if not second_stage_models:
            error_msg.append(f"Prediction failed: No second-stage models are loaded from '{SECOND_STAGE_MODELS_DIR}'.")
        return render_template(
            'prediction_app.html',
            error=" ".join(error_msg),
            models_loaded=False,
            mappings=ALL_MAPPINGS
        )

    try:
        # Retrieve numerical input scores
        n_score = float(request.form['n_score'])
        e_score = float(request.form['e_score'])
        a_score = float(request.form['a_score'])
        o_score = float(request.form['o_score'])
        c_score = float(request.form['c_score'])

        # Retrieve categorical input strings (these are the raw user selections)
        age_str = request.form['age']
        gender_str = request.form['gender']
        education_str = request.form['education'] # Keep as string for second stage
        country_str = request.form['country']
        ethinicity_str = request.form['ethinicity'] # Keep as string for second stage

        # Map categorical strings to their numerical 'Real' values for FIRST STAGE models
        # and for numerical features in the SECOND STAGE.
        age_val = AGE_MAPPING.get(age_str)
        gender_val = GENDER_MAPPING.get(gender_str)
        country_val = COUNTRY_MAPPING.get(country_str)
        
        # Validate if mappings returned None (i.e., invalid selection)
        if any(val is None for val in [age_val, gender_val, country_val]) or \
           education_str not in EDUCATION_MAPPING or ethinicity_str not in ETHINICITY_MAPPING:
            raise ValueError("One or more categorical inputs are invalid or missing.")

    except ValueError as e:
        return render_template(
            'prediction_app.html',
            error=f"Invalid input: {e}. Please ensure all scores are numeric and selections are valid.",
            models_loaded=True,
            model_names=sorted(first_stage_models.keys()),
            mappings=ALL_MAPPINGS,
            n_score=request.form.get('n_score', ''), e_score=request.form.get('e_score', ''),
            a_score=request.form.get('a_score', ''), o_score=request.form.get('o_score', ''),
            c_score=request.form.get('c_score', ''), age_selected=request.form.get('age', ''),
            gender_selected=request.form.get('gender', ''), education_selected=request.form.get('education', ''),
            country_selected=request.form.get('country', ''), ethinicity_selected=request.form.get('ethinicity', '')
        )
    except KeyError as e:
        return render_template(
            'prediction_app.html',
            error=f"Missing input field: {e}. Please ensure all required fields are provided.",
            models_loaded=True,
            model_names=sorted(first_stage_models.keys()),
            mappings=ALL_MAPPINGS
        )

    # --- FIRST STAGE: Predict Drug Usage ---
    # Prepare the input features for the first-stage models as a DataFrame.
    # All values here are numerical (mapped from strings for categorical inputs).
    input_data_first_stage = {
        'Nscore': [float(n_score)], 'Escore': [float(e_score)], 'Ascore': [float(a_score)], 'Oscore': [float(o_score)], 'Cscore': [float(c_score)],
        'Age': [float(age_val)], 'Gender': [float(gender_val)],
        'Education': [float(EDUCATION_MAPPING.get(education_str))], # Mapped to numerical for first stage
        'Country': [float(country_val)],
        'Ethinicity': [float(ETHINICITY_MAPPING.get(ethinicity_str))] # Mapped to numerical for first stage
    }
    try:
        input_df_first_stage = pd.DataFrame(input_data_first_stage, columns=EXPECTED_FEATURE_ORDER_FIRST_STAGE)
    except Exception as e:
        return render_template(
            'prediction_app.html',
            error=f"Error creating first-stage input DataFrame: {e}. Ensure feature order and names match.",
            models_loaded=True, model_names=sorted(first_stage_models.keys()), mappings=ALL_MAPPINGS,
            n_score=n_score, e_score=e_score, a_score=a_score, o_score=o_score, c_score=c_score,
            age_selected=age_str, gender_selected=gender_str, education_selected=education_str,
            country_selected=country_str, ethinicity_selected=ethinicity_str
        )

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
    impulsive_prediction = None
    ss_prediction = None
    second_stage_prediction_errors = []

    # Prepare input for the second stage model
    # Numerical features are cast to float.
    # Categorical features ('Education', 'Ethinicity') are kept as STRINGS
    # because the second-stage training script's ColumnTransformer expects them as 'object' dtypes for OneHotEncoding.
    input_data_second_stage = {
        'Nscore': [float(n_score)],
        'Escore': [float(e_score)],
        'Oscore': [float(o_score)],
        'Ascore': [float(a_score)],
        'Cscore': [float(c_score)],
        'Education': [education_str], # Passed as string
        'Age': [float(AGE_MAPPING.get(age_str))], # Passed as numerical, use mapped value
        'Ethinicity': [ethinicity_str] # Passed as string
    }
    
    for drug in substances_to_process_list:
        # Use .get() with a default value (e.g., 0) in case a drug prediction failed (-1) or is missing
        # Ensure the value is cast to float, as these are numerical drug usage indicators.
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
                impulsive_prediction = second_stage_models['Impulsive'].predict(input_df_second_stage)[0]
                print(f"Impulsive prediction: {impulsive_prediction:.4f}")
            except Exception as e:
                second_stage_prediction_errors.append(f"Error predicting Impulsive score: {e}")
                print(f"Error predicting Impulsive score: {e}") # Print to console for debugging
        else:
            second_stage_prediction_errors.append("Impulsive model not loaded.")
            print("Impulsive model not loaded.") # Print to console for debugging

        if 'SS' in second_stage_models:
            try:
                ss_prediction = second_stage_models['SS'].predict(input_df_second_stage)[0]
                print(f"SS prediction: {ss_prediction:.4f}")
            except Exception as e:
                second_stage_prediction_errors.append(f"Error predicting SS score: {e}")
                print(f"Error predicting SS score: {e}") # Print to console for debugging
        else:
            second_stage_prediction_errors.append("SS model not loaded.")
            print("SS model not loaded.") # Print to console for debugging

    # Render the template again, passing all input values and the prediction results
    return render_template(
        'prediction_app.html',
        predictions=predictions, # First stage predictions
        impulsive_prediction=f"{impulsive_prediction:.4f}" if impulsive_prediction is not None else "N/A",
        ss_prediction=f"{ss_prediction:.4f}" if ss_prediction is not None else "N/A",
        second_stage_errors=second_stage_prediction_errors, # Pass errors to UI
        n_score=n_score, e_score=e_score, a_score=a_score, o_score=o_score, c_score=c_score,
        age_selected=age_str, gender_selected=gender_str, education_selected=education_str,
        country_selected=country_str, ethinicity_selected=ethinicity_str,
        models_loaded=True, model_names=sorted(first_stage_models.keys()), mappings=ALL_MAPPINGS
    )

# --- Main execution block ---
if __name__ == '__main__':
    print("\n" + "="*80)
    print("CRITICAL: Ensure your model directories are correctly set up and contain your .joblib files.")
    print(f"First-stage models expected in: '{FIRST_STAGE_MODELS_DIR}'")
    print(f"Second-stage models expected in: '{SECOND_STAGE_MODELS_DIR}'")
    print("="*80 + "\n")

    app.run(debug=True)