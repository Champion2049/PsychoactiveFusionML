# ml_logic.py

import joblib
import os
import numpy as np
import pandas as pd

# Define the directory where your first-stage .joblib models are stored (drug usage classification)
FIRST_STAGE_MODELS_DIR = 'saved_models_usage'

# Define the directory where your second-stage .joblib models are stored (Impulsive, SS regression)
SECOND_STAGE_MODELS_DIR = 'trained_models'

# --- Mappings for Input Parameters ---
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
        print(f"Error: First-stage models directory '{models_dir}' not found.")
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
                    # print(f"Successfully loaded first-stage pipeline for: {drug_name} from {filepath}")
                except Exception as e:
                    print(f"Error loading first-stage model pipeline '{filename}' for drug '{drug_name}': {e}")
    
    print(f"Finished loading first-stage models. Total models loaded: {len(models)}")
    return models

def load_second_stage_models(models_dir):
    """Loads .joblib model pipeline files for Impulsive and SS prediction."""
    models = {}
    if not os.path.exists(models_dir):
        print(f"Error: Second-stage models directory '{models_dir}' not found.")
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
                    # print(f"Successfully loaded second-stage pipeline for: {target_name} from {filepath}")
                except Exception as e:
                    print(f"Error loading second-stage model pipeline '{filename}' for target '{target_name}': {e}")
    
    print(f"Finished loading second-stage models. Total models loaded: {len(models)}")
    return models

# Load all models once when the module is imported (will happen on Netlify Function cold start)
first_stage_models = load_first_stage_models(FIRST_STAGE_MODELS_DIR)
second_stage_models = load_second_stage_models(SECOND_STAGE_MODELS_DIR)

def predict_scores(input_data_raw):
    """
    Performs two-stage prediction based on raw input data.
    Args:
        input_data_raw (dict): Dictionary containing raw input values from the form.
    Returns:
        dict: A dictionary containing predictions for drug usage, Impulsive, and SS scores,
              along with any errors.
    """
    predictions = {}
    numerical_drug_predictions = {}
    second_stage_prediction_errors = []
    impulsive_prediction = "N/A"
    ss_prediction = "N/A"

    # Validate and map inputs
    try:
        n_score = float(input_data_raw['n_score'])
        e_score = float(input_data_raw['e_score'])
        a_score = float(input_data_raw['a_score'])
        o_score = float(input_data_raw['o_score'])
        c_score = float(input_data_raw['c_score'])

        age_str = input_data_raw['age']
        gender_str = input_data_raw['gender']
        education_str = input_data_raw['education']
        country_str = input_data_raw['country']
        ethinicity_str = input_data_raw['ethinicity']

        age_val = AGE_MAPPING.get(age_str)
        gender_val = GENDER_MAPPING.get(gender_str)
        country_val = COUNTRY_MAPPING.get(country_str)

        if any(val is None for val in [age_val, gender_val, country_val]) or \
           education_str not in EDUCATION_MAPPING or ethinicity_str not in ETHINICITY_MAPPING:
            raise ValueError("One or more categorical inputs are invalid or missing.")

    except (ValueError, KeyError) as e:
        return {"error": f"Invalid or missing input data: {e}"}

    # --- FIRST STAGE: Predict Drug Usage ---
    input_data_first_stage_dict = {
        'Nscore': [float(n_score)], 'Escore': [float(e_score)], 'Ascore': [float(a_score)], 'Oscore': [float(o_score)], 'Cscore': [float(c_score)],
        'Age': [float(age_val)], 'Gender': [float(gender_val)],
        'Education': [float(EDUCATION_MAPPING.get(education_str))], # Mapped to numerical for first stage
        'Country': [float(country_val)],
        'Ethinicity': [float(ETHINICITY_MAPPING.get(ethinicity_str))] # Mapped to numerical for first stage
    }
    try:
        input_df_first_stage = pd.DataFrame(input_data_first_stage_dict, columns=EXPECTED_FEATURE_ORDER_FIRST_STAGE)
    except Exception as e:
        return {"error": f"Error creating first-stage input DataFrame: {e}. Ensure feature order and names match."}

    for drug_name, pipeline_model in first_stage_models.items():
        try:
            prediction = pipeline_model.predict(input_df_first_stage)[0]
            numerical_drug_predictions[drug_name] = int(prediction)
            predictions[drug_name] = "Takes Drug" if prediction == 1 else "Does Not Take Drug"
        except Exception as e:
            predictions[drug_name] = f"Prediction Error: {e}"
            numerical_drug_predictions[drug_name] = -1
            print(f"Error predicting for {drug_name} (first stage): {e}")

    # --- SECOND STAGE: Predict Impulsive and SS ---
    input_data_second_stage_dict = {
        'Nscore': [float(n_score)], 'Escore': [float(e_score)], 'Oscore': [float(o_score)], 'Ascore': [float(a_score)], 'Cscore': [float(c_score)],
        'Education': [education_str], # Passed as string for second stage
        'Age': [float(age_val)], # Passed as numerical for second stage
        'Ethinicity': [ethinicity_str] # Passed as string for second stage
    }
    
    for drug in substances_to_process_list:
        input_data_second_stage_dict[drug] = [float(numerical_drug_predictions.get(drug, 0))]

    try:
        input_df_second_stage = pd.DataFrame(input_data_second_stage_dict, columns=EXPECTED_FEATURE_ORDER_SECOND_STAGE)
    except Exception as e:
        second_stage_prediction_errors.append(f"Error creating second-stage input DataFrame: {e}. Ensure feature order and names match.")

    if not second_stage_prediction_errors:
        if 'Impulsive' in second_stage_models:
            try:
                impulsive_prediction = second_stage_models['Impulsive'].predict(input_df_second_stage)[0]
            except Exception as e:
                second_stage_prediction_errors.append(f"Error predicting Impulsive score: {e}")
        else:
            second_stage_prediction_errors.append("Impulsive model not loaded.")

        if 'SS' in second_stage_models:
            try:
                ss_prediction = second_stage_models['SS'].predict(input_df_second_stage)[0]
            except Exception as e:
                second_stage_prediction_errors.append(f"Error predicting SS score: {e}")
        else:
            second_stage_prediction_errors.append("SS model not loaded.")

    return {
        "predictions": predictions,
        "impulsive_prediction": f"{impulsive_prediction:.4f}" if isinstance(impulsive_prediction, (float, np.float32, np.float64)) else "N/A",
        "ss_prediction": f"{ss_prediction:.4f}" if isinstance(ss_prediction, (float, np.float32, np.float64)) else "N/A",
        "second_stage_errors": second_stage_prediction_errors,
        # Also return the raw inputs for display in the frontend
        "input_values": {
            "n_score": n_score, "e_score": e_score, "a_score": a_score, "o_score": o_score, "c_score": c_score,
            "age_selected": age_str, "gender_selected": gender_str, "education_selected": education_str,
            "country_selected": country_str, "ethinicity_selected": ethinicity_str
        }
    }

# Initial model loading (will run once per cold start of the function)
# This ensures models are loaded into memory and reused for subsequent requests.
if not first_stage_models or not second_stage_models:
    print("WARNING: Not all models loaded successfully. Predictions may fail.")
    print(f"First-stage models loaded: {len(first_stage_models)}")
    print(f"Second-stage models loaded: {len(second_stage_models)}")

