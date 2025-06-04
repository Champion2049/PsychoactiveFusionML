# netlify/functions/predict.py
import json
import sys
import os

# --- Robust path manipulation for local and Netlify environments ---
# Get the directory where the current script (predict.py) is located
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the project root directory
# This assumes ml_logic.py is one level up from netlify/functions/
project_root_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the project root to sys.path if it's not already there
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
# --- End robust path manipulation ---

# Import the prediction logic from ml_logic.py
# This import will now look in the project_root_dir
from ml_logic import predict_scores

def handler(event, context):
    """
    Netlify Function handler for prediction requests.
    Receives POST requests, processes input, calls ML prediction logic,
    and returns a JSON response.
    """
    if event['httpMethod'] != 'POST':
        return {
            'statusCode': 405,
            'body': json.dumps({'error': 'Method Not Allowed. Only POST requests are accepted.'})
        }

    try:
        # Parse the JSON body from the request
        request_body = json.loads(event['body'])
        
        # Call the prediction logic from ml_logic.py
        # This function handles all validation, data preparation, and model inference
        prediction_results = predict_scores(request_body)

        # Check if there was an error during prediction
        if "error" in prediction_results:
            return {
                'statusCode': 400, # Bad Request
                'headers': { 'Content-Type': 'application/json' },
                'body': json.dumps({'error': prediction_results['error']})
            }
        
        # Return the successful prediction results as JSON
        return {
            'statusCode': 200,
            'headers': { 'Content-Type': 'application/json' },
            'body': json.dumps(prediction_results)
        }

    except json.JSONDecodeError:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid JSON in request body.'})
        }
    except Exception as e:
        # Catch any unexpected errors during function execution
        print(f"Unhandled error in Netlify function: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f'Internal server error: {e}'})
        }

