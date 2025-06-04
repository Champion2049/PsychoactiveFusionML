# netlify/functions/predict.py
import json
import sys
import os

# Add the parent directory of 'netlify/functions' to the Python path
# This allows importing 'ml_logic' from the root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the prediction logic from ml_logic.py
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

