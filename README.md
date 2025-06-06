# PsychoactiveFusionML: Predicting Drug Usage and Personality Traits

## 📊 Project Overview

PsychoactiveFusionML is a machine learning project designed to explore the intricate relationships between an individual's personality traits and their patterns of psychoactive substance use. Leveraging a two-stage predictive modeling approach, this application aims to provide insights into potential drug usage based on personality and demographic data, and subsequently predict specific behavioral traits like Impulsivity and Sensation Seeking. The project features a user-friendly web interface built with Flask, allowing users to input their data and receive predictions.

## ✨ Features

* **Two-Stage Prediction System:**

    * **Stage 1: Drug Usage Classification:** Predicts the likelihood of an individual using various psychoactive drugs based on their Big Five personality scores (N, E, O, A, C) and demographic information (Age, Gender, Education, Country, Ethnicity).

    * **Stage 2: Personality Trait Regression:** Predicts Impulsivity and Sensation Seeking scores based on the input personality and demographic data, along with the predicted drug usage from Stage 1.

* **Intuitive Web Interface:** A clean and responsive Flask web application for easy data input and prediction display.

* **Dynamic Theme Toggle:** Switch between a vibrant light mode and a sleek dark mode for enhanced user experience.

* **Modular Design:** Separates concerns with dedicated directories for models, templates, and the main Flask application.

* **Real-time Feedback:** Displays predictions directly on the web page.

## 🚀 Technologies Used

* **Backend:** Python 3, Flask

* **Machine Learning:** `scikit-learn` (models saved using `joblib`)

* **Data Handling:** `pandas`, `numpy`

* **Frontend:** HTML, CSS (Tailwind CSS framework), JavaScript

* **Deployment (Conceptual)::** Designed for easy deployment on web servers.

## 📦 Installation

To get a copy of this project up and running on your local machine for development and testing purposes, follow these steps.

### Prerequisites

* Python 3.8+

* `pip` (Python package installer)

### Steps

1.  **Clone the repository:**

    ```
    git clone [https://github.com/Champion2049/PsychoactiveFusionML.git](https://github.com/Champion2049/PsychoactiveFusionML.git)
    cd PsychoactiveFusionML
    ```

2.  **Create a virtual environment (recommended):**

    ```
    python -m venv venv
    ```

    * On Windows:

        ```
        .\venv\Scripts\activate
        ```

    * On macOS/Linux:

        ```
        source venv/bin/activate
        ```

3.  **Install dependencies:**

    ```
    pip install -r requirements.txt
    ```

    *(If `requirements.txt` is not provided, you'll need to create it manually with `Flask`, `scikit-learn`, `pandas`, `numpy`, and any other libraries your models might implicitly depend on, then run `pip install -r requirements.txt`)*. Example `requirements.txt`:

    ```
    Flask==2.3.3
    scikit-learn==1.3.0
    pandas==2.0.3
    numpy==1.25.2
    joblib==1.3.2
    ```

4.  **Place your trained models:**
    This project expects trained machine learning models (`.joblib` files) in specific directories:

    * **First-stage drug usage classification models:** Place these in a directory named `saved_models_usage/`. The `app.py` expects them to be named like `model_Drug_1.joblib`, `model_Drug_2.joblib`, etc.

    * **Second-stage personality trait regression models:** Place these in a directory named `trained_models/`. The `app.py` expects them to be named like `best_model_Impulsive_*.joblib` and `best_model_SS_*.joblib`.

    *Example directory structure after placing models:*

    ```
    PsychoactiveFusionML/
    ├── app.py
    ├── requirements.txt
    ├── templates/
    │   ├── home.html
    │   ├── prediction_app.html
    │   └── regression_app.html
    ├── saved_models_usage/
    │   ├── model_Drug_1.joblib
    │   ├── model_Drug_2.joblib
    │   └── ...
    └── trained_models/
        ├── best_model_Impulsive_LinearRegression.joblib
        └── best_model_SS_RandomForest.joblib
    └── third_model/
        └── psychoactive_drug_model.joblib
    ```

## 🏃‍♀️ Usage

1.  **Run the Flask application:**

    ```
    python app.py
    ```

    The application will typically start on `http://127.0.0.1:5000/`.

2.  **Navigate the application:**

    * Open your web browser and go to `http://127.0.0.1:5000/`.

    * From the home page, you can navigate to:

        * **Drug Usage & Personality Prediction:** Input personality scores and demographic details to get predictions.

        * **Psychoactive Drug Classification:** Explore placeholder information about drug classification (this page is currently for informational purposes).

3.  **Toggle Theme:** Use the "Dark Mode" / "Light Mode" button in the top right corner of any page to switch between themes. Your preference will be saved for future visits.

## 📁 Project Structure

* `app.py`: The main Flask application file. It handles routing, loads the ML models, processes user input, performs predictions, and renders HTML templates.

* `requirements.txt`: Lists Python dependencies required for the project.

* `templates/`: Contains all HTML template files for the web interface:

    * `home.html`: The main landing page.

    * `prediction_app.html`: The page where users input data for drug usage and personality trait prediction.

    * `regression_app.html`: The page for psychoactive drug classification.

* `saved_models_usage/`: Directory to store the first-stage (drug usage classification) models.

* `trained_models/`: Directory to store the second-stage (Impulsive and SS regression) models.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 📸 Screenshots (Coming Soon!)

Screenshots of the application in action will be added here to showcase its interface and functionality
