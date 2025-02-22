# PsychoactiveFusionML

PsychoactiveFusionML is a machine learning framework designed to analyze, classify, and predict the effects of psychoactive substances using various data-driven approaches. The project leverages deep learning, natural language processing (NLP), and statistical models to gain insights into psychoactive compounds, user experiences, and potential applications.

## Features
- **Data Processing**: Cleans and preprocesses datasets from multiple sources (research papers, experience reports, clinical studies, etc.).
- **Deep Learning Models**: Utilizes neural networks to classify substances based on effects, toxicity, and pharmacokinetics.
- **Natural Language Processing (NLP)**: Extracts meaningful insights from user experience reports and scientific literature.
- **Predictive Analytics**: Forecasts potential effects of novel compounds based on existing data.
- **Visualization**: Provides interactive visualizations for substance effect mapping.

## Installation
To install and set up PsychoactiveFusionML, follow these steps:

```bash
# Clone the repository
git clone https://github.com/Champion2049/PsychoactiveFusionML.git
cd PsychoactiveFusionML

# Create a virtual environment (optional but recommended)
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage
To run the main pipeline, execute:
```bash
python main.py --dataset data/substances.csv --model models/effect_classifier.pth
```
For additional options, use:
```bash
python main.py --help
```

## Directory Structure
```
PsychoactiveFusionML/
│── data/                # Datasets and processed data
│── models/              # Pre-trained and fine-tuned models
│── notebooks/           # Jupyter notebooks for experimentation
│── src/                 # Source code
│   ├── preprocessing.py # Data cleaning and preprocessing
│   ├── training.py      # Model training scripts
│   ├── inference.py     # Prediction and inference logic
│── main.py              # Main entry point
│── requirements.txt     # Project dependencies
│── README.md            # Documentation
```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add feature-name'`
4. Push to the branch: `git push origin feature-name`
5. Open a pull request.

## Contact
For questions or collaborations, please open an issue or reach out at `me.chirayu.6@gmail.com`.

---
Happy coding!

## Repository Link
[PsychoactiveFusionML on GitHub](https://github.com/Champion2049/PsychoactiveFusionML)