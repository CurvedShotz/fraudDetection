# Fraud Detection Model

This project aims to build and evaluate a machine learning model for detecting fraudulent transactions.

## Project Structure

- `data/`: For raw datasets or small samples.
- `notebooks/`: For exploratory analysis and scratch notebooks.
- `src/`: Source code for the pipeline.
  - `__init__.py`
  - `preprocess.py`: Functions for data loading, cleaning, encoding, scaling.
  - `train.py`: Script to train Logistic Regression and Random Forest models.
  - `evaluate.py`: Script to evaluate saved models and generate reports.
- `models/`: For saved model files.
- `outputs/`: For evaluation plots and reports.
- `requirements.txt`: Project dependencies.
- `README.md`: This file.

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd fraud_detection_model
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Preprocessing**: Run the preprocessing script to clean and prepare the data.
   ```bash
   python src/preprocess.py
   ```
2. **Training**: Train the models.
   ```bash
   python src/train.py
   ```
3. **Evaluation**: Evaluate the trained models.
   ```bash
   python src/evaluate.py
   ```
