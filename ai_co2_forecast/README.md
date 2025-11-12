# AI for Climate Action: Forecasting CO₂ Emissions Using Machine Learning

## Project Overview

This project aims to predict future CO₂ emissions of a country using historical data, enabling policymakers to make data-driven environmental decisions aligned with SDG 13 (Climate Action).

## Tech Stack

- Python (main language)
- Pandas, NumPy, Matplotlib, Scikit-learn
- Jupyter Notebook or VS Code with Python extension

## Dataset

- UN SDG / World Bank open CO₂ dataset (stored in `data/co2_emissions.csv`)

## Project Structure

- `data/`: Contains the dataset files
- `main.py`: Main Python script for data processing and model training
- `model.ipynb`: Jupyter Notebook for exploratory data analysis and modeling
- `README.md`: Project documentation

## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. Install required libraries:
   ```
   pip install pandas numpy matplotlib scikit-learn jupyter
   ```

## Usage

Run the main script:
```
python main.py
```

Or open the Jupyter Notebook:
```
jupyter notebook model.ipynb
```

## Phase 1 - Project Setup

This is the initial setup phase. Future phases will include data preprocessing, model development, and evaluation.
