# Machine Learning-Based Classification of Exoplanets Using Orbital and Physical Parameters

## Overview

This project applies machine learning techniques to classify exoplanets by their discovery method using orbital and physical parameters. Three models are compared: Random Forest, XGBoost, and a Neural Network (Keras/TensorFlow).

## Dataset

- **Source:** NASA Exoplanet Archive (2025)
- **File:** `all_exoplanets_2025.csv`
- **Size:** 38,090 records, 100 columns (raw)
- **Selected Features:**
  - `orbital_period_days` — Orbital period in days
  - `planet_mass_earth_masses` — Planet mass in Earth masses
  - `equilibrium_temperature_kelvin` — Equilibrium temperature (K)
  - `insolation_flux_earth_1` — Insolation flux relative to Earth
- **Target Variable:** `discovery_method` (11 classes: Transit, Radial Velocity, Imaging, Microlensing, etc.)

## Project Structure

```
.
├── Exoplanet.ipynb        # Main analysis notebook
├── all_exoplanets_2025.csv # Exoplanet dataset
├── Main-Machine Learning-Based Classification of Exoplanets...docx  # Research paper
└── README.md
```

## Methodology

1. **Data Collection** — Loaded the NASA Exoplanet Archive dataset.
2. **Data Cleaning & Preprocessing**
   - Selected 12 relevant columns from 100.
   - Handled missing values using KNN Imputation (k=5) for numerical features.
   - Encoded categorical labels with `LabelEncoder`.
   - Standardized features with `StandardScaler`.
3. **Exploratory Data Analysis (EDA)**
   - Discovery method distribution
   - Exoplanet discoveries per year
   - Mass distribution (log scale)
   - Orbital period vs. mass scatter plot
   - Pairplot of key features
4. **Model Training** (80/20 train-test split)
   - **Random Forest Classifier** (100 estimators)
   - **XGBoost Classifier**
   - **Neural Network** — 2 hidden layers (64, 32 neurons), dropout regularization, softmax output
5. **Evaluation**
   - Accuracy, precision, recall, F1-score
   - Confusion matrix
   - Feature importance (Random Forest, XGBoost, SHAP for Neural Network)

## Results

| Model          | Accuracy |
|----------------|----------|
| XGBoost        | ~95.3%   |
| Neural Network | See notebook for full results |
| Random Forest  | See notebook for full results |

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- tensorflow / keras
- shap

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow shap
```

## Usage

1. Ensure `all_exoplanets_2025.csv` is in the project root.
2. Open and run `Exoplanet.ipynb` in Jupyter Notebook or JupyterLab.

```bash
jupyter notebook Exoplanet.ipynb
```

## License

This project is for academic and research purposes.
