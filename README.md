# Gene Expression ML Practice Project

This repository contains a simple machine-learning workflow applied to a high-dimensional gene expression dataset.  
The goal of this project is to practice preprocessing, dimensionality reduction, model training, and evaluation techniques on real-world numerical data.

**Important Note:**  
The dataset used here does not include classification labels.  
To demonstrate the full supervised ML pipeline, synthetic (random) labels were generated purely for practice and illustration.  
Model accuracy, ROC curves, and confusion matrices in this project are **only for workflow demonstration**, not scientific interpretation.

---

## Overview

This project demonstrates:

- Loading and exploring a large gene expression matrix  
- Handling missing values (dropping and imputation)  
- Feature scaling using `StandardScaler`  
- Dimensionality reduction with PCA  
- Training a Random Forest classifier  
- Evaluating performance metrics  
- Hyperparameter tuning using GridSearchCV  
- Visualizing results  
- Saving trained models with `joblib`  

The emphasis is on learning and implementing the machine-learning pipeline rather than biological analysis.

---

## Dataset

- Input: `GSE68086_TEP_data_matrix.csv` (gene expression matrix)  
- Used as a numerical high-dimensional dataset  
- No biological conclusions are drawn  
- Synthetic labels were created only to allow model training steps  

---

## Code Structure

- **Data Loading & Cleaning:** missing value handling, transposition  
- **Preprocessing:** imputation, normalization  
- **Dimensionality Reduction:** PCA (50 components)  
- **Modeling:** RandomForestClassifier  
- **Evaluation:** confusion matrix, ROC curve, accuracy  
- **Hyperparameter Tuning:** GridSearchCV  
- **Model Saving:** `random_forest_model.pkl`  

---

## How to Run

1. Place the dataset CSV file in the project directory.  
2. Install dependencies: pip install -r requirements.txt
3. Run the script: AI-ML_Codes.py
---

## Limitations

- Labels are randomly generated for demonstration.  
- Performance metrics do **not** represent real predictive ability.  
- PCA components do not map back to biological features.  
- This project is intended for ML practice only.

---

## Future Work

This repository is an early learning project.  
Future versions of similar workflows will include real labels, domain-specific preprocessing, and more advanced modeling approaches.

---





