import pandas as pd

# Load the dataset
data = pd.read_csv('GSE68086_TEP_data_matrix.csv', index_col=0)

# Transpose the dataset
data_transposed = data.T
print(data_transposed.head())
# Check for missing values
print(data_transposed.isnull().sum())

# Option 1: Drop rows with missing values
data_transposed = data_transposed.dropna()

# Option 2: Impute missing values (e.g., using mean)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data_transposed), columns=data_transposed.columns)
from sklearn.preprocessing import StandardScaler

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(data_imputed)
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=50)  # Reduce to 50 components
X_pca = pca.fit_transform(X_normalized)

# Visualize explained variance
import matplotlib.pyplot as plt
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.show()
from sklearn.model_selection import train_test_split
import numpy as np

# Create dummy labels (replace with actual labels if available)
y = np.random.randint(0, 2, size=X_pca.shape[0])  # Binary classification (0 or 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict on test data
y_pred = model.predict(X_test)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and AUC
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform Grid Search
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)
from sklearn.metrics import accuracy_score

# Evaluate initial model
y_pred_initial = model.predict(X_test)
print("Initial Model Accuracy:", accuracy_score(y_test, y_pred_initial))

# Evaluate tuned model
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)
print("Tuned Model Accuracy:", accuracy_score(y_test, y_pred_tuned))
# Extract feature importance
importances = model.feature_importances_

# Plot feature importance
plt.bar(range(len(importances)), importances)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()import joblib

# Save the model
joblib.dump(model, 'random_forest_model.pkl')

