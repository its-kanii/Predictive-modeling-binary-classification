# Predictive Modeling - Binary Classification

## Task Overview
The goal of this task is to build and evaluate a binary classification model using a dataset of your choice. The task involves selecting an appropriate machine learning algorithm, evaluating its performance, and visualizing key metrics.

### Task Objectives:
1. Build a binary classification model using a dataset (e.g., Titanic, Heart Disease).
2. Select and apply an appropriate algorithm (e.g., Logistic Regression, Decision Trees).
3. Evaluate model performance using metrics such as:
   - Accuracy
   - Precision
   - Recall
   - ROC Curve
4. Visualize the results and performance metrics.

---

## Steps to Complete the Task

### 1. Dataset Selection
- Choose a dataset with binary target labels. Common options include:
  - Titanic Survival dataset
  - Heart Disease dataset
  - Any other binary classification dataset from repositories like Kaggle or UCI ML Repository.

### 2. Libraries and Tools
Ensure the following Python libraries are installed:
- **Pandas**: For data manipulation.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For machine learning algorithms and evaluation.
- **Matplotlib** & **Seaborn**: For visualizations.

Install them using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Analysis Process
#### a. **Data Preprocessing**
1. Load the dataset into a Pandas DataFrame:
   ```python
   import pandas as pd
   df = pd.read_csv('path_to_your_dataset.csv')
   ```
2. Handle missing values and encode categorical variables:
   ```python
   df.fillna(df.mean(), inplace=True)  # Example for handling missing values
   df = pd.get_dummies(df, drop_first=True)  # One-hot encoding
   ```
3. Split the data into features and target variable:
   ```python
   X = df.drop('target_column', axis=1)
   y = df['target_column']
   ```

#### b. **Train-Test Split**
Split the data into training and testing sets:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### c. **Model Selection and Training**
Train a binary classification model such as Logistic Regression or Decision Trees:
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

#### d. **Model Evaluation**
Evaluate the model using metrics like accuracy, precision, recall, and the ROC curve:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

#### e. **Visualizations**
Generate visualizations for metrics and performance:
- Confusion Matrix
- ROC Curve

Example:
```python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()
```

---

## Example Outputs
1. Confusion matrix plot.
2. ROC curve with the AUC score.
3. Printed metrics for accuracy, precision, and recall.

---

## Deliverables
1. Python code/script for building and evaluating the binary classification model.
2. Summary of key metrics and observations in a report or markdown file.
3. Visualizations for metrics and the ROC curve.

---
