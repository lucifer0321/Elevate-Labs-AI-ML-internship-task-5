# Elevate-Labs-AI-ML-internship-task-5
Implemented Decision Tree and Random Forest classifiers on the Heart Disease dataset. Includes data preprocessing, model training, feature importance analysis, cross-validation, and tree visualization. Compared accuracy and overfitting between both models using Scikit-learn.

#description
We used the Heart Disease dataset and applied Decision Tree and Random Forest classifiers using Scikit-learn. The data was split into training and testing sets, and we trained both models to predict heart disease. We visualized the Decision Tree, evaluated both models using accuracy and classification reports, analyzed feature importance, and used cross-validation to validate performance.

#code
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

url = "/content/archive.zip"
df = pd.read_csv(url)
df.head()
X = df.drop(columns='target')  # features
y = df['target']               # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtree = DecisionTreeClassifier(max_depth=4, random_state=42)
dtree.fit(X_train, y_train)

y_pred_tree = dtree.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
dtree = DecisionTreeClassifier(max_depth=4, random_state=42)
dtree.fit(X_train, y_train)

y_pred_tree = dtree.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_tree))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Feature Importance - Random Forest")
plt.show()
cv_scores = cross_val_score(rf, X, y, cv=5)
print("Cross-Validation Accuracy (Random Forest):", np.mean(cv_scores))
