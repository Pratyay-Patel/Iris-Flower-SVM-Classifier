"""
Iris Flower Classifier using Support Vector Machine (SVM)

This script trains an SVM model on the classic Iris dataset
to classify flowers into three species: Setosa, Versicolor, and Virginica.
"""

# iris_svm_classifier.py

import pickle
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# 1. Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train SVM model
svm = SVC(kernel="linear", random_state=42)
svm.fit(X_train_scaled, y_train)

# 5. Predictions
y_pred = svm.predict(X_test_scaled)

# 6. Evaluation
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("Confusion Matrix:\n")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# 7. Save model, scaler, and target names for Flask app
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("target_names.pkl", "wb") as f:
    pickle.dump(iris.target_names, f)

print("✅ Model, scaler, and target names saved successfully!")
