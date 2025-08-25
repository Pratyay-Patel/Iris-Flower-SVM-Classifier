# 🌸 Iris Flower SVM Classifier – Flask Web App  

A simple web application that classifies Iris flowers into **Setosa**, **Versicolor**, or **Virginica** using an **SVM model** trained with scikit-learn.  

---

## 🚀 Features
- Train and save an **SVM classifier** on the Iris dataset  
- Flask-powered backend with HTML frontend  
- Input validation and error handling  
- Instant prediction via a clean web interface  

---

## 📂 Project Structure
```
Iris-Flower-SVM-Classifier/
│── app.py                 # Flask backend
│── iris_svm_classifier.py # Training script
│── svm_model.pkl          # Trained SVM model
│── scaler.pkl             # Scaler for preprocessing
│── target_names.pkl       # Target labels
│── templates/
│    └── index.html        # Frontend UI
│── static/                # (Optional: CSS/JS/Images)
│── README.md              # Documentation
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Iris-Flower-SVM-Classifier.git
cd Iris-Flower-SVM-Classifier

# Create and activate venv
python -m venv venv
venv\Scripts\activate    # (Windows)
source venv/bin/activate # (Linux/Mac)

# Install dependencies
pip install flask scikit-learn numpy matplotlib
```

---

## 🏋️ Training the Model
Run the training script to generate model files:

```bash
python iris_svm_classifier.py
```

This will create:
- `svm_model.pkl`
- `scaler.pkl`
- `target_names.pkl`

---

## 🌐 Running the Web App
Start the Flask server:

```bash
python app.py
```

Then open 👉 [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.  

---

## 🖥️ Usage

### ✅ Example of Valid Input
```
Sepal Length: 5.1
Sepal Width: 3.5
Petal Length: 1.4
Petal Width: 0.2
```
➡️ Prediction: **Iris-setosa**

### ❌ Example of Invalid Input
```
Sepal Length: 3.0
Sepal Width: 3.5
Petal Length: 5.0
Petal Width: 1.5
```
➡️ Error: *Invalid value for sepal length: 3.0. Must be between 4.0 and 8.5.*

---

## 📊 Dataset
The **Iris dataset** contains:
- 150 samples
- 4 features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- 3 classes:
  - *Setosa*
  - *Versicolor*
  - *Virginica*

---


