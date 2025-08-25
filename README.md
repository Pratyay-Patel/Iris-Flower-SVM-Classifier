🌸 Iris Flower SVM Classifier – Flask Web App

This project is a simple Machine Learning + Web Application that classifies Iris flowers into three species (Setosa, Versicolor, Virginica) based on user inputs for sepal and petal dimensions.

It uses Support Vector Machine (SVM) for classification, a Flask backend for model serving, and a Bootstrap-powered frontend for interaction.

🚀 Features

Train an SVM classifier on the famous Iris dataset

Save and load model, scaler, and target labels using pickle

Web interface built with Flask + Bootstrap

Input validation with error messages for values outside dataset ranges

Displays predicted Iris species instantly

📂 Project Structure
Iris-Flower-SVM-Classifier/
│── app.py                # Flask backend
│── iris_svm_classifier.py # Training script
│── svm_model.pkl          # Trained SVM model
│── scaler.pkl             # Scaler for preprocessing
│── target_names.pkl       # Target labels
│── templates/
│    └── index.html        # Frontend UI
│── static/                # (Optional: CSS/JS/Images)
│── README.md              # Documentation
│── venv/                  # Virtual environment (not pushed to GitHub)

⚙️ Installation

Clone this repository:

git clone https://github.com/yourusername/Iris-Flower-SVM-Classifier.git
cd Iris-Flower-SVM-Classifier


Create a virtual environment:

python -m venv venv


Activate it:

On Windows (PowerShell):

venv\Scripts\activate


On Linux/Mac:

source venv/bin/activate


Install dependencies:

pip install flask scikit-learn numpy matplotlib

🏋️ Training the Model

Run the training script to generate the model and preprocessing files:

python iris_svm_classifier.py


This will create:

svm_model.pkl

scaler.pkl

target_names.pkl

🌐 Running the Web App

Start the Flask server:

python app.py


Then open your browser and go to:
👉 http://127.0.0.1:5000/

🖥️ Usage

Enter the flower’s measurements:

Sepal Length (4.0 – 8.5 cm)

Sepal Width (2.0 – 4.5 cm)

Petal Length (1.0 – 7.5 cm)

Petal Width (0.1 – 2.6 cm)

Click Predict

See the predicted Iris species or an error message if inputs are invalid.

🧪 Example Inputs

✅ Valid Input

Sepal Length: 5.1
Sepal Width: 3.5
Petal Length: 1.4
Petal Width: 0.2


➡️ Prediction: Iris-setosa

❌ Invalid Input

Sepal Length: 3.0
Sepal Width: 3.5
Petal Length: 5.0
Petal Width: 1.5


➡️ Error: Invalid value for sepal length: 3.0. Must be between 4.0 and 8.5.

📊 Dataset

The project uses the Iris dataset, a classic dataset in machine learning containing 150 samples of iris flowers with four features each:

Sepal Length

Sepal Width

Petal Length

Petal Width