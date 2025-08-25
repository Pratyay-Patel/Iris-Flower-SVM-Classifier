from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load saved model, scaler, and target names
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("target_names.pkl", "rb") as f:
    target_names = pickle.load(f)

# Define valid input ranges based on Iris dataset
VALID_RANGES = {
    "sepal_length": (4.0, 8.5),
    "sepal_width": (2.0, 4.5),
    "petal_length": (1.0, 7.5),
    "petal_width": (0.1, 2.6)
}

def validate_inputs(values):
    keys = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for k, v in zip(keys, values):
        low, high = VALID_RANGES[k]
        if v < low or v > high:
            return f"Invalid value for {k.replace('_', ' ')}: {v}. Must be between {low} and {high}."
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Get input values from form
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

            values = [sepal_length, sepal_width, petal_length, petal_width]

            # Validate inputs
            error = validate_inputs(values)
            if error:
                return render_template("index.html", error=error)

            # Scale and predict
            scaled_values = scaler.transform([values])
            pred_class = model.predict(scaled_values)[0]
            prediction = target_names[pred_class]

        except ValueError:
            error = "Please enter valid numeric values."

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
