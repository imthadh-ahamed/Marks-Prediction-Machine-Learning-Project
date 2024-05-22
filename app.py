from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("linear_regression_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from the form
        mid_sem_marks = float(request.form["MSE"])
        attendance = float(request.form["Attendance"])

        # Make a prediction using the loaded model
        input_data = [[mid_sem_marks, attendance]]
        prediction = model.predict(input_data)

        # Pass the prediction value to the template
        return render_template("index.html", prediction=prediction[0])
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)
