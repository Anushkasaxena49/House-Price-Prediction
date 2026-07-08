from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


with open("house_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["CRIM"]),
            float(request.form["ZN"]),
            float(request.form["INDUS"]),
            float(request.form["CHAS"]),
            float(request.form["NOX"]),
            float(request.form["RM"]),
            float(request.form["AGE"]),
            float(request.form["DIS"]),
            float(request.form["RAD"]),
            float(request.form["TAX"]),
            float(request.form["PTRATIO"]),
            float(request.form["B"]),
            float(request.form["LSTAT"])
        ]

        prediction = model.predict(np.array([features]))[0]

        return render_template(
            "index.html",
            prediction_text=f" Predicted House Price: ${prediction:.2f}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {e}"
        )

if __name__ == "__main__":
    app.run(debug=True)
