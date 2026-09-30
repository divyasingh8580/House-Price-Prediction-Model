from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/house_price_model.pkl")

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            values = [float(request.form[feature]) for feature in FEATURES]
            data = np.array(values).reshape(1, -1)

            predicted_value = model.predict(data)[0]

            # Dataset target is in units of $100,000.
            dollars = predicted_value * 100000

            prediction = f"${dollars:,.0f}"

        except Exception as e:
            error = "Please enter valid numeric values for all fields."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
