from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

modelo = joblib.load("walmart_model.pkl")


@app.route("/")
def home():
    return render_template(
        "formulario.html",
        prediction=None,
        datos=None
    )


@app.route("/predict", methods=["POST"])
def predict():

    datos = request.form

    fecha = pd.to_datetime(datos["Date"])

    entrada = pd.DataFrame([{
        "Store": int(datos["Store"]),
        "Temperature": float(datos["Temperature"]),
        "Fuel_Price": float(datos["Fuel_Price"]),
        "CPI": float(datos["CPI"]),
        "Unemployment": float(datos["Unemployment"]),
        "Year": fecha.year,
        "Month": fecha.month,
        "Day": fecha.day,
        "Week_of_Year": fecha.isocalendar().week
    }])

    prediccion = modelo.predict(entrada)

    return render_template(
        "formulario.html",
        prediction=round(float(prediccion[0]), 2),
        datos=datos
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)