from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib


app = Flask(__name__)

modelo = joblib.load("walmart_model.pkl")


@app.route("/")
def home():

    return render_template("formulario.html")


@app.route("/predict", methods=["POST"])
def predict():


    datos = request.form


    entrada = pd.DataFrame([{

        "Store": int(datos["Store"]),
        "Temperature": float(datos["Temperature"]),
        "Fuel_Price": float(datos["Fuel_Price"]),
        "CPI": float(datos["CPI"]),
        "Unemployment": float(datos["Unemployment"]),
        "Year": int(datos["Year"]),
        "Month": int(datos["Month"]),
        "Day": int(datos["Day"]),
        "Week_of_Year": int(datos["Week_of_Year"])

    }])


    prediccion = modelo.predict(entrada)


    return render_template(
        "formulario.html",
        prediction=round(float(prediccion[0]),2)
    )



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)