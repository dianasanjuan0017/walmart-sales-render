from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

modelo = joblib.load("walmart_model.pkl")

@app.route("/")
def home():
    return "Modelo de predicción de ventas Walmart activo"

@app.route("/predict", methods=["POST"])
def predict():

    datos = request.get_json()

    entrada = pd.DataFrame([{
        "Store": datos["Store"],
        "Temperature": datos["Temperature"],
        "Fuel_Price": datos["Fuel_Price"],
        "CPI": datos["CPI"],
        "Unemployment": datos["Unemployment"],
        "Year": datos["Year"],
        "Month": datos["Month"],
        "Day": datos["Day"],
        "Week_of_Year": datos["Week_of_Year"]
    }])

    prediccion = modelo.predict(entrada)

    return jsonify({
        "Weekly_Sales_Predicted": round(float(prediccion[0]), 2)
    })

if __name__ == "__main__":
    app.run(debug=True,port=3000)