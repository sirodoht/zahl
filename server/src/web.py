from flask import Flask, render_template, request, jsonify

from ml import get_predictions


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/data", methods=['POST'])
def data():
    content = request.json
    dataURI = content["dataURI"]
    pred = get_predictions(dataURI)
    return jsonify({
        "predictionArray": pred["prediction_list"],
        "topPrediction": pred["top_prediction"],
    })
