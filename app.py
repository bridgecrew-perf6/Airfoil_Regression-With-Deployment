import pickle
from flask import Flask, request, app, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import sklearn

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json['data']
    print(data)
    newdata = [list(data.values())]
    output = model.predict(newdata)[0]
    return jsonify(output)


@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)
    output = model.predict([data])[0]

    return render_template("home.html", prediction_text="Airfoil pressure is  {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)
