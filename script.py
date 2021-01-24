from flask import Flask, request, render_template
import flask
import os
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index2.html')

## Prediction
def ValuePredictor(to_predict_list):
    to_predict= np.array(to_predict_list).reshape(1,12)
    loaded_model= joblib.load('RandomForest_Model.pkl')
    result= loaded_model.predict(to_predict)
    return result[0]

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)

        if int(result) == 1:
            prediction = 'Income more than 50K'
        else:
            prediction = 'Income less that 50K'

        return render_template("result.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
