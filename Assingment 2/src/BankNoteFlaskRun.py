#
# Author: Jamey Johnston
# Title: Deploy Flask App to Docker
# Date: 2020/01/30
# Email: jameyj@tamu.edu
# Texas A&M University - MS in Analytics - Mays Business School
#

from flask import Flask, request, redirect, url_for, flash, jsonify
import json, pickle
import pandas as pd
import numpy as np
import os
import xgboost
from settings import APP_STATIC

# To test locally in Anaconda Powershell prompt in your conda env
# 
# 1.) $env:FLASK_APP="RedWineQualityFlaskRun:app"
# 2.) flask run

app = Flask(__name__)


# Simple "Hello World" for root
@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/banknote', methods=['POST'])
def makecalc():
    """
    Function run at each API call
    """
    modelfile = 'BanknoteForged.pickleRF.dat'
    model = pickle.load(open(os.path.join(APP_STATIC, modelfile), 'rb'))

    jsonfile = request.get_json()
    data = pd.read_json(json.dumps(jsonfile), orient='index')
    print(data)

    res = dict()

    # Headers of Data
    # "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"

    # create array from JSON data from service
    X = np.array(data[['variance', "skewness", "kurtosis", "entropy"]])

    print(X)

    ypred = model.predict(X)

    for i in range(len(ypred)):
        res[i] = ypred[i]

    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8081)
