#!/usr/bin/env python

import joblib
import os
from io import StringIO

import pandas as pd
import flask
from flask import Flask, Response

model_path = '/opt/ml/model'
model = joblib.load(os.path.join(model_path, "model.joblib"))

app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    return Response(response="\n", status=200)


@app.route("/invocations", methods=["POST"])
def predict():
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO(data)
        data = pd.read_csv(s, header=None)

        response = model.predict(data.loc[0:1].values)
        response = pd.DataFrame(response)
        response = response.to_csv(header=False, index=False)
    else:
        return flask.Response(response='CSV data only', status=415, mimetype='text/plain')

    return Response(response=response, status=200)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)