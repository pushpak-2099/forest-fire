from flask import Flask
from flask import request
from flask import render_template
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application


# importing the model
ridge_model=pickle.load(open('model/ridgereg.pkl','rb'))
scaler_model=pickle.load(open('model/scaler.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        new_scaled_data = scaler_model.transform(input_data)

        result=ridge_model.predict(new_scaled_data)

        return render_template('home.html',results=result[0])
    else:
        return render_template('home.html')


if __name__=="__main__":
    application.run(debug=True)
    