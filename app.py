from flask import Flask,request,render_template
import numpy as np 
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            age = int(request.form.get('age')),
            sex = int(request.form.get('sex')),
            is_smoking = int(request.form.get('is_smoking')),
            cigsPerDay = int(request.form.get('cigsPerDay')),
            BPMeds = int(request.form.get('BPMeds')),
            prevalentStroke =int(request.form.get('prevalentStroke')),
            prevalentHyp = int(request.form.get('prevalentHyp')),
            diabetes = float(request.form.get('diabetes')),
            totChol = float(request.form.get('totChol')),
            sysBP = float(request.form.get('sysBP')),
            diaBP = float(request.form.get('diaBP')),
            BMI = float(request.form.get('BMI')),
            heartRate = float(request.form.get('heartRate')),
            glucose = float(request.form.get('glucose'))
            
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        if results[0] == 1.0:
            return render_template('home.html',results = "You are at a Risk of Chronic heart diesease")
        elif results[0] == 0.0:
            return render_template('home.html',results = "Hooray!!! You are Fit and Fine")
        
    

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug= True)