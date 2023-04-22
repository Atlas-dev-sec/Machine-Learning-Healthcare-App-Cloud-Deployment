from flask import Flask,request,render_template,url_for
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline_heart_model.predict_pipeline_heart import HeartCustomData, HeartPredictPipeline
from src.pipeline_parkinson_model.predict_pipeline_parkinson import ParkinsonCustomData, ParkinsonPredictPipeline

application=Flask(__name__)

app = application

##Route for a home page
@app.route('/')
def index():
    return render_template('public/index.html')

@app.route('/predictdiabetes', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('public/diabetes-prediction.html')
    else:
        data=CustomData(
            pregnancies = int(request.form.get('pregnancies')),
            glucose = float(request.form.get('glucose')),
            bloodpressure = float(request.form.get('bloodpressure')),
            skinthickness = float(request.form.get('skinthickness')),
            insulin = float(request.form.get('insulin')),
            bmi = float(request.form.get('bmi')),
            diabetespedigreefunction = float(request.form.get('diabetespedigreefunction')),
            age = int(request.form.get('age'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(pred_df)
        return render_template('public/diabetes-prediction.html', results=results[0])
    
    
@app.route('/predictparkinson', methods=['GET','POST'])
def predict_parkinson_datapoint():
    if request.method == 'GET':
        return render_template('public/parkinson-prediction.html')
    else:
        data=ParkinsonCustomData(
                 fo = float(request.form.get('fo')),
                 fhi = float(request.form.get('fhi')),
                 flo= float(request.form.get('flo')),
                 jitter_percentage= float(request.form.get('jitter_percentage')),
                 jitter_abs = float(request.form.get('jitter_abs')),
                 rap= float(request.form.get('rap')),
                 ppq= float(request.form.get('ppq')),
                 ddp= float(request.form.get('ddp')),
                 shimmer= float(request.form.get('shimmer')),
                 shimmer_db= float(request.form.get('shimmer_db')),
                 shimmer_apq3= float(request.form.get('shimmer_apq3')),
                 shimmer_apq5= float(request.form.get('shimmer_apq5')),
                 mdvp= float(request.form.get('mdvp')),
                 shimmer_dda= float(request.form.get('shimmer_dda')),
                 nhr= float(request.form.get('nhr')),
                 hnr= float(request.form.get('hnr')),
                 rpde= float(request.form.get('rpde')),
                 dfa= float(request.form.get('dfa')),
                 spread_one = float(request.form.get('spread_one')),
                 spread_two= float(request.form.get('spread_two')), 
                 d2= float(request.form.get('d2')),
                 ppe = float(request.form.get('ppe')), 
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline = ParkinsonPredictPipeline()

        results = predict_pipeline.predict(pred_df)
        return render_template('public/parkinson-prediction.html', results=results[0])



@app.route('/predictheart', methods=['GET','POST'])
def predict_heart_datapoint():
    if request.method=='GET':
        return render_template('public/heart-prediction.html')
    else:
        data=HeartCustomData(
            age = int(request.form.get('age')),
            sex = int(request.form.get('sex')),
            cp = float(request.form.get('cp')),
            trestbps = float(request.form.get('trestbps')),
            chol = float(request.form.get('chol')),
            fbs = float(request.form.get('fbs')),
            restecg = float(request.form.get('restecg')),
            thalach = float(request.form.get('thalach')),
            exang = int(request.form.get('exang')),
            oldpeak = int(request.form.get('oldpeak')),
            slope = int(request.form.get('slope')),
            ca = int(request.form.get('ca')),
            thal = int(request.form.get('thal'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline = HeartPredictPipeline()

        results = predict_pipeline.predict(pred_df)
        return render_template('public/heart-prediction.html', results=results[0])



@app.route('/about', methods=['GET','POST'])
def about():
    return render_template('public/about.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)