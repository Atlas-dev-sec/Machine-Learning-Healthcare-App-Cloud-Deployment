import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifact\model.pkl'
            preprocessor_path = 'artifact\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, pregnancies: int, 
                 glucose: int, 
                 bloodpressure: int,
                 skinthickness:int,
                 insulin: int, 
                 bmi:int,
                 diabetespedigreefunction: int,
                 age: int 
                 ):
        self.pregnancies = pregnancies
        self.glucose = glucose
        self.bloodpressure = bloodpressure
        self.skinthickness = skinthickness
        self.insulin = insulin
        self.bmi = bmi
        self.diabetespedigreefunction = diabetespedigreefunction
        self.age = age
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Pregnancies": [self.pregnancies],
                "Glucose": [self.glucose],
                "BloodPressure": [self.bloodpressure],
                "SkinThickness": [self.skinthickness],
                "Insulin": [self.insulin],
                "BMI": [self.pregnancies],
                "DiabetesPedigreeFunction": [self.pregnancies],
                "Age": [self.pregnancies],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)