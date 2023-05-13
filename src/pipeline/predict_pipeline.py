import sys
import pandas as  pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            print(e)
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                 age: int,sex: int,is_smoking: int,cigsPerDay: int,BPMeds: int,
                 prevalentStroke: int,prevalentHyp: int,diabetes: int,totChol: int,
                 sysBP: int,diaBP: int,BMI: int,heartRate: int,glucose: int):
        
        self.age = age
        self.sex = sex
        self.is_smoking = is_smoking
        self.cigsPerDay = cigsPerDay
        self.BPMeds = BPMeds
        self.prevalentStroke = prevalentStroke
        self.prevalentHyp = prevalentHyp
        self.diabetes = diabetes
        self.totChol = totChol
        self.sysBP = sysBP
        self.diaBP = diaBP
        self.BMI = BMI
        self.heartRate = heartRate
        self.glucose = glucose

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'age':[self.age],
                'sex':[self.sex],
                'is_smoking':[self.is_smoking],
                'cigsPerDay':[self.cigsPerDay],
                'BPMeds':[self.BPMeds],
                'prevalentStroke':[self.prevalentStroke],
                'prevalentHyp':[self.prevalentHyp],
                'diabetes':[self.diabetes],
                'totChol':[self.totChol],
                'sysBP':[self.sysBP],
                'diaBP':[self.diaBP],
                'BMI':[self.BMI],
                'heartRate':[self.heartRate],
                'glucose':[self.glucose]

            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            print(e)
            raise CustomException(e,sys)