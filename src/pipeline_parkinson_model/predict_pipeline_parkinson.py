import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class ParkinsonPredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'parkinson_artifact\model.pkl'
            preprocessor_path = 'parkinson_artifact\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class ParkinsonCustomData:
    def __init__(self, 
                 fo: int, 
                 fhi: int, 
                 flo: int,
                 jitter_percentage:int,
                 jitter_abs: int, 
                 rap:int,
                 ppq: int,
                 ddp: int,
                 shimmer: int,
                 shimmer_db: int,
                 shimmer_apq3: int,
                 shimmer_apq5: int,
                 mdvp: int,
                 shimmer_dda: int,
                 nhr: int,
                 hnr: int,
                 rpde: int,
                 dfa: int,
                 spread_one : int,
                 spread_two : int,
                 d2: int,
                 ppe : int 
                 ):
        self.fo = fo
        self.fhi = fhi 
        self.flo = flo
        self.jitter_percentage = jitter_percentage
        self.jitter_abs = jitter_abs 
        self.rap = rap 
        self.ppq = ppq
        self.ddp = ddp
        self.shimmer = shimmer
        self.shimmer_db = shimmer_db
        self.shimmer_apq3 = shimmer_apq3
        self.shimmer_apq5 = shimmer_apq5
        self.mdvp = mdvp
        self.shimmer_dda = shimmer_dda
        self.nhr = nhr
        self.hnr = hnr
        self.rpde = rpde
        self.dfa = dfa
        self.spread_one = spread_one
        self.spread_two = spread_two
        self.d2 = d2
        self.ppe = ppe 
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "MDVP:Fo(Hz)":[self.fo], 
                "MDVP:Fhi(Hz)":[self.fhi],  
                "MDVP:Flo(Hz)":[self.flo],
                "MDVP:Jitter(%)":[self.jitter_percentage], 
                "MDVP:Jitter(Abs)":[self.jitter_abs], 
                "MDVP:RAP":[self.rap], 
                "MDVP:PPQ":[self.ppq],
                "Jitter:DDP":[self.ddp],
                "MDVP:Shimmer":[self.shimmer],
                "MDVP:Shimmer(dB)":[self.shimmer_db],
                "Shimmer:APQ3":[self.shimmer_apq3], 
                "Shimmer:APQ5":[self.shimmer_apq5], 
                "MDVP:APQ":[self.mdvp], 
                "Shimmer:DDA":[self.shimmer_dda], 
                "NHR":[self.nhr],
                "HNR":[self.hnr],
                "RPDE":[self.rpde], 
                "DFA":[self.dfa], 
                "spread1":[self.spread_one], 
                "spread2":[self.spread_two], 
                "D2":[self.d2],
                "PPE":[self.ppe],  
                
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)