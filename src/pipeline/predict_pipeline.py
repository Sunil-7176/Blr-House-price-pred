import os
import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor1.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 locality:str,
                 facing : str,
                 parking : str,
                 BHK : int,
                 bathrooms : int,
                 area : float,
                 price_per_sqft : float):
        
        self.locality = locality
        self.facing = facing
        self.parking = parking
        self.BHK = BHK
        self.bathrooms = bathrooms
        self.area = area
        self.price_per_sqft = price_per_sqft

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                
                'locality' : [self.locality],
                'facing' : [self.facing],
                'parking' : [self.parking],
                'BHK' : [self.BHK],
                'bathrooms' : [self.bathrooms],
                'area' : [self.area],
                'price_per_sqft' : [self.price_per_sqft]
           }

            return pd.DataFrame(custom_data_input_dict)
    
        except Exception as e:
            raise CustomException(e,sys)
                 