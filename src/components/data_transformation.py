import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from scipy.stats import zscore

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor1.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def clean_data(self, df):
        # Drop 'title' column
        df.drop('title', axis=1, inplace=True)

        # Retain only numerical values in 'area' column
        df['area'] = df['area'].apply(lambda x: float(x.split(' ')[0].replace(',', '')) if isinstance(x, str) else x)

        # Modify 'rent' column
        def rent_column_modify(val):
            if isinstance(val, str):
                if 'Lacs' in val:
                    return float(val.split(' ')[0].split('/')[0].replace(',', '')) * 1e5
                else:
                    return float(val.split('/')[0].replace(',', ''))
            else:
                return val

        df['rent'] = df['rent'].apply(rent_column_modify)

        # Retain only numerical values in 'price_per_sqft' column
        df['price_per_sqft'] = df['price_per_sqft'].apply(lambda x: float(x.replace('₹', '').split(' ')[0].replace(',', '')) if isinstance(x, str) else x)

        # Retain only numerical values in 'BHK' columns
        df['BHK'] = df['BHK'].apply(lambda x: int(x.split(' ')[0].replace('+', '')))

    #     z_scores = zscore(df['bathrooms'])
    #     threshold = 3

    # # Replace outliers with the median value
    #     df['bathrooms'] = np.where(np.abs(z_scores) > threshold, df['bathrooms'].median(), df['bathrooms'])

        # Change 'Don't Know' entries in 'facing' column to NaN
        df['facing'] = df['facing'].apply(lambda x: x if x != "Don't Know" else np.nan)

        # Change 'None' entries in 'parking' column to NaN
        df['parking'] = df['parking'].apply(lambda x: np.nan if x == 'None' else x)

        return df

    def get_data_transformer_object(self):
        try:
            ordinal_features = ['BHK', 'bathrooms']
            categorical_features = ['locality', 'facing', 'parking']
            continuous_features = ['area','price_per_sqft']

            ordinal_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ("scaler",StandardScaler(with_mean=False))
            ])

            continuous_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            logging.info(f"Categorical Columns: {categorical_features}")
            logging.info(f"ordinal columnsL {ordinal_features}")
            logging.info(f"continous columns: {continuous_features}")

            preprocessor = ColumnTransformer(transformers=
                [
                    ('ordinal', ordinal_pipeline, ordinal_features),
                    ('categorical', categorical_pipeline, categorical_features),
                    ('continuous', continuous_pipeline, continuous_features)
                ]
                
            )

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)
        
    def inititate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Apply data cleaning
            train_df = self.clean_data(train_df)
            test_df = self.clean_data(test_df)

            logging.info("Reading of train and test data is completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "rent"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Appkying preprocessing on the training and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            # Find the index of the 'rent' column
            rent_column_index = list(train_df.columns).index('rent')

# Reorder columns to move 'rent' to the last column
            column_order = list(range(train_arr.shape[1]))
            column_order.remove(rent_column_index)  # Remove 'rent' from the original position
            column_order.append(rent_column_index)  # Add 'rent' to the last position

            train_arr = train_arr[:, column_order]
            test_arr = test_arr[:, column_order]

            logging.info("Saved preproccessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )



        except Exception as e:
            raise CustomException(e,sys)