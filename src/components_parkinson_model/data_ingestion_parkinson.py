import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components_parkinson_model.data_transformation_parkinson import ParkinsonDataTransformation
from src.components_parkinson_model.data_transformation_parkinson import ParkinsonDataTransformationConfig

from src.components_parkinson_model.model_trainer_parkinson import ParkinsonModelTrainerConfig
from src.components_parkinson_model.model_trainer_parkinson import ParkinsonModelTrainer

@dataclass
class ParkinsonDataIngestionConfig:
    train_data_path: str=os.path.join('parkinson_artifact', "train.csv")
    test_data_path: str=os.path.join('parkinson_artifact', "test.csv")
    raw_data_path: str=os.path.join('parkinson_artifact', "data.csv")

class ParkinsonDataIngestion:
    def __init__(self):
        self.ingestion_config=ParkinsonDataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\parkinson notebook\data\parkinsons_data.csv')
            logging.info('Read the dataset as data frame')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2,random_state=2)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")


            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj=ParkinsonDataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = ParkinsonDataTransformation()
    train_arr, test_arr, _=data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ParkinsonModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))