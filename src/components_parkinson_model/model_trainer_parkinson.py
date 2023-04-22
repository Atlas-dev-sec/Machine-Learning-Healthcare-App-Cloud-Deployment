import os
import sys

from dataclasses import dataclass
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)


from sklearn.metrics import accuracy_score
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ParkinsonModelTrainerConfig:
    trained_model_file_path=os.path.join("parkinson_artifact", "model.pkl")


class ParkinsonModelTrainer:
    def __init__(self):
        self.model_trainer_config=ParkinsonModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models= {
                "Support Vector Machine": SVC(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }
            
            params = {
                "Support Vector Machine": {
                    'shrinking':[True, False]
                },
                "Decision Tree":{
                    'criterion':['gini', 'entropy', 'log_loss'],
                },
                "Random Forest":{
                    'n_estimators':[8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Classifier":{
                    'learning_rate':[.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }

            }

            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test, models=models, param = params)

            # to get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            #logging.info("The best model was: {}", best_model)
            best_model = models[best_model_name]
            print(best_model)

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            acc_score = accuracy_score(y_test, predicted)
            return acc_score
        
        except:
            pass