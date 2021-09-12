'''
Script to help check on all the important/complex functions in churn_library
Author: Harshit Sati
Date: 10-09-2021
'''
from os import path
import logging
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    logging.info("Testing import_data()")
    try:
        df = import_data("./data/BankChurners.csv")
        logging.info("SUCCESS : Testing import_data ")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    logging.info("Testing perform_eda()")
    eda_col = ['Income_Category', 'Education_Level',
               'Marital_Status', 'Card_Category',
               'Gender', 'Customer_Age',
               'Total_Trans_Ct', 'Attrition_Flag']
    for col in eda_col:
        try:
            assert path.exists("./images/eda/{}_distribution.jpg".format(col))
            logging.info("SUCCESS : {}_distribution.jpg present! ".format(col))
        except AssertionError as err:
            logging.error(
                "FAILIURE : Can't find image in path named {}_distribution.jpg ".format(col))
            raise err

    graph_types = ['heatmap', 'histplots']
    for graph in graph_types:
        try:
            assert path.exists("./images/eda/numerical_{}.jpg".format(graph))
            logging.info("SUCCESS : numerical_{}.jpg present!".format(graph))
        except AssertionError as err:
            logging.error(
                "FAILIURE : Can't find image in path named numerical_{}.jpg".format(graph))
            raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    feat_imp_model = ["Random_Forest", "Logistic_Regression", "XGBoost"]
    logging.info("Testing perform_feature_engineering()")
    for key in feat_imp_model:
        try:
            assert path.exists("./images/results/{}.jpg".format(key))
            logging.info("SUCCESS : {}.jpg present!".format(key))
        except AssertionError as err:
            logging.error("Image not found in path named {}.jpg".format(key))
            raise err


def test_train_models():
    '''
    test train_models
    '''
    logging.info("Testing perform_feature_engineering()")
    models = ["logistic", "rfc", "xgboost"]
    for model in models:
        try:
            assert path.exists("./models/{}_model.pkl".format(model))
            logging.info("SUCCESS : {}_model.pkl present!".format(model))
        except AssertionError as err:
            logging.error("Model not found in path named {}.jpg".format(model))
            raise err


if __name__ == "__main__":
    test_import(cl.import_data)
    test_eda()
    test_perform_feature_engineering()
    test_train_models()
