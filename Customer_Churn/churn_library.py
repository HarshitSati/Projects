'''
Project to predict which customers are likely to churn
Author : Harshit Sati
Date: 10-09-2021
'''

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            churn_df: pandas dataframe
    '''

    churn_df = pd.read_csv(pth)
    churn_df = churn_df.drop(columns=churn_df.columns[-2:])
    churn_df.replace({'Unknown': np.nan}, inplace=True)
    churn_df = churn_df.dropna()
    return churn_df


def perform_eda(churn_df):
    '''
    perform eda on churn_df and save figures to images folder
    input:
            churn_df: pandas dataframe

    output:
            None
    '''

    churn_df['Attrition_Flag'].value_counts().plot(kind='pie', shadow=True, explode=[
        0, 0.2], autopct='%.2f', figsize=(8, 6))
    plt.title('Ratio of customer')
    plt.savefig("./images/eda/Attrition_Flag_distribution.jpg")

    num_corr_mat = churn_df.corr()  # correlation matrix for numerical columns
    plt.figure(figsize=(16, 10))
    sns.heatmap(num_corr_mat, cmap='Reds', annot=True)
    plt.title('Numerical Correlation Matrix')
    plt.savefig("./images/eda/numerical_heatmap.jpg")

    churn_df.hist(figsize=(16, 16))
    plt.savefig("./images/eda/numerical_histplots.jpg")

    plot_columns = [
        'Income_Category',
        'Education_Level',
        'Marital_Status',
        'Card_Category',
        'Gender']
    for col in plot_columns:
        plt.figure(figsize=(8, 4))
        plt.title('{} with respect to customer churn'.format(col))
        sns.countplot(data=churn_df, x=col, hue='Attrition_Flag')

        plt.savefig("./images/eda/{}_distribution.jpg".format(col))
        plt.show()

    plt.figure(figsize=(16, 4))
    plt.title('Customer Age Distribution with respect to Customer Churn')
    sns.countplot(data=churn_df, x='Customer_Age', hue='Attrition_Flag')
    plt.grid(False)
    plt.savefig("./images/eda/Customer_Age_distribution.jpg")

    plt.figure(figsize=(30, 6))
    plt.title('Total Transaction Distribution with respect to Customer Churn')
    plt.xticks(rotation=-45)
    sns.countplot(data=churn_df, x='Total_Trans_Ct', hue='Attrition_Flag')
    plt.grid(False)
    plt.savefig("./images/eda/Total_Trans_Ct_distribution.jpg")


def encode_dataframe(churn_df):
    '''
    helper function to turn each categorical column into a numerical column with the
    help of OrdinalEncoder

    input:
            df: pandas dataframe
            categorical_columns: list of columns that contain categorical features

    output:
            df: pandas dataframe with Ordinal columns
    '''
    categorical_columns_selector = selector(dtype_include=object)
    categorical_columns = categorical_columns_selector(churn_df)
    categorical_preprocessor = OrdinalEncoder()
    churn_df[categorical_columns] = categorical_preprocessor.fit_transform(
        churn_df[categorical_columns])

    return churn_df


def perform_feature_engineering(churn_df):
    '''
    input:
              churn_df: pandas dataframe

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    target = churn_df['Attrition_Flag']
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    data = churn_df[keep_cols]

    numerical_preprocessor = StandardScaler()
    churn_df = numerical_preprocessor.fit_transform(data)
    # balancing the imbalanced class using SMOTE
    imbalanced_preprocessor = SMOTE(random_state=0)
    data, target = imbalanced_preprocessor.fit_resample(data, target)

    x_train, x_test, y_train, y_test = train_test_split(
        data, target, test_size=0.33, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(model, x_test, y_test):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            model (dict): contains all models as keys and thier test and train predictions as items
            x_test: test features
            y_test: test response values

    output:
             None
    '''

    for key in model:
        plot_confusion_matrix(model[key][0], x_test, model[key][1])

        plt.savefig("./images/results/{}_confusion_matrix.jpg".format(key))
        plot_roc_curve(model[key][0], x_test, model[key][1])

        plt.savefig("./images/results/{}_roc_curve.jpg".format(key))

        plt.figure(figsize=(8, 6))
        plt.text(0.4, 0.8,
                 "{} model classification report".format(key),
                 {'fontsize': 20}, fontproperties='monospace')
        plt.text(
            0.1, 0.1,
            classification_report(y_test, model[key][1]),
            {'fontsize': 20}, fontproperties='monospace')

        plt.axis("off")
        plt.savefig(
            "./images/results/{}_classification_report.jpg".format(key),
            bbox_inches="tight")


def feature_importance_plot(feat_imp_model, x_test, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            feat_imp_model (dict): model object containing feature_importances_
            x_test: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    for key in feat_imp_model:
        importance_metric = feat_imp_model[key]
        col_indices = np.argsort(importance_metric)[::-1]
        col_names = [x_test.columns[i] for i in col_indices]

        plt.figure(figsize=(8, 5))
        plt.title("{} Feature Importance".format(key))
        plt.xlabel("Importance")
        plt.barh(range(x_test.shape[1]), importance_metric[col_indices])
        plt.grid(False)
        _ = plt.yticks(range(x_test.shape[1]), col_names)
        plt.savefig(output_pth + "{}.jpg".format(key), bbox_inches='tight')
        plt.show()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    rfc = RandomForestClassifier()
    param_grid_rfc = {
        'n_estimators': [200, 700],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid_rfc, cv=5)
    cv_rfc.fit(x_train, y_train)
    y_test_preds_rfc = cv_rfc.predict(x_test)
    y_train_preds_rfc = cv_rfc.predict(x_train)

    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_test_preds_lr = lr.predict(x_test)
    y_train_preds_lr = lr.predict(x_train)

    xgb = XGBClassifier(eval_metric='mlogloss')
    xgb.fit(x_train, y_train)
    y_test_preds_xgb = xgb.predict(x_test)
    y_train_preds_xgb = xgb.predict(x_train)

    joblib.dump(lr, "./models/logistic_model.pkl")
    joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
    joblib.dump(xgb, "./models/xgboost_model.pkl")

    feat_imp_model = {
        "Random_Forest": cv_rfc.best_estimator_.feature_importances_,
        "Logistic_Regression": lr.coef_[-1],
        "XGBoost": xgb.feature_importances_
    }
    feature_importance_plot(feat_imp_model, x_test, "./images/results/")

    model = {
        "Random_Forest": [cv_rfc, y_test_preds_rfc, y_train_preds_rfc],
        "Logistic_Regression": [lr, y_test_preds_lr, y_train_preds_lr],
        "XGBoost": [xgb, y_test_preds_xgb, y_train_preds_xgb]
    }

    classification_report_image(model, x_test, y_test)


if __name__ == "__main__":
    churn_df = import_data('./data/BankChurners.csv')
    perform_eda(churn_df)
    encoded_churn_df = encode_dataframe(churn_df)
    x_train, x_test, y_train, y_test = perform_feature_engineering(encoded_churn_df)
    train_models(x_train, x_test, y_train, y_test)
