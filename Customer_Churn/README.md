## Predict Customer Churn

- Project **Predict Customer Churn** assigned under ML DevOps Engineer Nanodegree Udacity


## Project Description

The project is as example of good software engineering practices in Data science.<br/>
The **churn rate**, also known as the rate of attrition or customer churn, 
is the rate at which customers stop doing business with an entity. 
It is most commonly expressed as the percentage of service subscribers 
who discontinue their subscriptions within a given time period.<br/>
For more information about attrition check [here](https://www.investopedia.com/terms/c/churnrate.asp#:~:text=The%20churn%20rate%2C%20also%20known,within%20a%20given%20time%20period.) <br/>
Or read my medium blog on good software practices [here](https://harshitsati.medium.com/guide-for-effective-code-reviews-b1c5165432ae)<br/>
This projects aims at predicting if a customer will churn or not.
![EDA 1](https://github.com/HarshitSati/Projects/blob/main/Customer_Churn/images/eda/Attrition_Flag_distribution.jpg)
![EDA 2](https://github.com/HarshitSati/Projects/blob/main/Customer_Churn/images/eda/Income_Category_distribution.jpg)
![EDA 3](https://github.com/HarshitSati/Projects/blob/main/Customer_Churn/images/eda/Customer_Age_distribution.jpg)
![EDA 4](https://github.com/HarshitSati/Projects/blob/main/Customer_Churn/images/eda/Education_Level_distribution.jpg)

## Running Files

Run the churn_library.py to obtain all information which includes: <br/>
- EDA
- Model
- Classification report
- Logs
![report 1](https://github.com/HarshitSati/Projects/blob/main/Customer_Churn/images/results/Logistic_Regression_classification_report.jpg)
![report 2](https://github.com/HarshitSati/Projects/blob/main/Customer_Churn/images/results/Random_Forest_classification_report.jpg)
![report 3](https://github.com/HarshitSati/Projects/blob/main/Customer_Churn/images/results/XGBoost_classification_report.jpg)
### Logging and Testing
churn_script_logging_and_tests.py logs all the successes and failures 
that occur in the important functions of churn_library.py while testing the important functions.</br>
A normal run will results in the logs file to look like this
![logs](https://github.com/HarshitSati/Projects/blob/main/Customer_Churn/logs/SS_logs_normal.png)


