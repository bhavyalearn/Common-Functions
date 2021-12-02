#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import basic packages

import warnings
warnings.filterwarnings("ignore")

import os
 
import pandas as pd               
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

#data preparation modules
from sklearn.model_selection import train_test_split   

# import all major classification models

from sklearn.linear_model import LinearRegression  
from sklearn.tree import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import svm

from sklearn.model_selection import GridSearchCV

from sklearn.metrics.regression import mean_squared_error 
from sklearn.metrics import mean_absolute_error


# In[17]:


#Load Data

def loadData(path, file,sheetName='Sheet1',sep=","):
    """
    Function to load data from Excel or CSV File. Will improve to load other formats in future
    takes following arguments:
    path - path of the file
    file - file name including extention
    sheetName - sheetname from which data is to be read. Default is 'Sheet1'. Not required for CSV file types
    sep - separater character to be used for CSV files. Default is comma
    returns a dataframe
    """
    
    if 'csv' in file:
        try:
            data = pd.read_csv(path+file,sep=sep)
        except FileNotFoundError:
            print("File not found")
            return None
        except PermissionError:
            print('No Permission to read the file')
            return None 
    elif 'xlsx' in file:
        try:
            data = pd.read_excel(path+file,sheet_name=sheetName)
        except FileNotFoundError:
            print("File not found")
            return None
         
        except PermissionError:
            print('No Permission to read the file')
            return None 
    else: 
        return ("Unknown file type")
    
    return data


def prepData(data,colsForOrder,categories,colsForDummy=None,removeNA=None):
    """
    Function to make data ready for processing by model
    Does the following tasks presently:
    1. creates dummy columns
    2. creates columns to have ordinal values
    3. removes NA values, using 'any' 
    """
    
    if removeNA != None:
        data.dropna(how='any',inplace=True)
    
    data.reset_index(inplace = True,drop = True)
             
    if colsForDummy != None:
        data = pd.get_dummies(data,columns = colsForDummy,prefix=colsForDummy)
        
    if colsForOrder != 'No':
        columns = colsForOrder
        categories_ = categories
        oe = OrdinalEncoder(categories=categories_)
        data[columns] = oe.fit_transform(data[columns])
   
    return data  
    


# def trainTestSplit(x,y,testSize = 0.25):
#     """
#     splits your 
#     """
#     x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=testSize,stratify=y)
#     return x_train,x_test,y_train,y_test

def scaleData(typeOfScaling,x_train,x_test):
    
    """
    Scales data using standard scaler or minmaxscaler. Takes three parameters.
    1. TypeOfScaling: Type of scaling to be done, Standard Scaler or MinMaxScaler
    2. x_train = Training data
    3. x_test = testing data
    
    Returns x_train and x_test
    
    """
    if typeOfScaling == 'Standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        return x_train, x_test
    elif typeOfScaling == 'MinMax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        return x_train, x_test
    else:
        return ("Scaling type not supported")
    

# def runModel(xtrain,ytrain):    
#     logreg = LogisticRegression(C=10,solver = gri,max_iter=2000,class_weight='balanced')    
#     model = logreg.fit(xtrain, ytrain)

#     return model
    
def testModel(modelType, modelName,xts,yts):
    """
    Tests models' outcome in terms of accuracy and F1 score for classifiers and mean_squared_error for regression
    algorithems. Supports Linear Regression, Logistic Regression, RandomForest and SVM at this time
    
    Returns Predited Y, Intercept, Coefficients, Accuracy and mean_squared_error for Linear Regression
    Returns Predited Y, Intercept, Coefficients, Accuracy and F1 Score for Classifier models
    
    ROC AUC curver to be added.
    
    modelType: use "LR" for Linear Regression, "LogReg",RF", "SVM" respectively for 
    Logistic Regression, Random Forest and SVM classifiers
    
    continue from here - 12/11/2021
    """
    test_results = []
    
    y_pred = model.predict(xts)
    accuracy = accuracy_score(yts, y_pred)
    f1_score_ = f1_score(yts,y_pred)
    
    test_result.append([y_pred,accuracy,f1_score])
    
    if modelType == 'Linear Regression':
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        test_result.append(rmse)

    return test_result

    
def heatMap(y_test,y_pred):
    plt.figure(figsize=(7,5))
    conf= confusion_matrix(y_test,y_pred)
    sns.heatmap(conf,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'],
                annot=True,fmt='d',cmap='GnBu')
    plt.show()
    
def getFeatureWeights(data,coef):
    """
    Creates a Panda's dataframe with weights of all the features of a model
    takes data and 
    """
    features = coef[0]
    columns = data.drop(['Attrition'],axis=1).columns
    parameters = pd.DataFrame(data = features, columns = ['Weight'],index = columns)
    return parameters