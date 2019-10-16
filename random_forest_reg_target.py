import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns #drawing statistical graphics

from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
from sklearn import metrics 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from category_encoders import *

from sklearn.preprocessing import OneHotEncoder, StandardScaler

#read in files - df = training data; dfTest = test data
df = pd.read_csv(r"C:\Users\lily\Documents\SS\ML\IndividualComp\tcdml1920-income-ind\training_with_labels.csv")
dfTest = pd.read_csv(r"C:\Users\lily\Documents\SS\ML\IndividualComp\tcdml1920-income-ind\tcd ml 2019-20 income prediction test (without labels).csv")

#training data (features) X = all data except for income
X = df.drop('Income in EUR', axis=1)
#training data (target) y = income 
y = df['Income in EUR'].values

#splitting data from training file into train and test data to check RMSE locally 
#X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.8)

#test data (features) to be used in predicting submission values
X_test = dfTest.drop('Income', axis=1)

#numerical data - num_feat = columns which are numerical; num_trans = what is to be done to them in 
#preprocessing, i.e. imputer for handling missing values (default is mean); standardscaler to standardize values 
num_feat = ['Year of Record', 'Age', 'Size of City', 'Wears Glasses', 'Body Height [cm]']
num_trans = Pipeline(steps=[('imputer', SimpleImputer()), ('scaler', StandardScaler())])

#categorical data - cat_feat = columns with categorical data; cat_trans = preprocessing, i.e.
#target encoding (from category_encoders) and then simpleimputer to fill missing values
#initially: OneHotEncoder + LinearRegression
cat_feat = ['Gender', 'Country', 'Profession', 'University Degree', 'Hair Color']
cat_trans = Pipeline(steps=[('encoder', TargetEncoder()),('imputer', SimpleImputer(strategy='mean'))])

#define preprocessing method using above specifications
preprocessor = ColumnTransformer(transformers=[('num', num_trans, num_feat), ('cat', cat_trans, cat_feat)])

#pipeline with preprocessing and then application of adaboostregressor with randomforestregressor as base estimator
rf = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', AdaBoostRegressor(RandomForestRegressor(random_state=0)))])

#preprocessing + regressor (i.e. what we put into the pipeline) to training data
#rf.fit(X_train, y_train)
rf.fit(X, y)

#preprocessing + predicting using pipeline defined processes - test features; y_pred = predicted
#value to be submitted
y_pred = rf.predict(X_test)

#create dataframe from submission file, set income column to be predicted values and write to a new file (or could have overwritten)
dfSub = pd.read_csv(r"C:\Users\lily\Documents\SS\ML\IndividualComp\tcdml1920-income-ind\submissions\3\tcd ml 2019-20 income prediction submission file.csv")
dfSub['Income'] = y_pred
#print(dfTest.head(25))
dfSub.to_csv(r"C:\Users\lily\Documents\SS\ML\IndividualComp\tcdml1920-income-ind\submissions\3\example.csv", index=False)

#when training file data was split to train and test, this printed the RMSE
#print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))