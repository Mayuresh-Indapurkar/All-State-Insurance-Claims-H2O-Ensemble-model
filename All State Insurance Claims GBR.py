import pandas as pd
import numpy as np
import csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
import datetime

#Transform the variables by standardizing them and taking logarithm of train data. 
def funcTransformData(df, cross=True, scaler=None):
    df = df.drop('id', axis=1)
    df = funcCatCont(df)
    if cross:
        y = df.loss.values
        X = df.drop('loss', axis=1)
        X_train, X_cross, y_train, y_cross = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.funcTransformData(X_train)
        X_cross = scaler.funcTransformData(X_cross)
        y_train = np.log(y_train)
        return X_train, X_cross, y_train, y_cross, scaler
    else:
        X_test = funcCatCont(df)
        X_test = X_test.as_matrix()
        X_test = scaler.funcTransformData(X_test)
        return X_test

#Transform the categorical variables into continous variables.
def funcCatCont(df):
    for i in range(1, 117): # Get categorical columns
        col_name = "cat{}".format(i)
        df[col_name] = df[col_name].astype('category')
    cat_cols = df.select_dtypes(['category']).columns # Convert categorical to continous
    df[cat_cols] = df[cat_cols].apply(lambda x: x.cat.codes)
    return df

def funcPredict(X, gbr):
    return np.exp(gbr.predict(X))

if __name__ == '__main__':
    path = 'M:/DataScience/Masters/3.AppliedMultivariateAnalysis/Final-AllState/data/train.csv'
    df = pd.read_csv(path)    
    X_train, X_cross, y_train, y_cross, scaler = funcTransformData(df)
    gbr = GradientBoostingRegressor(n_estimators=250, learning_rate=0.1, max_depth=7,
                                    subsample=0.9, random_state=42, 
                                    loss='ls', verbose=2).fit(X_train, y_train)    
    mse = mean_squared_error(y_cross, funcPredict(X_cross, gbr))
    mae = mean_absolute_error(y_cross, funcPredict(X_cross, gbr))
    print("MSE  :   {}".format(mse))
    print("MAE  :   {}".format(mae))
    path = 'M:/DataScience/Masters/3.AppliedMultivariateAnalysis/Final-AllState/data/test.csv'
    df_test = pd.read_csv(path)
    id_test = df_test.id.values
    X_test = funcTransformData(df_test, cross=False, scaler=scaler)
    pred_test = funcPredict(X_test, gbr)
    op = []
    for i in range(0, len(pred_test)):
        op.append([id_test[i], pred_test[i]])
    with open('M:/DataScience/Masters/3.AppliedMultivariateAnalysis/Final-AllState/testOP/GBR1.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'loss'])
        for row in op:
            writer.writerow(row)