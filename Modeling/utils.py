import numpy as np
import pandas as pd
from utils import *
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from sklearn.base import clone
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

import torch                     # pytorch
import torch.nn as nn            # neural network components
import torch.optim as optim      # optimizers
import torch.nn.functional as F  # commonly used functions

#====================Functions for getting raw data=======================
path_head = '/Users/lixingji/Desktop/NYUSH/Term 5/Machine Learning/Project/data/'
# path head need to be changed to your running directry
def get_raw_train():
    path = path_head + 'train.csv'
    return pd.read_csv(path)

def get_raw_store():
    path = path_head + 'stores.csv'
    return pd.read_csv(path)

def get_raw_oil():
    path = path_head + 'oil.csv'
    return pd.read_csv(path)

def get_raw_holiday():
    path = path_head + 'holidays_events.csv'
    return pd.read_csv(path)

def get_raw_all():
    t = get_raw_train()
    s = get_raw_store()
    o = get_raw_oil()
    h = get_raw_holiday()
    return t,s,o,h



#===================fuctions to make cutoffs===============================
def make_cutoffs(df, train_day, test_day, stride = False):
    # input: df with date column, number of days used in each cutoff used for train / test
            # stride: number of days each cutoff jumps, by defalt it will equal to train_day
    # output: to df with cutoff lable, one for train one for test
    df['date'] = pd.to_datetime(df['date'])
    if not stride:
        stride = train_day
    use = df.sort_values(by = 'date')
    d1, d_stop = use['date'].tolist()[0], use['date'].tolist()[-1]
    d2 = d1 + pd.DateOffset(days=train_day)
    d3 = d2 + pd.DateOffset(days=test_day)
    train_df_lis, test_df_lis = [],[]
    while d3 <= d_stop:
        # get dfs for one cut off
        sub_train = df.loc[(df['date']>=d1)&(df['date']<d2)].copy()
        sub_test = df.loc[(df['date']>=d2)&(df['date']<d3)].copy()
        sub_train['cutoff'] = d1
        sub_test['cutoff'] = d1
        train_df_lis.append(sub_train)
        test_df_lis.append(sub_test)

        #update ds
        d1 = d1 + pd.DateOffset(days=stride)
        d2 = d1 + pd.DateOffset(days=train_day)
        d3 = d2 + pd.DateOffset(days=test_day)
    return pd.concat(train_df_lis), pd.concat(test_df_lis)

#=========== function use to run the model over cutoffs====================
def over_cutoffs(model, model_name , train_df, test_df, show_process = True, return_pred = True):
    cutoffs = list({i for i in train_df['cutoff']})
    track = []
    process = 0
    pred_lis = []
    for c in cutoffs:
        ## make a new model at each cutoff
        sub_model = clone(model)
        
        ## get training and testing data for the cutoff
        train_use = train_df.loc[train_df['cutoff'] == c].drop(columns = 'cutoff')
        test_use = test_df.loc[test_df['cutoff'] == c].drop(columns = 'cutoff')
        date_df = pd.concat([train_use[['date']], test_use[['date']]]).drop_duplicates()
        date_df['time'] = [i for i in range(date_df.shape[0])]
        
        train_y = train_use['sales'].values
        train_x = train_use.merge(date_df, on = 'date').drop(columns = ['sales','date']).values
        
        test_y = test_use['sales'].values
        test_x = test_use.merge(date_df, on = 'date').drop(columns = ['sales','date']).values
        
        ## fit the model with training data for this cutoff
        model.fit(train_x, train_y)
        
        ## get the predicted values of test data
        pred_y = model.predict(test_x)

        ## evaluate the result by MSE
        mse = mean_squared_error(test_y,pred_y)
        
        ## append the result to the track list
        track.append({
            'cutoff':c,
            'model':model_name,
            'MSE':mse
        })
        
        ## show the process if show_process
        if show_process:
            process += 1
            if process % 5 == 0:
                print(f'{process} cutoffs processed')
    return pd.DataFrame(track)






#===================useful check functions=================================
def check_type(df, col = None):
    # get the data type of all columns 
    # or just for the specific column(s)
    # col can be a list or a str
    if type(col) == type('str'):
        t = {type(i) for i in df[col]}
        return pd.DataFrame([{'col':col, 'type':t}])
    elif col:
        return pd.concat([check_type(df, col = c) for c in col])
    else:
        col = [i for i in df.columns]
        return check_type(df,col)

