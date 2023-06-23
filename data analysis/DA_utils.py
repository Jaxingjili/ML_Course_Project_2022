import numpy as np
import pandas as pd

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

