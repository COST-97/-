import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
from sklearn.decomposition import PCA
import xgboost as xgb
from xgboost import XGBClassifier

pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')


def group_feature(df, key, target, aggs):
    agg_dict = {}
    for ag in aggs:
        agg_dict[f'{target}_{ag}'] = ag
    print(agg_dict)
    t = df.groupby(key)[target].agg(agg_dict).reset_index()
    return t


# def extract_feature(df, train):
#     t = group_feature(df, 'ship', 'x', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
#     train = pd.merge(train, t, on='ship', how='left')
#
#     t = group_feature(df, 'ship', 'x', ['count'])
#     train = pd.merge(train, t, on='ship', how='left')
#
#     t = group_feature(df, 'ship', 'y', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
#     train = pd.merge(train, t, on='ship', how='left')
#
#     t = group_feature(df, 'ship', 'v', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
#     train = pd.merge(train, t, on='ship', how='left')
#
#     t = group_feature(df, 'ship', 'd', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
#     train = pd.merge(train, t, on='ship', how='left')
#
#     p_all = np.concatenate((np.arange(0.05,0.5,0.05),np.arange(0.55,1,0.05)))
#     for p in p_all:
#         t = df.groupby('ship')['x'].agg({'x_%f'%int(100*p) : lambda x: x.quantile(p)}).reset_index()
#         train = pd.merge(train, t, on='ship', how='left')
#         t = df.groupby('ship')['y'].agg({'y_%f'%int(100*p) : lambda x: x.quantile(p)}).reset_index()
#         train = pd.merge(train, t, on='ship', how='left')
#         t = df.groupby('ship')['v'].agg({'v_%f'%int(100*p) : lambda x: x.quantile(p)}).reset_index()
#         train = pd.merge(train, t, on='ship', how='left')
#         t = df.groupby('ship')['d'].agg({'d_%f'%int(100*p) : lambda x: x.quantile(p)}).reset_index()
#         train = pd.merge(train, t, on='ship', how='left')
#
#     t = group_feature(df, 'ship', 'x_preq', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
#     train = pd.merge(train, t, on='ship', how='left')
#
#     t = df.groupby('ship')['x_preq'].agg({'x_preq_argmax': lambda x: np.argmax(x)}).reset_index()
#     train = pd.merge(train, t, on='ship', how='left')
#
#     t = group_feature(df, 'ship', 'y_preq', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
#     train = pd.merge(train, t, on='ship', how='left')
#     t = df.groupby('ship')['y_preq'].agg({'y_preq_argmax': lambda x: np.argmax(x)}).reset_index()
#     train = pd.merge(train, t, on='ship', how='left')
#
#     t = group_feature(df, 'ship', 'v_preq', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
#     train = pd.merge(train, t, on='ship', how='left')
#     t = df.groupby('ship')['v_preq'].agg({'v_preq_argmax': lambda x: np.argmax(x)}).reset_index()
#     train = pd.merge(train, t, on='ship', how='left')
#
#     t = group_feature(df, 'ship', 'd_preq', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
#     train = pd.merge(train, t, on='ship', how='left')
#     t = df.groupby('ship')['d_preq'].agg({'d_preq_argmax': lambda x: np.argmax(x)}).reset_index()
#     train = pd.merge(train, t, on='ship', how='left')
#
#     train['x_max_x_min'] = train['x_max'] - train['x_min']
#     train['x_max_x_min_'] = train['x_max'] + train['x_min']
#     train['x_max_y_max'] = train['x_max'] - train['y_max']
#     train['x_max_y_max_'] = train['x_max'] + train['y_max']
#     train['x_max_y_min'] = train['x_max'] - train['y_min']
#     train['x_max_y_min_'] = train['x_max'] + train['y_min']
#
#     train['y_max_y_min'] = train['y_max'] - train['y_min']
#     train['y_max_y_min_'] = train['y_max'] + train['y_min']
#     train['y_max_x_max'] = train['y_max'] - train['x_max']
#     train['y_max_x_max_'] = train['y_max'] + train['x_max']
#     train['y_max_x_min'] = train['y_max'] - train['x_min']
#     train['y_max_x_min_'] = train['y_max'] + train['x_min']
#
#     train['x_y_length_sum'] = train['x_max_x_min']+train['y_max_y_min']
#     train['diagonal_length'] = np.sqrt(train['x_max_x_min']**2+train['y_max_y_min']**2)
#     train['x_y_sum'] = train['x_min'] + train['x_max'] + train['y_min']+train['y_max']
#
#     train['slope'] = train['y_max_y_min'] / np.where(train['x_max_x_min'] == 0, 0.001, train['x_max_x_min'])
#     train['area'] = train['x_max_x_min'] * train['y_max_y_min']
#
#     train['x_0_25'] = (train['x_min'] + train['x_mean'])/2
#     train['x_0_12_5'] = (train['x_min'] + train['x_0_25']) / 2
#     train['x_0_37_5'] = (train['x_mean'] + train['x_0_25']) / 2
#
#     train['y_0_25'] = (train['y_min'] + train['y_mean']) / 2
#     train['y_0_12_5'] = (train['y_min'] + train['y_0_25']) / 2
#     train['y_0_37_5'] = (train['y_mean'] + train['y_0_25']) / 2
#
#     train['x_0_75'] = (train['x_max'] + train['x_mean']) / 2
#     train['x_0_62_5'] = (train['x_mean'] + train['x_0_75']) / 2
#     train['x_0_87_5'] = (train['x_max'] + train['x_0_75']) / 2
#
#     train['y_0_75'] = (train['y_max'] + train['y_mean']) / 2
#     train['y_0_62_5'] = (train['y_mean'] + train['y_0_75']) / 2
#     train['y_0_87_5'] = (train['y_max'] + train['y_0_75']) / 2
#
#     train['v_0_25'] = (train['v_min'] + train['v_mean'])/2
#     train['v_0_75'] = (train['v_max'] + train['v_mean']) / 2
#
#     train['d_0_25'] = (train['d_min'] + train['d_mean'])/2
#     train['d_0_75'] = (train['d_max'] + train['d_mean']) / 2
#
#     mode_hour = df.groupby('ship')['hour'].agg(lambda x: x.value_counts().index[0]).to_dict()
#     train['mode_hour'] = train['ship'].map(mode_hour)
#
#     t = group_feature(df, 'ship', 'hour', ['max', 'min'])
#     train = pd.merge(train, t, on='ship', how='left')
#
#     hour_nunique = df.groupby('ship')['hour'].nunique().to_dict()
#     date_nunique = df.groupby('ship')['date'].nunique().to_dict()
#     train['hour_nunique'] = train['ship'].map(hour_nunique)
#     train['date_nunique'] = train['ship'].map(date_nunique)
#
#     t = df.groupby('ship')['time'].agg({'diff_time': lambda x: np.max(x) - np.min(x)}).reset_index()
#     t['diff_day'] = t['diff_time'].dt.days
#     t['diff_second'] = t['diff_time'].dt.seconds
#     train = pd.merge(train, t, on='ship', how='left')
#     # print(train)
#     # input()
#     return train
def extract_feature(df, train):
    t = group_feature(df, 'ship', 'x', ['max', 'min', 'mean', 'var','std', 'skew', 'sum', 'count'])
    train = pd.merge(train, t, on='ship', how='left')

    # t = group_feature(df, 'ship', 'x', ['count'])
    # train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'y', ['max', 'min', 'mean','var', 'std', 'skew', 'sum','count'])
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'v', ['max', 'min', 'mean', 'var','std', 'skew', 'sum','count'])
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'v_sin', ['max', 'min', 'mean', 'var','std', 'skew', 'sum','count'])
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'v_cos', ['max', 'min', 'mean', 'var','std', 'skew', 'sum','count'])
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'd_tan', ['max', 'min', 'mean', 'var','std', 'skew', 'sum','count'])
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'd', ['max', 'min', 'mean', 'var','std', 'skew', 'sum','count'])
    train = pd.merge(train, t, on='ship', how='left')

    t = df.groupby('ship')['v_sin'].agg({'v_work_num': lambda x:x[x.values > 0.5].count()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_notwork_num':lambda x:(x[x.values <= 0.5]).count()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_work_mean': lambda x:  (x[x.values>0.5]).mean()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_work_var': lambda x:  (x[x.values>0.5]).var()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_work_std': lambda x: (x[x.values>0.5]).std()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_work_skew': lambda x: (x[x.values>0.5]).skew()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_work_sum': lambda x:  (x[x.values>0.5]).sum()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = df.groupby('ship')['v_cos'].agg({'v_work_num_cos': lambda x:x[x.values > 0.5].count()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_notwork_num_cos':lambda x:(x[x.values <= 0.5]).count()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_work_mean_cos': lambda x:  (x[x.values>0.5]).mean()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_work_var_cos': lambda x:  (x[x.values>0.5]).var()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_work_std_cos': lambda x: (x[x.values>0.5]).std()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_work_skew_cos': lambda x: (x[x.values>0.5]).skew()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_work_sum_cos': lambda x:  (x[x.values>0.5]).sum()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    # t = df.groupby('ship')['v'].agg({'v_work': lambda x: [v for v in x if v>0.5]}).reset_index()
    # t['v_work_mean'] = t['v_work'].mean()
    # t['v_work_std'] = t['v_work'].std()
    # train = pd.merge(train, t, on='ship', how='left')
    #
    # def work_num(x,work=True):
    #     work_nums = 0
    #     for v in x:
    #         if v>0.5:
    #             work_nums+=1
    #     if work:
    #         return work_nums
    #     else:
    #         return np.shape(x)[0]-work_nums

    t = df.groupby('ship')['v_sin'].agg({'v_notwork_mean': lambda x:  (x[x.values<=0.5]).mean()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_notwork_var': lambda x:  (x[x.values<=0.5]).var()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_notwork_std': lambda x: (x[x.values<=0.5]).std()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_notwork_skew': lambda x: (x[x.values <= 0.5]).skew()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_notwork_sum': lambda x:  (x[x.values<=0.5]).sum()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = df.groupby('ship')['v_cos'].agg({'v_notwork_mean_cos': lambda x:  (x[x.values<=0.5]).mean()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_notwork_var_cos': lambda x:  (x[x.values<=0.5]).var()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_notwork_std_cos': lambda x: (x[x.values<=0.5]).std()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_notwork_skew_cos': lambda x: (x[x.values <= 0.5]).skew()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_notwork_sum_cos': lambda x:  (x[x.values<=0.5]).sum()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    # t = df.groupby('ship')['v'].agg({'v_work': lambda x: [v for v in x if v>0.5]}).reset_index()
    # t['v_work_mean'] = t['v_work'].mean()
    # t['v_work_std'] = t['v_work'].std()
    # t['v_work_skew'] = t['v_work'].skew()
    # t['v_work_sum'] = t['v_work'].sum()
    # train = pd.merge(train, t, on='ship', how='left')
    #
    # t = df.groupby('ship')['v'].agg({'v_notwork': lambda x: [v for v in x if v<=0.5]}).reset_index()
    # t['v_notwork_mean'] = t['v_notwork'].mean()
    # t['v_notwork_std'] = t['v_notwork'].std()
    # t['v_notwork_skew'] = t['v_notwork'].skew()
    # t['v_notwork_sum'] = t['v_notwork'].sum()
    # train = pd.merge(train, t, on='ship', how='left')

    # def fea_fun(x):
    #     x = x[x.values>0.5]
        # vs = [v for v in x if v>0.5]
        # vs = pd.DataFrame(vs)
    # input()
    t = df.groupby('ship')['time'].agg({'diff_time': lambda x: np.max(x) - np.min(x)}).reset_index()
    t['diff_day'] = t['diff_time'].dt.days
    t['diff_second'] = t['diff_time'].dt.seconds
    train = pd.merge(train, t, on='ship', how='left')

    train['diff_second_025'] = train['diff_second']*0.25
    train['diff_second_05'] = train['diff_second'] * 0.5
    train['diff_second_075'] = train['diff_second'] * 0.75

    # p_all = np.concatenate((np.arange(0.05,0.5,0.05),np.arange(0.55,1,0.05)))
    p_all =np.arange(0.05,1,0.05)
    for p in p_all:
        print("p:",p)
        t = df.groupby('ship')['x'].agg({'x_%d'%int(100*p) : lambda x: x.quantile(p)}).reset_index()
        train = pd.merge(train, t, on='ship', how='left')
        t = df.groupby('ship')['y'].agg({'y_%d'%int(100*p) : lambda x: x.quantile(p)}).reset_index()
        train = pd.merge(train, t, on='ship', how='left')
        t = df.groupby('ship')['v_sin'].agg({'v_work_sin_%d'%int(100*p) : lambda x: (x[x.values>0.5]).quantile(p)}).reset_index()
        train = pd.merge(train, t, on='ship', how='left')
        t = df.groupby('ship')['v_sin'].agg({'v_notwork_sin_%d'%int(100*p) : lambda x: (x[x.values<=0.5]).quantile(p)}).reset_index()
        train = pd.merge(train, t, on='ship', how='left')
        t = df.groupby('ship')['v_cos'].agg(
            {'v_work_cos_%d' % int(100 * p): lambda x: (x[x.values > 0.5]).quantile(p)}).reset_index()
        train = pd.merge(train, t, on='ship', how='left')
        t = df.groupby('ship')['v_cos'].agg(
            {'v_notwork_cos_%d' % int(100 * p): lambda x: (x[x.values <= 0.5]).quantile(p)}).reset_index()
        train = pd.merge(train, t, on='ship', how='left')
        t = df.groupby('ship')['d'].agg({'d_%d'%int(100*p) : lambda x: x.quantile(p)}).reset_index()
        train = pd.merge(train, t, on='ship', how='left')


    t = group_feature(df, 'ship', 'x_preq', ['max', 'min', 'mean','var', 'std', 'skew', 'sum','count'])
    train = pd.merge(train, t, on='ship', how='left')

    t = df.groupby('ship')['x_preq'].agg({'x_preq_argmax': lambda x: np.argmax(x)}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'y_preq', ['max', 'min', 'mean', 'var','std', 'skew', 'sum','count'])
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['y_preq'].agg({'y_preq_argmax': lambda x: np.argmax(x)}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'v_preq', ['max', 'min', 'mean','var', 'std', 'skew', 'sum','count'])
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_preq'].agg({'v_preq_argmax': lambda x: np.argmax(x)}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'd_preq', ['max', 'min', 'mean','var', 'std', 'skew', 'sum','count'])
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['d_preq'].agg({'d_preq_argmax': lambda x: np.argmax(x)}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    train['x_max_x_min'] = train['x_max'] - train['x_min']
    train['x_max_x_min_'] = train['x_max'] + train['x_min']
    train['x_max_y_max'] = train['x_max'] - train['y_max']
    train['x_max_y_max_'] = train['x_max'] + train['y_max']
    train['x_max_y_min'] = train['x_max'] - train['y_min']
    train['x_max_y_min_'] = train['x_max'] + train['y_min']

    train['y_max_y_min'] = train['y_max'] - train['y_min']
    train['y_max_y_min_'] = train['y_max'] + train['y_min']
    train['y_max_x_max'] = train['y_max'] - train['x_max']
    train['y_max_x_max_'] = train['y_max'] + train['x_max']
    train['y_max_x_min'] = train['y_max'] - train['x_min']
    train['y_max_x_min_'] = train['y_max'] + train['x_min']

    train['x_y_length_sum'] = train['x_max_x_min']+train['y_max_y_min']
    train['diagonal_length'] = np.sqrt(train['x_max_x_min']**2+train['y_max_y_min']**2)
    train['x_y_sum'] = train['x_min'] + train['x_max'] + train['y_min']+train['y_max']

    train['slope'] = train['y_max_y_min'] / np.where(train['x_max_x_min'] == 0, 0.001, train['x_max_x_min'])
    train['area'] = train['x_max_x_min'] * train['y_max_y_min']

    train['y_max_x_min_05'] = train['y_max_x_min']/2
    train['x_max_y_min_05'] = train['x_max_y_min'] / 2
    train['x_max_y_max_05'] = train['x_max_y_max'] / 2

    train['y_max_x_min_square'] = train['y_max_x_min']**2
    train['x_max_y_min_square'] = train['x_max_y_min'] ** 2
    train['x_max_y_max_square'] = train['x_max_y_max'] ** 2

    train['y_max_x_min__x_max_y_min'] = train['y_max_x_min'] - train['x_max_y_min']
    train['y_max_x_min__x_max_y_max'] = train['y_max_x_min'] - train['x_max_y_max']
    train['x_max_y_min__x_max_y_max'] = train['x_max_y_min'] - train['x_max_y_max']

    train['y_max_x_min+x_max_y_min'] = train['y_max_x_min'] + train['x_max_y_min']
    train['y_max_x_min+x_max_y_max'] = train['y_max_x_min'] + train['x_max_y_max']
    train['x_max_y_min+x_max_y_max'] = train['x_max_y_min'] + train['x_max_y_max']

    train['y_max_x_min/x_max_y_min'] = train['y_max_x_min'] / np.where(train['x_max_y_min']==0,0.001,train['x_max_y_min'])
    train['y_max_x_min/x_max_y_max'] = train['y_max_x_min'] / np.where(train['x_max_y_max']==0,0.001,train['x_max_y_max'])
    train['x_max_y_min/x_max_y_max'] = train['x_max_y_min'] / np.where(train['x_max_y_max']==0,0.001,train['x_max_y_max'])

    train['y_max_x_min*x_max_y_min'] = train['y_max_x_min'] * train['x_max_y_min']
    train['y_max_x_min*x_max_y_max'] = train['y_max_x_min'] * train['x_max_y_max']
    train['x_max_y_min*x_max_y_max'] = train['x_max_y_min'] * train['x_max_y_max']
    # train['x_0_25'] = (train['x_min'] + train['x_mean'])/2
    # train['x_0_12_5'] = (train['x_min'] + train['x_0_25']) / 2
    # train['x_0_37_5'] = (train['x_mean'] + train['x_0_25']) / 2
    #
    # train['y_0_25'] = (train['y_min'] + train['y_mean']) / 2
    # train['y_0_12_5'] = (train['y_min'] + train['y_0_25']) / 2
    # train['y_0_37_5'] = (train['y_mean'] + train['y_0_25']) / 2
    #
    # train['x_0_75'] = (train['x_max'] + train['x_mean']) / 2
    # train['x_0_62_5'] = (train['x_mean'] + train['x_0_75']) / 2
    # train['x_0_87_5'] = (train['x_max'] + train['x_0_75']) / 2
    #
    # train['y_0_75'] = (train['y_max'] + train['y_mean']) / 2
    # train['y_0_62_5'] = (train['y_mean'] + train['y_0_75']) / 2
    # train['y_0_87_5'] = (train['y_max'] + train['y_0_75']) / 2
    #
    # train['v_0_25'] = (train['v_min'] + train['v_mean'])/2
    # train['v_0_75'] = (train['v_max'] + train['v_mean']) / 2
    #
    # train['d_0_25'] = (train['d_min'] + train['d_mean'])/2
    # train['d_0_75'] = (train['d_max'] + train['d_mean']) / 2

    mode_hour = df.groupby('ship')['hour'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_hour'] = train['ship'].map(mode_hour)

    t = group_feature(df, 'ship', 'hour', ['max', 'min'])
    train = pd.merge(train, t, on='ship', how='left')

    hour_nunique = df.groupby('ship')['hour'].nunique().to_dict()
    date_nunique = df.groupby('ship')['date'].nunique().to_dict()
    train['hour_nunique'] = train['ship'].map(hour_nunique)
    train['date_nunique'] = train['ship'].map(date_nunique)
    # print(train)
    # input()
    return train


def extract_dt(df):
    df['time'] = pd.to_datetime(df['time'], format='%m%d %H:%M:%S')
    # df['month'] = df['time'].dt.month
    # df['day'] = df['time'].dt.day
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    # df = df.drop_duplicates(['ship','month'])
    df['weekday'] = df['time'].dt.weekday

    df['d'] = df['d']/180*np.pi
    df['v_sin'] = df['v']*np.sin(df['d'])
    df['v_cos'] = df['v']*np.cos(df['d'])
    df['d_tan'] = np.sin(df['d'])/np.where(np.cos(df['d'])==0,0.001,np.cos(df['d']))

    x_freq, y_freq, v_freq, d_freq = [], [], [], []
    for _, data in df.groupby('ship'):
        x_fft = np.array(abs(fft(data['x'])))
        y_fft = np.array(abs(fft(data['y'])))
        v_fft = np.array(abs(fft(data['v'])))
        d_fft = np.array(abs(fft(data['d'])))

        x_fft[0] = 0
        y_fft[0] = 0
        v_fft[0] = 0
        d_fft[0] = 0

        x_freq.append(x_fft)
        y_freq.append(y_fft)
        v_freq.append(v_fft)
        d_freq.append(d_fft)
    df['x_preq'] = np.concatenate(np.array(x_freq), axis=0)
    df['y_preq'] = np.concatenate(np.array(y_freq), axis=0)
    df['v_preq'] = np.concatenate(np.array(v_freq), axis=0)
    df['d_preq'] = np.concatenate(np.array(d_freq), axis=0)
    return df


path = r'D:\a_zhy\MachineLearning\game\input'
train = pd.read_csv(path + r'\train.csv')
# train = df.drop_duplicates(['ship','type'])
test = pd.read_csv(path + r'\test.csv')

train = extract_dt(train)
test = extract_dt(test)
# print(train)
# 去除特定列下面的重复行
train_label = train.drop_duplicates('ship')
test_label = test.drop_duplicates('ship')

# train_label['type'].value_counts(1)

type_map = dict(zip(train_label['type'].unique(), np.arange(3)))
type_map_rev = {v:k for k,v in type_map.items()}
train_label['type'] = train_label['type'].map(type_map)
# print(train_label)
# input()
# train_label = extract_feature(train, train_label)

# test_label = extract_feature(test, test_label)
train_label = pd.read_csv("train_label.csv")
test_label = pd.read_csv("test_label.csv")

# features_impotances = ['y_max_x_min','y_max','x_min','x_max_y_min',
#                        'v_std','x','y','v_skew','x_skew','v_mean',
#                        'y_min','x_max','v','y_skew','d_mean']
# path = r'D:\a_zhy\MachineLearning\game'
# feature_impotance = pd.read_csv(path + r'\feature_importance_7.csv')
# features = [feature_impotance['name'][index] for index in range(len(feature_impotance))
#             if feature_impotance['score'][index]>100]
features = [x for x in train_label.columns if x not in ['ship','type','time','diff_time','date']]
target = 'type'
print(len(features), ','.join(features))

# pca = PCA(n_components=len(features))
X = train_label[features].copy()
# X = pd.DataFrame(pca.fit_transform(X))
# print('pca.explained_variance_ratio_:',pca.explained_variance_ratio_)
# input()
X_test = test_label[features].copy()
# X_test = pd.DataFrame(pca.fit_transform(X_test))
y = train_label[target]

print("X_shape:",X.shape)
# X: (7000, 45)
# y: (7000,)
# print("X:",X.shape)
# print("y:",y.shape)
# input()

# 自定义F1评价函数
def f1_score_vail(pred, data_vail):
    labels = data_vail.get_label()
    pred = np.argmax(pred.reshape(3, -1), axis=0)      # lgb的predict输出为各类型概率值
    score_vail = metrics.f1_score(y_true=labels, y_pred=pred, average='macro')
    return 'f1_score', score_vail, True

# model = XGBClassifier(objective='multi:softprob',num_class=3)
model = XGBClassifier(
    # max_depth=6
    # , learning_rate=0.39
    # , n_estimators=150
    max_depth=10
    , learning_rate=0.2
    , n_estimators=250
    , reg_alpha=0.004
    , n_jobs=-1
    , reg_lambda=0.002
    , importance_type='total_cover'
)
n_splits = 20
fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
models = []
pred = np.zeros((len(test_label),3))
oof = np.zeros((len(X), 3))
for index, (train_idx, val_idx) in enumerate(fold.split(X, y)):

    # train_set = lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx])
    # val_set = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])
    #
    # model = lgb.train(params, train_set, valid_sets=[train_set, val_set],
    #                   feval=f1_score_vail, verbose_eval=100,early_stopping_rounds=500)
    model.fit(X.iloc[train_idx],y.iloc[train_idx],
              eval_set=[(X.iloc[val_idx],y.iloc[val_idx])],
              eval_metric="mlogloss",
              early_stopping_rounds=20,
              verbose=False
              )
    models.append(model)
    val_pred = model.predict_proba(X.iloc[val_idx])
    oof[val_idx] = val_pred
    val_y = y.iloc[val_idx]
    val_pred = np.argmax(val_pred, axis=1)
    print(index, 'val f1', metrics.f1_score(val_y, val_pred, average='macro'),'\n')
    # 0.8695539641133697
    # 0.8866211724839532

    test_pred = model.predict_proba(X_test)
    pred += test_pred/n_splits

def predict_fun(X, train=True, alpha=None):
    alpha_all = np.linspace(0.00001,0.0001,num=10)
    # sigmoid = lambda x: 1 / (1 + np.exp(-x))
    # probabilities = pd.DataFrame(sigmoid(X))
    probabilities = pd.DataFrame(X)
    if train:
        f1_temp = 0
        for alpha in alpha_all:
            oof = []
            for i in range(len(X)):
                top_Two_idx = probabilities.iloc[i, :].argsort()
                top_idx_1 = top_Two_idx.iloc[-1]
                top_idx_2 = top_Two_idx.iloc[-2]
                top_1 = probabilities.iloc[i, top_idx_1]
                top_2 = probabilities.iloc[i, top_idx_2]
                if top_1-top_2<alpha:
                    if 0 in [top_idx_1,top_idx_2]:
                        oof.append(0)
                    else:
                        oof.append(1)
                else:
                    oof.append(top_idx_1)

            f1 = metrics.f1_score(oof, y, average='macro')
            if f1>f1_temp:
                f1_temp = f1
                alpha_best = alpha
                oof_best = oof
            print('{}_oof_f1:{}\n'.format(alpha,f1))
        return oof_best,alpha_best
    else:
        oof = []
        for i in range(len(X)):
            top_Two_idx = probabilities.iloc[i, :].argsort()
            top_idx_1 = top_Two_idx.iloc[-1]
            top_idx_2 = top_Two_idx.iloc[-2]
            top_1 = probabilities.iloc[i, top_idx_1]
            top_2 = probabilities.iloc[i, top_idx_2]
            if top_1 - top_2 < alpha:
                if 0 in [top_idx_1, top_idx_2]:
                    oof.append(0)
                else:
                    oof.append(1)
            else:
                oof.append(top_idx_1)
        return oof

oof_best,alpha_best = predict_fun(oof,train=True)

# oof = np.argmax(oof, axis=1)
print('oof f1', metrics.f1_score(oof_best, y, average='macro'))
# 0.8701544575329372
# oof f1 0.8834391705594066
# 0.0003_oof_f1:0.8874464830732484
# 0.0003_oof_f1:0.8858759101289192
# oof f1 0.8864824585747254
# 0.0003_oof_f1:0.8895655480296759
# oof f1 0.8910662835852365
# 0.0003_oof_f1:0.8924315318216453

# pred = np.argmax(pred, axis=1)
pred = predict_fun(pred,train=False,alpha=alpha_best)
sub = test_label[['ship']]
sub['pred'] = pred

print(sub['pred'].value_counts(1))
sub['pred'] = sub['pred'].map(type_map_rev)

import datetime

sub.to_csv((r"D:\a_zhy\MachineLearning\game\submit\8\submit_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"),
           header=None,index=None, encoding='utf-8')
# sub.to_csv('result4.csv', index=None, header=None, encoding='utf-8')

# ret = []
# for index, model in enumerate(models):
#     df = pd.DataFrame()
#     df['name'] = model.feature_name_
#     df['score'] = model.feature_importances_
#     df['fold'] = index
#     ret.append(df)
#
# df = pd.concat(ret)
#
# df = df.groupby('name', as_index=False)['score'].mean()
# df = df.sort_values(['score'], ascending=False)
# print(df)
# df.to_csv('feature_importance_8.csv',encoding='utf-8')
# import numpy as np
# from collections import defaultdict, Counter
# import scipy.stats
#
# def calculate_entropy(list_values):
#     counter_values = Counter(list_values).most_common()
#     probabilities = [elem[1]/len(list_values) for elem in counter_values]
#     entropy=scipy.stats.entropy(probabilities)
#     return entropy
#
# def calculate_statistics(list_values):
#     n5 = np.nanpercentile(list_values, 5)
#     n25 = np.nanpercentile(list_values, 25)
#     n75 = np.nanpercentile(list_values, 75)
#     n95 = np.nanpercentile(list_values, 95)
#     median = np.nanpercentile(list_values, 50)
#     mean = np.nanmean(list_values)
#     std = np.nanstd(list_values)
#     var = np.nanvar(list_values)
#     rms = np.nanmean(np.sqrt(list_values**2))
#     nmax_id = np.nanargmax(list_values)
#     nmin_id = np.nanargmin(list_values)
#     nmax = np.nanmax(list_values)
#     nmin = np.nanmin(list_values)
#     return [n5, n25, n75, n95, median, mean, std, var, rms, nmax_id, nmin_id, nmax, nmin]
#
# def calculate_crossings(list_values):
#     zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
#     no_zero_crossings = len(zero_crossing_indices)
#     mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
#     no_mean_crossings = len(mean_crossing_indices)
#     return [no_zero_crossings, no_mean_crossings]
#
# def get_features(list_values):
#     entropy = calculate_entropy(list_values)
#     crossings = calculate_crossings(list_values)
#     statistics = calculate_statistics(list_values)
#     return [entropy] + crossings + statistics
#
# # data = [-1,3,4]
# # print(np.nonzero(np.diff(np.array(data) > 0))[0])
