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

pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')


def group_feature(df, key, target, aggs):
    agg_dict = {}
    for ag in aggs:
        agg_dict[f'{target}_{ag}'] = ag
    print(agg_dict)
    t = df.groupby(key)[target].agg(agg_dict).reset_index()
    return t


def extract_feature(df, train):
    t = group_feature(df, 'ship', 'x', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'x', ['count'])
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'y', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'v', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'd', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'x_preq', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['x_preq'].agg({'x_preq_argmax': lambda x: np.argmax(x)}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'y_preq', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['y_preq'].agg({'y_preq_argmax': lambda x: np.argmax(x)}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'v_preq', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_preq'].agg({'v_preq_argmax': lambda x: np.argmax(x)}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'd_preq', ['max', 'min', 'mean', 'std', 'skew', 'sum'])
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['d_preq'].agg({'d_preq_argmax': lambda x: np.argmax(x)}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    train['x_max_x_min'] = train['x_max'] - train['x_min']
    train['y_max_y_min'] = train['y_max'] - train['y_min']
    train['y_max_x_min'] = train['y_max'] - train['x_min']
    train['x_max_y_min'] = train['x_max'] - train['y_min']
    train['slope'] = train['y_max_y_min'] / np.where(train['x_max_x_min'] == 0, 0.001, train['x_max_x_min'])
    train['area'] = train['x_max_x_min'] * train['y_max_y_min']

    train['x_0_25'] = (train['x_min'] + train['x_mean'])/2
    train['y_0_25'] = (train['y_min'] + train['y_mean']) / 2
    train['x_0_75'] = (train['x_max'] + train['x_mean']) / 2
    train['y_0_75'] = (train['y_max'] + train['y_mean']) / 2

    mode_hour = df.groupby('ship')['hour'].agg(lambda x: x.value_counts().index[0]).to_dict()
    train['mode_hour'] = train['ship'].map(mode_hour)

    t = group_feature(df, 'ship', 'hour', ['max', 'min'])
    train = pd.merge(train, t, on='ship', how='left')

    hour_nunique = df.groupby('ship')['hour'].nunique().to_dict()
    date_nunique = df.groupby('ship')['date'].nunique().to_dict()
    train['hour_nunique'] = train['ship'].map(hour_nunique)
    train['date_nunique'] = train['ship'].map(date_nunique)

    t = df.groupby('ship')['time'].agg({'diff_time': lambda x: np.max(x) - np.min(x)}).reset_index()
    t['diff_day'] = t['diff_time'].dt.days
    t['diff_second'] = t['diff_time'].dt.seconds
    train = pd.merge(train, t, on='ship', how='left')
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
train_label = extract_feature(train, train_label)

test_label = extract_feature(test, test_label)

# features_impotances = ['y_max_x_min','y_max','x_min','x_max_y_min',
#                        'v_std','x','y','v_skew','x_skew','v_mean',
#                        'y_min','x_max','v','y_skew','d_mean']
path = r'D:\a_zhy\MachineLearning\game'
feature_impotance = pd.read_csv(path + r'\feature_importance.csv')
features = [feature_impotance['name'][index] for index in range(len(feature_impotance))
            if feature_impotance['score'][index]>100]
# features = [x for x in train_label.columns if x not in ['ship','type','time','diff_time','date']]
target = 'type'
# print(len(features), ','.join(features))

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

params = {
    'n_estimators': 5000,
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'early_stopping_rounds': 100,
    'learning_rate': 0.1,
    'num_leaves': 50,
    'max_depth': 6,

    'subsample': 0.8,
    'colsample_bytree': 0.8,

}
# data_train = lgb.Dataset(X, y, silent=True)
# cv_results = lgb.cv(
#     params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, feval=f1_score_vail,
#     early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)
#
# print('best n_estimators:', len(cv_results['multi-logclass-mean']))
# input()

n_splits = 10
fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
models = []
pred = np.zeros((len(test_label),3))
oof = np.zeros((len(X), 3))
for index, (train_idx, val_idx) in enumerate(fold.split(X, y)):

    train_set = lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx])
    val_set = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])

    model = lgb.train(params, train_set, valid_sets=[train_set, val_set],
                      feval=f1_score_vail, verbose_eval=100)
    models.append(model)
    val_pred = model.predict(X.iloc[val_idx])
    oof[val_idx] = val_pred
    val_y = y.iloc[val_idx]
    val_pred = np.argmax(val_pred, axis=1)
    print(index, 'val f1', metrics.f1_score(val_y, val_pred, average='macro'),'\n')
    # 0.8695539641133697
    # 0.8866211724839532

    test_pred = model.predict(X_test)
    pred += test_pred/n_splits

oof = np.argmax(oof, axis=1)
print('oof f1', metrics.f1_score(oof, y, average='macro'))
# 0.8701544575329372


pred = np.argmax(pred, axis=1)
sub = test_label[['ship']]
sub['pred'] = pred

print(sub['pred'].value_counts(1))
sub['pred'] = sub['pred'].map(type_map_rev)

import datetime

sub.to_csv((r"D:\a_zhy\MachineLearning\game\submit\submit_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"),
           header=None,index=None, encoding='utf-8')
# sub.to_csv('result4.csv', index=None, header=None, encoding='utf-8')

ret = []
for index, model in enumerate(models):
    df = pd.DataFrame()
    df['name'] = model.feature_name()
    df['score'] = model.feature_importance()
    df['fold'] = index
    ret.append(df)

df = pd.concat(ret)

df = df.groupby('name', as_index=False)['score'].mean()
df = df.sort_values(['score'], ascending=False)
print(df)
# df.to_csv('feature_importance.csv',encoding='utf-8')
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
