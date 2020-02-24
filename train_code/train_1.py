import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt

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

    train['x_max_x_min'] = train['x_max'] - train['x_min']
    train['y_max_y_min'] = train['y_max'] - train['y_min']
    train['y_max_x_min'] = train['y_max'] - train['x_min']
    train['x_max_y_min'] = train['x_max'] - train['y_min']
    train['slope'] = train['y_max_y_min'] / np.where(train['x_max_x_min'] == 0, 0.001, train['x_max_x_min'])
    train['area'] = train['x_max_x_min'] * train['y_max_y_min']

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

features = [x for x in train_label.columns if x not in ['ship','type','time','diff_time','date']]
target = 'type'
print(len(features), ','.join(features))

params = {
    'n_estimators': 5000,
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'early_stopping_rounds': 100,
}


fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X = train_label[features].copy()
y = train_label[target]
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

    test_pred = model.predict(test_label[features])
    pred += test_pred/5

oof = np.argmax(oof, axis=1)
print('oof f1', metrics.f1_score(oof, y, average='macro'))
# 0.8701544575329372


pred = np.argmax(pred, axis=1)
sub = test_label[['ship']]
sub['pred'] = pred

print(sub['pred'].value_counts(1))
sub['pred'] = sub['pred'].map(type_map_rev)
sub.to_csv('result.csv', index=None, header=None,encoding='utf-8')

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