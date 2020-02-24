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

X = train_label[features].copy()
# X = (X-X.mean())/X.std()
# X = (X-X.min())/(X.max()-X.min())
X_test = test_label[features].copy()
# X_test = (X_test-X_test.mean())/X_test.std()
# X_test = (X_test-X_test.min())/(X_test.max()-X_test.min())
y = train_label[target]
# X: (7000, 45)
# y: (7000,)


# 自定义F1评价函数
def f1_score_vail(pred, data_vail):
    labels = data_vail.get_label()
    pred = np.argmax(pred.reshape(3, -1), axis=0)      # lgb的predict输出为各类型概率值
    score_vail = metrics.f1_score(y_true=labels, y_pred=pred, average='macro')
    return 'f1_score', score_vail, True


'''模型优化———————————————————'''
from sklearn.model_selection import train_test_split
X_train_, X_val_, y_train_, y_val_ = train_test_split(X, y, random_state=0, test_size=0.2)

### 数据转换
print('数据转换')
lgb_train = lgb.Dataset(X_train_, y_train_, free_raw_data=False)
lgb_eval = lgb.Dataset(X_val_, y_val_, reference=lgb_train, free_raw_data=False)

print('设置参数')
params = {
    'metric': 'multi_logloss',
    'nthread': 4,
    'n_estimators': 5000,
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'early_stopping_rounds': 100,
}

# 交叉验证(调参)
print('交叉验证')
max_auc = float('1')
best_params = {}


# 准确率
# print("调参1：提高准确率")
# for num_leaves in range(10, 11, 20):
#     for max_depth in range(3, 4, 2):
#         params['num_leaves'] = num_leaves
#         params['max_depth'] = max_depth
#
#         cv_results = lgb.cv(
#             params,
#             lgb_train,
#             seed=1,
#             nfold=5,
#             # metrics=['multi_logloss'],
#             feval=f1_score_vail,
#             early_stopping_rounds=10,
#             verbose_eval=True
#         )
#         # print(cv_results)
#         # input()
#         mean_auc = pd.Series(cv_results['multi_logloss-mean']).min()
#         boost_rounds = pd.Series(cv_results['multi_logloss-mean']).idxmin()
#
#         if mean_auc <= max_auc:
#             max_auc = mean_auc
#             best_params['num_leaves'] = num_leaves
#             best_params['max_depth'] = max_depth
# if 'num_leaves' and 'max_depth' in best_params.keys():
#     params['num_leaves'] = best_params['num_leaves']
#     params['max_depth'] = best_params['max_depth']

# # 过拟合
# print("调参2：降低过拟合")
# for max_bin in range(5, 256, 10):
#     for min_data_in_leaf in range(1, 102, 10):
#         params['max_bin'] = max_bin
#         params['min_data_in_leaf'] = min_data_in_leaf
#
#         cv_results = lgb.cv(
#             params,
#             lgb_train,
#             seed=1,
#             nfold=5,
#             metrics=['multi_logloss'],
#             early_stopping_rounds=10,
#             verbose_eval=True
#         )
#
#         mean_auc = pd.Series(cv_results['multi_logloss-mean']).max()
#         boost_rounds = pd.Series(cv_results['multi_logloss-mean']).idxmax()
#
#         if mean_auc >= max_auc:
#             max_auc = mean_auc
#             best_params['max_bin'] = max_bin
#             best_params['min_data_in_leaf'] = min_data_in_leaf
# if 'max_bin' and 'min_data_in_leaf' in best_params.keys():
#     params['min_data_in_leaf'] = best_params['min_data_in_leaf']
#     params['max_bin'] = best_params['max_bin']
#
# print("调参3：降低过拟合")
# for feature_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
#     for bagging_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
#         for bagging_freq in range(0, 50, 5):
#             params['feature_fraction'] = feature_fraction
#             params['bagging_fraction'] = bagging_fraction
#             params['bagging_freq'] = bagging_freq
#
#             cv_results = lgb.cv(
#                 params,
#                 lgb_train,
#                 seed=1,
#                 nfold=5,
#                 metrics=['multi_logloss'],
#                 early_stopping_rounds=10,
#                 verbose_eval=True
#             )
#
#             mean_auc = pd.Series(cv_results['multi_logloss-mean']).max()
#             boost_rounds = pd.Series(cv_results['multi_logloss-mean']).idxmax()
#
#             if mean_auc >= max_auc:
#                 max_auc = mean_auc
#                 best_params['feature_fraction'] = feature_fraction
#                 best_params['bagging_fraction'] = bagging_fraction
#                 best_params['bagging_freq'] = bagging_freq
#
# if 'feature_fraction' and 'bagging_fraction' and 'bagging_freq' in best_params.keys():
#     params['feature_fraction'] = best_params['feature_fraction']
#     params['bagging_fraction'] = best_params['bagging_fraction']
#     params['bagging_freq'] = best_params['bagging_freq']
#
# print("调参4：降低过拟合")
# for lambda_l1 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
#     for lambda_l2 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.4, 0.6, 0.7, 0.9, 1.0]:
#         params['lambda_l1'] = lambda_l1
#         params['lambda_l2'] = lambda_l2
#         cv_results = lgb.cv(
#             params,
#             lgb_train,
#             seed=1,
#             nfold=5,
#             metrics=['multi_logloss'],
#             early_stopping_rounds=10,
#             verbose_eval=True
#         )
#
#         mean_auc = pd.Series(cv_results['multi_logloss-mean']).max()
#         boost_rounds = pd.Series(cv_results['multi_logloss-mean']).idxmax()
#
#         if mean_auc >= max_auc:
#             max_auc = mean_auc
#             best_params['lambda_l1'] = lambda_l1
#             best_params['lambda_l2'] = lambda_l2
# if 'lambda_l1' and 'lambda_l2' in best_params.keys():
#     params['lambda_l1'] = best_params['lambda_l1']
#     params['lambda_l2'] = best_params['lambda_l2']
#
# print("调参5：降低过拟合2")
# for min_split_gain in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#     params['min_split_gain'] = min_split_gain
#
#     cv_results = lgb.cv(
#         params,
#         lgb_train,
#         seed=1,
#         nfold=5,
#         metrics=['multi_logloss'],
#         early_stopping_rounds=10,
#         verbose_eval=True
#     )
#
#     mean_auc = pd.Series(cv_results['multi_logloss-mean']).max()
#     boost_rounds = pd.Series(cv_results['multi_logloss-mean']).idxmax()
#
#     if mean_auc >= max_auc:
#         max_auc = mean_auc
#
#         best_params['min_split_gain'] = min_split_gain
# if 'min_split_gain' in best_params.keys():
#     params['min_split_gain'] = best_params['min_split_gain']

# print(best_params)
# 参数
# params = {
#     'n_estimators': 5000,
#     'boosting_type': 'gbdt',
#     'objective': 'multiclass',
#     'num_class': 3,
#     'early_stopping_rounds': 100,
# }

'''模型训练———————————————————'''
fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
    pred += test_pred/5

# print("参数：\n",params)
oof = np.argmax(oof, axis=1)
print('oof f1', metrics.f1_score(oof, y, average='macro'))
# 0.8701544575329372


pred = np.argmax(pred, axis=1)
sub = test_label[['ship']]
sub['pred'] = pred

print(sub['pred'].value_counts(1))
sub['pred'] = sub['pred'].map(type_map_rev)
sub.to_csv('result_3.csv', index=None, header=None)

# ret = []
# for index, model in enumerate(models):
#     df = pd.DataFrame()
#     df['name'] = model.feature_name()
#     df['score'] = model.feature_importance()
#     df['fold'] = index
#     ret.append(df)
#
# df = pd.concat(ret)
#
# df = df.groupby('name', as_index=False)['score'].mean()
# df = df.sort_values(['score'], ascending=False)
# print(df)
