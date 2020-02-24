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

def calc_ent(x):
    """
        calculate shanno ent of x
    """
    # print(x)
    # input()
    x = np.array(x)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    # print(ent)
    # input()
    return ent

def calc_condition_ent(x, y):
    """
        calculate ent H(y|x)
    """
    x = np.array(x)
    y = np.array(y)
    # print('x',x)
    # print('y',y)
    # calc ent(y|x)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent

    return ent

def calc_ent_grap(x,y):
    """
        calculate ent grap
    """
    x = np.array(x)
    y = np.array(y)
    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent

    return ent_grap


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

    t = df.groupby('ship')['x'].agg({'x_ent': lambda x: calc_ent(x)}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = df.groupby('ship')['y'].agg({'y_ent': lambda x: calc_ent(x)}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = df.groupby('ship')['v'].agg({'v_ent': lambda x: calc_ent(x)}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = df.groupby('ship')['d'].agg({'d_ent': lambda x: calc_ent(x)}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = df.groupby('ship')['x','y'].agg({'x_y_condition_ent': lambda x: calc_condition_ent(x.iloc[:,0],x.iloc[:,1])}).reset_index()
    # print(t['x_y_condition_ent']['x']==t['x_y_condition_ent']['y'])
    # print('_____________')
    # print(t['x_y_condition_ent']['y'])
    # input()
    t_ship = t['ship']
    t_ = pd.DataFrame(t['x_y_condition_ent']['x'],columns=['x_y_condition_ent'])
    t = pd.concat([t_ship,t_],axis=1)
    train = pd.merge(train, t, on='ship', how='left')

    t = df.groupby('ship')['x','y'].agg({'y_x_condition_ent': lambda x: calc_condition_ent(x.iloc[:,1],x.iloc[:,0])}).reset_index()
    # print(t['y_x_condition_ent']['x'] == t['y_x_condition_ent']['y'])
    # print(t['y_x_condition_ent']['x'])
    # input()
    t_ship = t['ship']
    t_ = pd.DataFrame(t['y_x_condition_ent']['x'], columns=['y_x_condition_ent'])
    t = pd.concat([t_ship, t_], axis=1)
    train = pd.merge(train, t, on='ship', how='left')

    t = df.groupby('ship')['x','y'].agg({'x_y_grap_ent': lambda x: calc_ent_grap(x.iloc[:,0],x.iloc[:,1])}).reset_index()
    t_ship = t['ship']
    t_ = pd.DataFrame(t['x_y_grap_ent']['x'], columns=['x_y_grap_ent'])
    t = pd.concat([t_ship, t_], axis=1)
    train = pd.merge(train, t, on='ship', how='left')

    t = df.groupby('ship')['x','y'].agg({'y_x_grap_ent': lambda x: calc_ent_grap(x.iloc[:,1],x.iloc[:,0])}).reset_index()
    t_ship = t['ship']
    t_ = pd.DataFrame(t['y_x_grap_ent']['x'], columns=['y_x_grap_ent'])
    t = pd.concat([t_ship, t_], axis=1)
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

    train['x_0_25'] = (train['x_min'] + train['x_mean'])/2
    train['x_0_12_5'] = (train['x_min'] + train['x_0_25']) / 2
    train['x_0_37_5'] = (train['x_mean'] + train['x_0_25']) / 2

    train['y_0_25'] = (train['y_min'] + train['y_mean']) / 2
    train['y_0_12_5'] = (train['y_min'] + train['y_0_25']) / 2
    train['y_0_37_5'] = (train['y_mean'] + train['y_0_25']) / 2

    train['x_0_75'] = (train['x_max'] + train['x_mean']) / 2
    train['x_0_62_5'] = (train['x_mean'] + train['x_0_75']) / 2
    train['x_0_87_5'] = (train['x_max'] + train['x_0_75']) / 2

    train['y_0_75'] = (train['y_max'] + train['y_mean']) / 2
    train['y_0_62_5'] = (train['y_mean'] + train['y_0_75']) / 2
    train['y_0_87_5'] = (train['y_max'] + train['y_0_75']) / 2

    train['v_0_25'] = (train['v_min'] + train['v_mean'])/2
    train['v_0_75'] = (train['v_max'] + train['v_mean']) / 2

    train['d_0_25'] = (train['d_min'] + train['d_mean'])/2
    train['d_0_75'] = (train['d_max'] + train['d_mean']) / 2

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

features = [x for x in train_label.columns if x not in ['ship','type','time','diff_time','date']]
# path = r'D:\a_zhy\MachineLearning\game'
# feature_impotance = pd.read_csv(path + r'\catboost_feature_importance.csv')
# features = [feature_impotance['name'][index] for index in range(len(feature_impotance))
#             if feature_impotance['score'][index]>1]
target = 'type'
print(features)
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

from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8, random_state=1234)


# from sklearn.preprocessing import StandardScaler,MinMaxScaler
# X = StandardScaler().fit_transform(X)
# X_test = StandardScaler().fit_transform(X_test)
# X = MinMaxScaler().fit_transform(X)
# X_test = MinMaxScaler().fit_transform(X_test)

# from sklearn.feature_selection import SelectKBest,chi2
# X = SelectKBest(chi2,k=20).fit_transform(X,y)

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

import catboost
# from catboost import cv,Pool
#
# params = {}
# params['loss_function'] = 'MultiClass'
# params['iterations'] = 200
# params['custom_loss'] = 'AUC'
# params['random_seed'] = 63
# params['learning_rate'] = 0.1
#
# cv_data = cv(
#     params = params,
#     pool = Pool(X, label=y),
#     fold_count=5,
#     shuffle=True,
#     partition_random_seed=0,
#     plot=True,
#     stratified=False,
#     verbose=False
# )
#
# best_value = np.min(cv_data['test--MultiClass-mean'])
# best_iter = np.argmin(np.array(cv_data['test-MultiClass-mean']))
# print('Best validation MultiClass score, not stratified: {:.4f}±{:.4f} on step {}'.format(
#     best_value,
#     cv_data['test-MultiClass-std'][best_iter],
#     best_iter)
# )

cat_model = catboost.CatBoostClassifier(
    # iterations=100,
    # learning_rate=0.03,
    # depth=6,
    # l2_leaf_reg=1,
    # random_seed=22,
    # loss_function='CrossEntropy',
    # custom_metric=['F1'],
    # eval_metric='F1'
    loss_function='MultiClass',
    custom_loss=['AUC'],
    use_best_model=True,
    # early_stopping_rounds=20
)

# params = {
#     'n_estimators': 5000,
#     'boosting_type': 'gbdt',
#     'objective': 'multiclass',
#     'num_class': 3,
#     'early_stopping_rounds': 100,
#     'learning_rate': 0.1,
#     'num_leaves': 50,
#     'max_depth': 6,
#
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#
# }
# data_train = lgb.Dataset(X, y, silent=True)
# cv_results = lgb.cv(
#     params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, feval=f1_score_vail,
#     early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)
#
# print('best n_estimators:', len(cv_results['multi_logclass-mean']))
# input()
#
# n_splits = 2
# fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
# models = []
# pred = np.zeros((len(test_label),3))
# oof = np.zeros((len(X), 3))

cat_model.fit(X_train, y_train, eval_set=(X_validation, y_validation),
              verbose=False)
print('Model is fitted: ' + str(cat_model.is_fitted()))
print('Model params:')
print(cat_model.get_params())

# cat_model.save_model(r'model/catboost_model.bin')
# cat_model.load_model(r'model/catboost_model.bin')

# oof = cat_model.predict(X)
# oof_pred = cat_model.predict_proba(X)
# print('oof_pred:')
# print(oof_pred)

oof_pred_raw = cat_model.predict(data=X,prediction_type='RawFormulaVal')
# oof_pred_raw = cat_model.predict_proba(X)
# print('oof_pred_raw:')
# print(oof_pred_raw)

def predict_fun(X, train=True, alpha=None):
    alpha_all = np.linspace(0.0003,0.0005,num=10)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    probabilities = pd.DataFrame(sigmoid(X))
    # probabilities = pd.DataFrame(X)
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

# print(pred)
# input()
# for index, (train_idx, val_idx) in enumerate(fold.split(X, y)):
#     cat_model.fit(X.iloc[train_idx], y.iloc[train_idx])
#     val_pred = cat_model.predict(X.iloc[val_idx])
#     oof[val_idx] = val_pred
#     val_y = y.iloc[val_idx]
#     val_pred = np.argmax(val_pred, axis=1)
#     print(index, 'val_f1',metrics.f1_score(val_y, val_pred, average='macro'),'\n')
#
#     test_pred = cat_model.predict(X_test)
#     pred += test_pred/n_splits

# oof = np.argmax(oof, axis=1)
# print('oof f1', metrics.f1_score(oof, y, average='macro'))
# oof f1 0.9228113488425856
# alpha_best = 0.0003
oof_best,alpha_best = predict_fun(oof_pred_raw,train=True)
# oof_best = predict_fun(oof_pred_raw,train=False,alpha=alpha_best)
# print('best_alpha:{} oof_f1:{}'.format(alpha_best,metrics.f1_score(oof_best, y, average='macro')))
# best_alpha:0.001 oof_f1:0.9053746144046276
# best_alpha:0.0005 oof_f1:0.9056541576340976
# best_alpha:0.0003 oof_f1:0.9060542462610108
# best_alpha:0.00039999999999999996 oof_f1:0.9075640518904398
# best_alpha:0.0003444444444444444 oof_f1:0.9075640518904398
print('best_alpha:{} oof_f1:{}'.format(alpha_best,metrics.f1_score(oof_best, y, average='macro')))

# pred = cat_model.predict(X_test)
# pred = np.argmax(pred, axis=1)
# pred = predict_fun(X_test,train=False,alpha=alpha_best)
oof_pred_raw_ = cat_model.predict(data=X_test,prediction_type='RawFormulaVal')
pred = predict_fun(oof_pred_raw_,train=False,alpha=alpha_best)
sub = test_label[['ship']]
sub['pred'] = pred

print(sub['pred'].value_counts(1))
# 0.0    0.6325
# 1.0    0.2390
# 2.0    0.1285

sub['pred'] = sub['pred'].map(type_map_rev)
# sub.to_csv('result4.csv', index=None, header=None, encoding='utf-8')
import datetime

sub.to_csv((r"D:\a_zhy\MachineLearning\game\submit\5\submit_" +
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"),
           header=None, index=None, encoding='utf-8')

ret = []
for index, model in enumerate([cat_model]):
    df = pd.DataFrame()
    df['name'] = model.feature_names_
    df['score'] = model.feature_importances_
    df['fold'] = index
    ret.append(df)

df = pd.concat(ret)

df = df.groupby('name', as_index=False)['score'].mean()
df = df.sort_values(['score'], ascending=False)
print(df)
df.to_csv('catboost_feature_importance_5.csv',encoding='utf-8')
