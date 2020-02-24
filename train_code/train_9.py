import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from functools import reduce
from sklearn import tree, svm, naive_bayes, neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

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
    t = group_feature(df, 'ship', 'x', ['max', 'min', 'mean', 'var', 'std', 'skew', 'sum', 'count'])
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'y', ['max', 'min', 'mean', 'var', 'std', 'skew', 'sum', 'count'])
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'v', ['max', 'min', 'mean', 'var', 'std', 'skew', 'sum', 'count'])
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'v_sin', ['max', 'min', 'mean', 'var', 'std', 'skew', 'sum', 'count'])
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'v_cos', ['max', 'min', 'mean', 'var', 'std', 'skew', 'sum', 'count'])
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'd_tan', ['max', 'min', 'mean', 'var', 'std', 'skew', 'sum', 'count'])
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'd', ['max', 'min', 'mean', 'var', 'std', 'skew', 'sum', 'count'])
    train = pd.merge(train, t, on='ship', how='left')

    t = df.groupby('ship')['v_sin'].agg({'v_work_num': lambda x: x[x.values > 0.5].count()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_notwork_num': lambda x: (x[x.values <= 0.5]).count()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_work_mean': lambda x: (x[x.values > 0.5]).mean()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_work_var': lambda x: (x[x.values > 0.5]).var()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_work_std': lambda x: (x[x.values > 0.5]).std()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_work_skew': lambda x: (x[x.values > 0.5]).skew()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_work_sum': lambda x: (x[x.values > 0.5]).sum()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = df.groupby('ship')['v_cos'].agg({'v_work_num_cos': lambda x: x[x.values > 0.5].count()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_notwork_num_cos': lambda x: (x[x.values <= 0.5]).count()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_work_mean_cos': lambda x: (x[x.values > 0.5]).mean()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_work_var_cos': lambda x: (x[x.values > 0.5]).var()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_work_std_cos': lambda x: (x[x.values > 0.5]).std()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_work_skew_cos': lambda x: (x[x.values > 0.5]).skew()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_work_sum_cos': lambda x: (x[x.values > 0.5]).sum()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = df.groupby('ship')['v_sin'].agg({'v_notwork_mean': lambda x: (x[x.values <= 0.5]).mean()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_notwork_var': lambda x: (x[x.values <= 0.5]).var()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_notwork_std': lambda x: (x[x.values <= 0.5]).std()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_notwork_skew': lambda x: (x[x.values <= 0.5]).skew()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_sin'].agg({'v_notwork_sum': lambda x: (x[x.values <= 0.5]).sum()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = df.groupby('ship')['v_cos'].agg({'v_notwork_mean_cos': lambda x: (x[x.values <= 0.5]).mean()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_notwork_var_cos': lambda x: (x[x.values <= 0.5]).var()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_notwork_std_cos': lambda x: (x[x.values <= 0.5]).std()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_notwork_skew_cos': lambda x: (x[x.values <= 0.5]).skew()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_cos'].agg({'v_notwork_sum_cos': lambda x: (x[x.values <= 0.5]).sum()}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = df.groupby('ship')['time'].agg({'diff_time': lambda x: np.max(x) - np.min(x)}).reset_index()
    t['diff_day'] = t['diff_time'].dt.days
    t['diff_second'] = t['diff_time'].dt.seconds
    train = pd.merge(train, t, on='ship', how='left')

    train['diff_second_025'] = train['diff_second'] * 0.25
    train['diff_second_05'] = train['diff_second'] * 0.5
    train['diff_second_075'] = train['diff_second'] * 0.75

    # p_all = np.arange(0.05, 1, 0.05)
    p_all = np.linspace(0.05,0.95,num=19,endpoint=True)
    for p in p_all:
        print("p:", p)
        t = df.groupby('ship')['x'].agg({'x_%d' % int(100 * p): lambda x: x.quantile(p)}).reset_index()
        train = pd.merge(train, t, on='ship', how='left')
        t = df.groupby('ship')['y'].agg({'y_%d' % int(100 * p): lambda x: x.quantile(p)}).reset_index()
        train = pd.merge(train, t, on='ship', how='left')
        t = df.groupby('ship')['v_sin'].agg(
            {'v_work_sin_%d' % int(100 * p): lambda x: (x[x.values > 0.5]).quantile(p)}).reset_index()
        train = pd.merge(train, t, on='ship', how='left')
        t = df.groupby('ship')['v_sin'].agg(
            {'v_notwork_sin_%d' % int(100 * p): lambda x: (x[x.values <= 0.5]).quantile(p)}).reset_index()
        train = pd.merge(train, t, on='ship', how='left')
        t = df.groupby('ship')['v_cos'].agg(
            {'v_work_cos_%d' % int(100 * p): lambda x: (x[x.values > 0.5]).quantile(p)}).reset_index()
        train = pd.merge(train, t, on='ship', how='left')
        t = df.groupby('ship')['v_cos'].agg(
            {'v_notwork_cos_%d' % int(100 * p): lambda x: (x[x.values <= 0.5]).quantile(p)}).reset_index()
        train = pd.merge(train, t, on='ship', how='left')
        t = df.groupby('ship')['d'].agg({'d_%d' % int(100 * p): lambda x: x.quantile(p)}).reset_index()
        train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'x_preq', ['max', 'min', 'mean', 'var', 'std', 'skew', 'sum', 'count'])
    train = pd.merge(train, t, on='ship', how='left')

    t = df.groupby('ship')['x_preq'].agg({'x_preq_argmax': lambda x: np.argmax(x)}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'y_preq', ['max', 'min', 'mean', 'var', 'std', 'skew', 'sum', 'count'])
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['y_preq'].agg({'y_preq_argmax': lambda x: np.argmax(x)}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'v_preq', ['max', 'min', 'mean', 'var', 'std', 'skew', 'sum', 'count'])
    train = pd.merge(train, t, on='ship', how='left')
    t = df.groupby('ship')['v_preq'].agg({'v_preq_argmax': lambda x: np.argmax(x)}).reset_index()
    train = pd.merge(train, t, on='ship', how='left')

    t = group_feature(df, 'ship', 'd_preq', ['max', 'min', 'mean', 'var', 'std', 'skew', 'sum', 'count'])
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

    train['x_y_length_sum'] = train['x_max_x_min'] + train['y_max_y_min']
    train['diagonal_length'] = np.sqrt(train['x_max_x_min'] ** 2 + train['y_max_y_min'] ** 2)
    train['x_y_sum'] = train['x_min'] + train['x_max'] + train['y_min'] + train['y_max']

    train['slope'] = train['y_max_y_min'] / np.where(train['x_max_x_min'] == 0, 0.001, train['x_max_x_min'])
    train['area'] = train['x_max_x_min'] * train['y_max_y_min']

    # train['y_max_x_min_05'] = train['y_max_x_min']/2
    # train['x_max_y_min_05'] = train['x_max_y_min'] / 2
    # train['x_max_y_max_05'] = train['x_max_y_max'] / 2
    #
    # train['y_max_x_min_square'] = train['y_max_x_min']**2
    # train['x_max_y_min_square'] = train['x_max_y_min'] ** 2
    # train['x_max_y_max_square'] = train['x_max_y_max'] ** 2

    # train['y_max_x_min__x_max_y_min'] = train['y_max_x_min'] - train['x_max_y_min']
    # train['y_max_x_min__x_max_y_max'] = train['y_max_x_min'] - train['x_max_y_max']
    # train['x_max_y_min__x_max_y_max'] = train['x_max_y_min'] - train['x_max_y_max']
    #
    # train['y_max_x_min+x_max_y_min'] = train['y_max_x_min'] + train['x_max_y_min']
    # train['y_max_x_min+x_max_y_max'] = train['y_max_x_min'] + train['x_max_y_max']
    # train['x_max_y_min+x_max_y_max'] = train['x_max_y_min'] + train['x_max_y_max']
    #
    # train['y_max_x_min/x_max_y_min'] = train['y_max_x_min'] / np.where(train['x_max_y_min']==0,0.001,train['x_max_y_min'])
    # train['y_max_x_min/x_max_y_max'] = train['y_max_x_min'] / np.where(train['x_max_y_max']==0,0.001,train['x_max_y_max'])
    # train['x_max_y_min/x_max_y_max'] = train['x_max_y_min'] / np.where(train['x_max_y_max']==0,0.001,train['x_max_y_max'])

    # train['y_max_x_min*x_max_y_min'] = train['y_max_x_min'] * train['x_max_y_min']
    # train['y_max_x_min*x_max_y_max'] = train['y_max_x_min'] * train['x_max_y_max']
    # train['x_max_y_min*x_max_y_max'] = train['x_max_y_min'] * train['x_max_y_max']
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
    return train


def extract_dt(df):
    df['time'] = pd.to_datetime(df['time'], format='%m%d %H:%M:%S')
    # df['month'] = df['time'].dt.month
    # df['day'] = df['time'].dt.day
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    # df = df.drop_duplicates(['ship','month'])
    df['weekday'] = df['time'].dt.weekday

    df['d'] = df['d'] / 180 * np.pi
    df['v_sin'] = df['v'] * np.sin(df['d'])
    df['v_cos'] = df['v'] * np.cos(df['d'])
    df['d_tan'] = np.sin(df['d']) / np.where(np.cos(df['d']) == 0, 0.001, np.cos(df['d']))

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
type_map_rev = {v: k for k, v in type_map.items()}
train_label['type'] = train_label['type'].map(type_map)
# print(train_label)
# input()
# train_label = extract_feature(train, train_label)
# test_label = extract_feature(test, test_label)

# train_label.to_csv("train_label.csv",index=None, encoding='utf-8')
# test_label.to_csv("test_label.csv",index=None, encoding='utf-8')

train_label = pd.read_csv("train_label.csv")
test_label = pd.read_csv("test_label.csv")
# features_impotances = ['y_max_x_min','y_max','x_min','x_max_y_min',
#                        'v_std','x','y','v_skew','x_skew','v_mean',
#                        'y_min','x_max','v','y_skew','d_mean']
# path = r'D:\a_zhy\MachineLearning\game'
# feature_impotance = pd.read_csv(path + r'\feature_importance_7.csv')
# features = [feature_impotance['name'][index] for index in range(len(feature_impotance))
#             if feature_impotance['score'][index]>100]
features = [x for x in train_label.columns if x not in ['ship', 'type', 'time', 'diff_time', 'date']]
target = 'type'
print(len(features), ','.join(features))

# pca = PCA(n_components=len(features))
X = train_label[features].copy()
# X_isnan = np.isnan(X).any()
# X_isnan.to_csv("X_isnan.csv")
# train.dropna(inplace=True)
X = X.fillna(0)
X = StandardScaler().fit_transform(X)
# X = MinMaxScaler().fit_transform(X)
# X = pd.DataFrame(pca.fit_transform(X))
# print('pca.explained_variance_ratio_:',pca.explained_variance_ratio_)
# input()
X_test = test_label[features].copy()
X_test = X_test.fillna(0)
X_test = StandardScaler().fit_transform(X_test)
# X_test = MinMaxScaler().fit_transform(X_test)
# X_test = pd.DataFrame(pca.fit_transform(X_test))
y = train_label[target]
print("X_shape:", X.shape)


# X: (7000, 45)
# y: (7000,)
# print("X:",X.shape)
# print("y:",y.shape)
# input()

# 自定义F1评价函数
def f1_score_vail(pred, data_vail):
    labels = data_vail.get_label()
    pred = np.argmax(pred.reshape(3, -1), axis=0)  # lgb的predict输出为各类型概率值
    score_vail = metrics.f1_score(y_true=labels, y_pred=pred, average='macro')
    return 'f1_score', score_vail, True


clfs = {
    # 'svm': svm.SVC(kernel='rbf', probability=True),
    #     'decision_tree': tree.DecisionTreeClassifier(),
    #     'naive_gaussian': naive_bayes.GaussianNB(),
    #     'logistic_reg': LogisticRegression(),
    #     'K_neighbor': neighbors.KNeighborsClassifier(),
    #     'bagging_knn': BaggingClassifier(neighbors.KNeighborsClassifier()),
    #     'bagging_tree': BaggingClassifier(tree.DecisionTreeClassifier()),
        # 'random_forest' : RandomForestClassifier(),
        # 'adaboost': AdaBoostClassifier(),
        # 'RandomForest':RandomForestClassifier(),
        # 'gradient_boost' : GradientBoostingClassifier(),
    'lightgbm': LGBMClassifier(boosting_type='gbdt', objective='multiclass', metric='multi_error'),
        'xgboost': XGBClassifier(objective='multi:softprob'),
        'catboost': CatBoostClassifier(loss_function='MultiClass', custom_loss=['F1'])
        }


def get_oof(clf, n_folds, X_train, y_train, X_test, model_name):
    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]
    classnum = len(np.unique(y_train))
    # kf = KFold(n_splits=n_folds, random_state=1)
    kf = StratifiedKFold(n_splits=n_folds, random_state=22)
    oof_train = np.zeros((ntrain, classnum))
    oof_test = np.zeros((ntest, classnum))

    for i, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
        kf_X_train = X_train[train_index]  # 数据
        kf_y_train = y_train[train_index]  # 标签

        kf_X_val = X_train[val_index]  # k-fold的验证集

        # if model_name in ['svm', 'RF', 'KNN']:
        #     clf.fit(kf_X_train, kf_y_train)
        if model_name in ['catboost', 'xgboost', 'lightgbm']:
            clf.fit(kf_X_train, kf_y_train, verbose=False)
        else:
            clf.fit(kf_X_train, kf_y_train)
        oof_train[val_index] = clf.predict_proba(kf_X_val)

        oof_test += clf.predict_proba(X_test)
    oof_test = oof_test / float(n_folds)
    return oof_train, oof_test


# 使用stacking方法
# 第一级，重构特征当做第二级的训练集
newfeature_list = []
newtestdata_list = []

for i, clf in enumerate(clfs.items()):
    print(i + 1, ":" + clf[0] + " start training...")
    oof_train_, oof_test_ = get_oof(clf=clf[1], n_folds=5, X_train=X, y_train=y,
                                    X_test=X_test, model_name=clf[0])
    newfeature_list.append(oof_train_)
    newtestdata_list.append(oof_test_)

print('Feature combination...')
newfeature = reduce(lambda x, y: np.concatenate((x, y), axis=1), newfeature_list)
newtestdata = reduce(lambda x, y: np.concatenate((x, y), axis=1), newtestdata_list)

print("predict...")
# alpha = 0.8
clfs_num = len(clfs)
def pred_fun(pred,clfs_num,alpha=False,train=True):
    if train:
        m,n = np.shape(pred)
        alpha_list = np.arange(0.5,0.8,0.01)
        for alpha in alpha_list:
            result = []
            best_f1 = 0
            for i in range(m):
                if np.max(pred[i,:n//clfs_num]>alpha):
                    result.append(np.argmax(pred[i,:n//clfs_num]))
                else:
                    tempt = pred[i,:n//clfs_num]+ pred[i,n//clfs_num:2*n//clfs_num]+ pred[i,2*n//clfs_num:]
                    result.append(np.argmax(tempt))
            f1 = metrics.f1_score(result, y, average='macro')
            if best_f1<f1:
                best_f1 = f1
                best_result = result
                best_alpha = alpha
            print('oof_f1:', best_f1)
        return best_result,best_alpha
    else:
        m,n = np.shape(pred)
        result = []
        for i in range(m):
            if np.max(pred[i,:n//clfs_num]>alpha):
                result.append(np.argmax(pred[i,:n//clfs_num]))
            else:
                tempt = pred[i,:n//clfs_num]+ pred[i,n//clfs_num:2*n//clfs_num]+ pred[i,2*n//clfs_num:]
                result.append(np.argmax(tempt))
        return result

oof,best_alpha = pred_fun(newfeature,clfs_num,train=True)
print("best_alpha:",best_alpha)
pred = pred_fun(newtestdata,clfs_num,alpha=best_alpha,train=False)
print('oof_best_f1', metrics.f1_score(oof, y, average='macro'))

# print('Hyperparameter optimization...')
# def acc_model(params,X_train,Y_train):
#     clf = CatBoostClassifier(**params)
#     return cross_val_score(clf, X_train, Y_train).mean()
#
# param_space = {
#     'iterations': hp.choice('iterations', [490,500,510]),
#     'depth': hp.choice('depth', [4,5,6]),
#     'learning_rate': hp.choice('learning_rate', [0.01,0.03,0.05]),
#     'l2_leaf_reg': hp.choice('l2_leaf_reg', [2,3,4]),
#     'loss_function':hp.choice('loss_function',['MultiClass']),
#     'custom_loss':hp.choice('custom_loss',['F1'])
#     # 'n_estimators': hp.choice('n_estimators', range(100,500)),
#     # 'n_estimators': hp.choice('n_estimators', range(100, 500)),
#     # 'criterion': hp.choice('criterion', ["gini", "entropy"])
#     }
#
# best = 0
# def f(params):
#     global best,param_best
#     acc = acc_model(params,X,y)
#     if acc > best:
#         best = acc
#         param_best=params
#     # print('\nnew best:', best, params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
# fmin(f, param_space, algo=tpe.suggest, max_evals=100, trials=Trials(),verbose=0)
# print('best:')
# print(param_best)
# {'criterion': 'gini', 'max_depth': 75, 'n_estimators': 207}

# param_best = {'criterion': 'gini', 'max_depth': 10, 'n_estimators': 50}
# clf_second = RandomForestClassifier(**param_best)
# clf_second.fit(newfeature, y)
# oof = clf_second.predict_proba(newfeature)
# pred = clf_second.predict_proba(newtestdata)
# clf = {'catboost':CatBoostClassifier(loss_function='MultiClass',custom_loss=['F1'])}
# clf = {'catboost':CatBoostClassifier(**param_best)}
# clf = { 'lightgbm':LGBMClassifier(boosting_type='gbdt',objective='multiclass',metric='multi_error')}
# print(clf.values())
# oof, pred = get_oof(clf=clf['catboost'], n_folds=5, X_train=X, y_train=y,
#                                 X_test=X_test, model_name='catboost')
# pred = np.argmax(pred, axis=1)
# print('f1', metrics.f1_score(pred, label_test, average='macro'))

def predict_fun(X, train=True, alpha=None):
    alpha_all = np.linspace(0.0001, 0.0004, num=11,endpoint=True)
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
                if top_1 - top_2 < alpha:
                    if 0 in [top_idx_1, top_idx_2]:
                        oof.append(0)
                    else:
                        oof.append(1)
                else:
                    oof.append(top_idx_1)

            f1 = metrics.f1_score(oof, y, average='macro')
            if f1 > f1_temp:
                f1_temp = f1
                alpha_best = alpha
                oof_best = oof
            print('{}_oof_f1:{}\n'.format(alpha, f1))
        return oof_best, alpha_best
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


# oof_best, alpha_best = predict_fun(oof, train=True)
# print('oof f1', metrics.f1_score(oof_best, y, average='macro'))
# oof f1 0.8731420735410836
# oof f1 0.8541103035857004
# pred = predict_fun(pred, train=False, alpha=alpha_best)
sub = test_label[['ship']]
sub['pred'] = pred

print(sub['pred'].value_counts(1))
sub['pred'] = sub['pred'].map(type_map_rev)

import datetime

sub.to_csv(
    (r"D:\a_zhy\MachineLearning\game\submit\9\submit_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"),
    header=None, index=None, encoding='utf-8')

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
# df.to_csv('feature_importance_7_3.csv',encoding='utf-8')
