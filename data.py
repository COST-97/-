import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')
path = r'D:\a_zhy\MachineLearning\game\input'
train_path = path + r'\hy_round1_train_20200102'
test_path = path + r'\hy_round1_testA_20200102'

train_files = os.listdir(train_path)
test_files = os.listdir(test_path)
# print(len(train_files), len(test_files))

# train_files[:3]
# test_files[:3]
# df = pd.read_csv(f'{train_path}/6966.csv')
# df.head()
# df['type'].unique()
# df.shape

ret = []
for file in tqdm(train_files):
    df = pd.read_csv(f'{train_path}/{file}')
    ret.append(df)
df = pd.concat(ret)
df.columns = ['ship','x','y','v','d','time','type']

# df.to_hdf(path + r'\train.h5', 'df', mode='w')
# df.to_csv(path + r'\train.csv',encoding='utf-8')

ret = []
for file in tqdm(test_files):
    df = pd.read_csv(f'{test_path}/{file}')
    ret.append(df)
df = pd.concat(ret)
df.columns = ['ship','x','y','v','d','time']
# df.to_hdf(path + r'\test.h5', 'df', mode='w')
# df.to_csv(path + r'\test.csv',encoding='utf-8')
print(df.shape)
print(df.head())

