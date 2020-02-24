import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
path = r'D:\a_zhy\MachineLearning\game\input'
type_dict = {'围网':'wei','拖网':'tuo','刺网':'ci'}
train = pd.read_csv(path + r'\train.csv',encoding='utf-8')
# t = train[train['ship']==2963]
# for i in range(2010,2020):
#     t = train[train['ship']==i]
#     # print(t)
#     print(t['type'])
#     # v_min_index = np.argmin(t['v'])
#     print(np.mean(t['v']))
#     v_work = np.mean([v for v in t['v'] if v>0.5])
#     print(v_work)
#     plt.subplot(121)
#     plt.plot(t['v'])
#     # plt.show()
#     plt.subplot(122)
#     plt.plot(t['d'])
#     plt.show()


def show_path(type_name):
    ids = train[train['type']==type_name]['ship'].unique()
    ids = [ids[np.random.randint(len(ids))] for x in range(10)]
    t = train[train['ship'].isin(ids)]

    f, ax = plt.subplots(5,2, figsize=(8,20))
    for index, cur_id in enumerate(ids):
        cur = t[t['ship']==cur_id]
        i = index//2
        j = index % 2
        ax[i,j].plot(cur['x'], cur['y'])
#         if i==0 and j==0:
        ax[i,j].set_title(cur_id)
    plt.show()

# train[train['ship']==2963]

# show_path('围网')
# train[train['ship']==4022]
show_path('拖网')
# show_path('刺网')
# train[train['ship']==1415]

