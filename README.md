# -
第一阶段比赛后的总结
参考的baseline是https://github.com/jt120/tianchi_ship_2019
天池智慧海洋比赛2019 https://tianchi.aliyun.com/competition/entrance/231768/introduction?spm=5176.12281949.1003.1.493e5cfde2Jbke

成绩在0.85左右，做了一些特征工程，主要使用了lightgbm,catboost,xgboost等。


1. 先运行data.py，生成相关数据。（注意数据集的路径）
2. eda.py是对数据特征的观察。
3. train_code里面是使用各种算法的py文件。
