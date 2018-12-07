#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 'longjh' time: 2018/11/26

# 存放已经调参好的模型
# 逻辑回归
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier

# 构建模型
clfs = {
    'lr': LogisticRegression(C=0.1, penalty='l1'),
    'linear_svc': SVC(kernel='linear', C=1, probability=True),
    'poly_svc': SVC(kernel='poly', C=1, probability=True),
    'dt': DecisionTreeClassifier(max_depth=4),
    'xgb': XGBClassifier(learning_rate=0.1, n_estimators=42, max_depth=5, min_child_weight=1, gamma=0,
                         subsample=0.8, colsample_bytree=0.8, nthread=4, scale_pos_weight=1, seed=112),
    'lgb': LGBMClassifier(learning_rate=0.1, n_estimators=42, max_depth=5, min_child_weight=1, gamma=0,
                          subsample=0.8, colsample_bytree=0.8, nthread=4, scale_pos_weight=1, seed=112)
}
