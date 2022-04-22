from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost
import pickle 
import time
import onnxmltools
import os
import random
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import onnxruntime as rt
import onnx
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression 
from sklearn import svm

def model_train_iris():
    iris = load_iris()
    y=iris['target']
    X=iris['data']
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.4, random_state = 42)
    model = LogisticRegression(max_iter=100)
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    metrics.accuracy_score(prediction,test_y)
    with open('/data/code_yang/cx/iris/test.pkl', 'wb') as f:
        pickle.dump(clf, f)


def model_train(X, y, ttsize, rm, md_choice, ev):
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = ttsize, random_state = rm)
    if md_choice=='svm':
        model = svm.SVC(probability=True)
    if md_choice=='lr':
        model = LogisticRegression(max_iter=200)
    model.fit(train_X, train_y) 
    prediction = model.predict(test_X)
    if ev=='acc':
        res=metrics.accuracy_score(test_y, prediction)
    if ev=='auc':
        res=metrics.roc_auc_score(test_y, prediction)
    return res