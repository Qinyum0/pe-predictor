#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay,roc_curve,roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold,ShuffleSplit,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
from sklearn.calibration import calibration_curve  
import streamlit as st
import pandas as pd
import joblib


# In[2]:


data = pd.read_csv('PED7.csv')


# In[3]:


data_ = data.copy()
data_


# In[4]:


X = data_.drop('is_dead', axis=1)
y = data_['is_dead']
#定义标签到数值的映射
label_to_num ={'dead':0,'survival':1}
#划分训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X, y ,test_size=0.3,random_state=42)
#将数据集划分为训练集和测试集，70%为训练集，30%为测试集，random_state设定随机种子保证结果可重复


# 随机森林模型

# In[5]:


#训练随机森林模型
best_rf_model = RandomForestClassifier(bootstrap= True, max_depth= None, max_features= 'sqrt', min_samples_leaf= 10,
                                       min_samples_split= 20, n_estimators= 50)
best_rf_model.fit(X_train,y_train)


# 创建简单Web应用

# In[6]:


# 保存模型（替换 best_rf_model 为你的模型变量名）
joblib.dump(best_rf_model, 'model.pkl') 


# In[7]:


# 验证文件是否生成
import os
if os.path.exists('model.pkl'):
    print("模型文件已成功生成！文件大小：", os.path.getsize('model.pkl'), "字节")
else:
    print("模型文件生成失败，请检查路径和权限")


# In[ ]:





# In[ ]:




