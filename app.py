#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegressionCV
# from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay,roc_curve,roc_auc_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
# from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# import xgboost as xgb
# import lightgbm as lgb
# from sklearn.model_selection import KFold,ShuffleSplit,cross_val_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
# from sklearn.metrics import roc_curve,auc
# from sklearn.model_selection import StratifiedKFold
# import seaborn as sns
# from sklearn.calibration import calibration_curve  
# import streamlit as st
import pandas as pd
import joblib


# In[2]:


data = pd.read_csv('PED7.csv')


# In[3]:


data_ = data.copy()
data_


# In[7]:


X = data_.drop('is_dead', axis=1)
y = data_['is_dead']
#定义标签到数值的映射
label_to_num ={'survival':0,'dead':1}
#划分训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X, y ,test_size=0.3,random_state=42)
#将数据集划分为训练集和测试集，70%为训练集，30%为测试集，random_state设定随机种子保证结果可重复


# 随机森林模型

# In[8]:


#训练随机森林模型
best_rf_model = RandomForestClassifier(bootstrap= True, max_depth= None, max_features= 'sqrt', min_samples_leaf= 10,
                                       min_samples_split= 20, n_estimators= 50)
best_rf_model.fit(X_train,y_train)


# In[10]:


# 4. 保存模型（关键！）
joblib.dump(best_rf_model, 'model.pkl')


# 创建简单Web应用

# In[11]:


def predict_mortality(patient_data):
    """使用训练好的模型进行预测"""
    # 将输入数据转换为DataFrame
    input_df = pd.DataFrame([patient_data])
    
    # 进行预测(概率和类别)
    proba = best_rf_model.predict_proba(input_df)[0]
    prediction = best_rf_model.predict(input_df)[0]
    
    return prediction, proba


# In[12]:

import streamlit as st
def main():
    # 页面设置
    st.set_page_config(page_title="PE Mortality Predictor", layout="wide")
    
    # 应用标题和描述
    st.title('PE Mortality Risk Prediction Model')
    st.markdown("""
    This tool predicts mortality risk for pulmonary embolism (PE) patients using a Random Forest model based on clinical parameters.
    """)
    
    # 创建侧边栏用于输入参数
    st.sidebar.header('Patient Parameters')
    
    # 收集用户输入
    age = st.sidebar.slider('Age', 18, 100, 50)
    WBC = st.sidebar.number_input('WBC (K/uL)', 0, 100, 8)
    RBC = st.sidebar.number_input('RBC (m/uL)', 0, 10, 4)
    RDW = st.sidebar.number_input('RDW (%)', 0, 100, 14)
    Cl = st.sidebar.number_input('Cl (mEq/L)', 0, 200, 100)
    GLU = st.sidebar.number_input('GLU (mg/dL)', 0, 200, 100)
    INR = st.sidebar.number_input('INR (sec)', 0, 10, 1)
    BUN = st.sidebar.number_input('BUN (mg/dL)', 0, 100, 20)
    
    # 当用户点击预测按钮时
    if st.sidebar.button('Predict'):
        # 准备输入数据
        patient_data = {
            'age': age,
            'WBC': WBC,
            'RBC': RBC,
            'RDW': RDW,
            'Cl': Cl,
            'GLU': GLU,
            'INR': INR,
            'BUN': BUN
        }
        
        # 进行预测
        prediction, proba = predict_mortality(patient_data)
        
        # 显示结果
        st.subheader('Prediction Results')
        if prediction == 1:
            st.error(f'High Risk: Mortality probability {proba[1]*100:.2f}%')
        else:
            st.success(f'Low Risk: Survival probability {proba[0]*100:.2f}%')
        
        # 显示概率条
        st.write('Risk Probability:')
        st.progress(proba[1])
        
        # 显示详细概率
        st.write(f'Survival Probability: {proba[0]*100:.2f}%')
        st.write(f'Mortality Probability: {proba[1]*100:.2f}%')

if __name__ == '__main__':
    main()


# In[13]:


get_ipython().system('jupyter nbconvert --to script "肺栓塞模型网站1.1.ipynb"')


# In[ ]:





# In[ ]:




