import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# 必须在所有Streamlit命令之前设置页面配置
st.set_page_config(
    page_title="PE Mortality Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载预训练模型
try:
    # 确保model.pkl文件存在于同一目录
    best_rf_model = joblib.load("model.pkl")  
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()  # 如果模型加载失败则停止应用

def predict_mortality(patient_data):
    """使用预训练模型进行预测"""
    try:
        input_df = pd.DataFrame([patient_data])
        # 确保输入字段与模型训练时完全一致
        proba = best_rf_model.predict_proba(input_df)[0]
        prediction = best_rf_model.predict(input_df)[0]
        return prediction, proba
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def main():
    st.title('PE Mortality Risk Prediction Model')
    st.markdown("""
    This tool predicts mortality risk for pulmonary embolism (PE) patients.
    """)
    
    # 侧边栏输入
    st.sidebar.header('Patient Parameters')
    age = st.sidebar.slider('Age', 18, 100, 50)
    WBC = st.sidebar.number_input('WBC (K/uL)', 0, 100, 8)
    RBC = st.sidebar.number_input('RBC (m/uL)', 0, 10, 4)
    RDW = st.sidebar.number_input('RDW (%)', 0, 100, 14)
    Cl = st.sidebar.number_input('Cl (mEq/L)', 0, 200, 100)
    GLU = st.sidebar.number_input('GLU (mg/dL)', 0, 200, 100)
    INR = st.sidebar.number_input('INR (sec)', 0, 10, 1)
    BUN = st.sidebar.number_input('BUN (mg/dL)', 0, 100, 20)
    
    if st.sidebar.button('Predict'):
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
        
        prediction, proba = predict_mortality(patient_data)
        
        if prediction is not None:
            st.subheader('Prediction Results')
            if prediction == 1:
                st.error(f'High Risk: Mortality probability {proba[1]*100:.2f}%')
            else:
                st.success(f'Low Risk: Survival probability {proba[0]*100:.2f}%')
            
            st.progress(proba[1])
            st.write(f'Survival: {proba[0]*100:.2f}% | Mortality: {proba[1]*100:.2f}%')

if __name__ == '__main__':
    main()