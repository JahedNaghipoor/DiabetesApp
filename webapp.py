import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pickle

def main():

 st.write(""" # Diabetes Detection """)
 df = pd.read_csv("diabetes.csv")
 st.subheader("Data Information:")
 st.dataframe(df.head(20))
 st.bar_chart(df)

 X = df.iloc[:,0:8].values
 y = df.iloc[:,-1].values

 X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=111)

 user_input = get_user_input()
 st.subheader('User Input:')
 st.write(user_input)

 rf = RandomForestClassifier()
 rf.fit(X_train, y_train)

 pickle.dump(rf, open('model.pkl','wb'))  

 model = pickle.load(open('model.pkl','rb'))
 y_pred = model.predict(X_test)


 st.subheader('Model Test Accuracy Score: '+ str(round(accuracy_score(y_test, y_pred),4)*100)+' %')

 prediction = rf.predict(user_input)

 st.subheader('Diabetes: ' + ('Negative', 'Positive')[int(prediction)])

def get_user_input():
    pregnencies = st.sidebar.slider("Pregnencies", 0,17,3)
    glucose = st.sidebar.slider("Glucose", 0,199,117)
    blood_pressure = st.sidebar.slider("Blood pressure", 0,122,72)
    skin_thickness = st.sidebar.slider("Skin thickness", 0,99,23)
    insulin = st.sidebar.slider("Insulin", 0.0,846.0,30.0)
    BMI = st.sidebar.slider("BMI", 0.0,67.1,32.0)
    DPF = st.sidebar.slider("DPF", 0.078,2.42,0.3725)
    age = st.sidebar.slider("Age", 21,81,29)
    
    user_data = {
        'Pregnencies': pregnencies,
        'Glucose': glucose,
        'Blood pressure': blood_pressure,
        'Skin thickness': skin_thickness,
        'Insulin':insulin,
        'BMI':BMI,
        'DPF':DPF,
        'Age':age
        }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

if __name__ == '__main__':
    main()



