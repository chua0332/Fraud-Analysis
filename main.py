import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import pickle


#create the app object first
app = FastAPI()

#Load both the model and the preprocessor pkl files
model_in = open('XGB_classifier.pkl','rb')
XGBmodel = pickle.load(model_in)
preprocessor_in = open('preprocessor.pkl','rb')
preprocessor = pickle.load(preprocessor_in)



#Index route, opens automatically on https://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'hello dear!'}

@app.post('/predict')
def predict_fraud(category: object,
                  amt: float,
                  gender: object,
                  job: object,
                  merchant: object,
                  age: int):
    input_df = pd.DataFrame(dict(category=[category],amt=[amt],gender=[gender],job=[job],merchant=[merchant],age=[age]))
    
    X_processed = preprocessor.transform(input_df)
    
    XGB_pred = XGBmodel.predict(X_processed)
    
    return {'prediction': int(XGB_pred)}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)



    

