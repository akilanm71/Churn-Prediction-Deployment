from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from bertopic import BERTopic
from transformers import pipeline
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_info = topic_model.get_topic_info()


app = FastAPI(title="Churn + Topic + Generator API")
## Model Setup loading Data
@app.get("/")
def home():
    return {"message": "API running"}
## Churn Model
model = joblib.load(r"model/churn_lightgbm.pkl")
feature_columns = joblib.load("features.pkl")

## Topic Model
topic_model = BERTopic.load("./topic_model_final_gen",embedding_model=embedding_model)

## Text Generator Model
generatort2t = pipeline(
    "text2text-generation",
    model="google/flan-t5-base")

class CustomerInput(BaseModel):
    
    Gender: int
    Age: float
    Married: int
    Number_of_Dependents: float
    Referred_a_Friend: int
    Number_of_Referrals: float
    Tenure_in_Months: float
    Phone_Service: int
    Avg_Monthly_Long_Distance_Charges: float
    Multiple_Lines: int
    Internet_Service: int
    Avg_Monthly_GB_Download: float
    Online_Security: int
    Online_Backup: int
    Device_Protection_Plan: int
    Premium_Tech_Support: int
    Streaming_TV: int
    Streaming_Movies: int
    Streaming_Music: int
    Unlimited_Data: int
    Contract: int
    Paperless_Billing: int
    Monthly_Charge: float
    Satisfaction_Score: float

    Offer: str
    Internet_Type: str
    City_Grouped: str
    Payment_Method: str 

    review_text: str

## Churn Pred Preprocessing
def preprocess_input(input_dict, feature_columns):
    
    df = pd.DataFrame([input_dict])
    
    cat_cols = ["Offer", "Internet_Type", "City_Grouped", "Payment_Method"]
    
    df = pd.get_dummies(df, columns=cat_cols)
    
    df = df.reindex(columns=feature_columns, fill_value=0)
    
    bool_cols = df.select_dtypes(include=["bool", "uint8"]).columns
    df[bool_cols] = df[bool_cols].astype(int) 
    
    return df

## Label Generator Function
def generate_label(prompt):
    output = generatort2t(prompt, max_new_tokens=10, do_sample=False)
    return output[0]["generated_text"]

## Prediction

@app.post("/predict")
def predict(customer: CustomerInput):
    
    input_dict = customer.dict()  
    review_text = input_dict.pop("review_text")
    processed_df = preprocess_input(input_dict,feature_columns)
    
    prediction = model.predict(processed_df)[0]
    probability = model.predict_proba(processed_df)[0][1]
    
    topic, topic_prob = topic_model.transform([review_text])

    label = generate_label(review_text)
    
    topic_id = int(topic[0])
    topic_name = topic_info.loc[topic_info["Topic"] == topic_id,"Name"].values[0]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability),
        "topic": int(topic[0]),
        "topic_probability": float(topic_prob[0]) if topic_prob is not None else None,
        "generated_label": label,
        "topic_id": topic_id,
        "topic_name": topic_name
    }