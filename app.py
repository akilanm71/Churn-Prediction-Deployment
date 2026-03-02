import gradio as gr
import joblib
import pandas as pd
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import pipeline

torch.set_num_threads(1)

# =========================
# LOAD MODELS (ONLY ONCE)
# =========================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)

topic_model = BERTopic.load(
    "./topic_model_final_gen",
    embedding_model=embedding_model
)

topic_info = topic_model.get_topic_info()

churn_model = joblib.load("model/churn_lightgbm.pkl")
feature_columns = joblib.load("features.pkl")

#  flan-t5-base may be heavy on free tier
# If memory issue, change to flan-t5-small
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1
)


# =========================
# PREPROCESS FUNCTION
# =========================

def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])

    cat_cols = ["Offer", "Internet_Type", "City_Grouped", "Payment_Method"]

    df = pd.get_dummies(df, columns=cat_cols)
    df = df.reindex(columns=feature_columns, fill_value=0)

    bool_cols = df.select_dtypes(include=["bool", "uint8"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


# =========================
# MAIN PREDICTION FUNCTION
# =========================

def predict(
    Gender, Age, Married, Number_of_Dependents,
    Referred_a_Friend, Number_of_Referrals,
    Tenure_in_Months, Phone_Service,
    Avg_Monthly_Long_Distance_Charges,
    Multiple_Lines, Internet_Service,
    Avg_Monthly_GB_Download, Online_Security,
    Online_Backup, Device_Protection_Plan,
    Premium_Tech_Support, Streaming_TV,
    Streaming_Movies, Streaming_Music,
    Unlimited_Data, Contract,
    Paperless_Billing, Monthly_Charge,
    Satisfaction_Score,
    Offer, Internet_Type,
    City_Grouped, Payment_Method,
    review_text
):

    input_dict = {
        "Gender": Gender,
        "Age": Age,
        "Married": Married,
        "Number_of_Dependents": Number_of_Dependents,
        "Referred_a_Friend": Referred_a_Friend,
        "Number_of_Referrals": Number_of_Referrals,
        "Tenure_in_Months": Tenure_in_Months,
        "Phone_Service": Phone_Service,
        "Avg_Monthly_Long_Distance_Charges": Avg_Monthly_Long_Distance_Charges,
        "Multiple_Lines": Multiple_Lines,
        "Internet_Service": Internet_Service,
        "Avg_Monthly_GB_Download": Avg_Monthly_GB_Download,
        "Online_Security": Online_Security,
        "Online_Backup": Online_Backup,
        "Device_Protection_Plan": Device_Protection_Plan,
        "Premium_Tech_Support": Premium_Tech_Support,
        "Streaming_TV": Streaming_TV,
        "Streaming_Movies": Streaming_Movies,
        "Streaming_Music": Streaming_Music,
        "Unlimited_Data": Unlimited_Data,
        "Contract": Contract,
        "Paperless_Billing": Paperless_Billing,
        "Monthly_Charge": Monthly_Charge,
        "Satisfaction_Score": Satisfaction_Score,
        "Offer": Offer,
        "Internet_Type": Internet_Type,
        "City_Grouped": City_Grouped,
        "Payment_Method": Payment_Method
    }

    # Churn
    processed_df = preprocess_input(input_dict)
    prediction = churn_model.predict(processed_df)[0]
    probability = churn_model.predict_proba(processed_df)[0][1]

    # Topic
    topic, topic_prob = topic_model.transform([review_text])
    topic_id = int(topic[0])

    topic_name = topic_info.loc[
        topic_info["Topic"] == topic_id, "Name"
    ].values[0]

    # Label generation
    label_output = generator(
        review_text,
        max_new_tokens=10,
        do_sample=False
    )
    generated_label = label_output[0]["generated_text"]

    return (
        int(prediction),
        round(float(probability), 3),
        topic_id,
        topic_name,
        generated_label
    )


# =========================
# GRADIO UI
# =========================

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Gender (0/1)"),
        gr.Number(label="Age"),
        gr.Number(label="Married (0/1)"),
        gr.Number(label="Number_of_Dependents"),
        gr.Number(label="Referred_a_Friend (0/1)"),
        gr.Number(label="Number_of_Referrals"),
        gr.Number(label="Tenure_in_Months"),
        gr.Number(label="Phone_Service (0/1)"),
        gr.Number(label="Avg_Monthly_Long_Distance_Charges"),
        gr.Number(label="Multiple_Lines (0/1)"),
        gr.Number(label="Internet_Service (0/1)"),
        gr.Number(label="Avg_Monthly_GB_Download"),
        gr.Number(label="Online_Security (0/1)"),
        gr.Number(label="Online_Backup (0/1)"),
        gr.Number(label="Device_Protection_Plan (0/1)"),
        gr.Number(label="Premium_Tech_Support (0/1)"),
        gr.Number(label="Streaming_TV (0/1)"),
        gr.Number(label="Streaming_Movies (0/1)"),
        gr.Number(label="Streaming_Music (0/1)"),
        gr.Number(label="Unlimited_Data (0/1)"),
        gr.Number(label="Contract"),
        gr.Number(label="Paperless_Billing (0/1)"),
        gr.Number(label="Monthly_Charge"),
        gr.Number(label="Satisfaction_Score"),
        gr.Textbox(label="Offer"),
        gr.Textbox(label="Internet_Type"),
        gr.Textbox(label="City_Grouped"),
        gr.Textbox(label="Payment_Method"),
        gr.Textbox(label="Customer Review", lines=3)
    ],
    outputs=[
        gr.Number(label="Churn Prediction"),
        gr.Number(label="Churn Probability"),
        gr.Number(label="Topic ID"),
        gr.Textbox(label="Topic Name"),
        gr.Textbox(label="Generated Label")
    ],
    title="Churn + Topic + Generator System",
    description="End-to-End Customer Churn & Topic Analysis"
)

interface.launch()