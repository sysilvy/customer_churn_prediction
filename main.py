import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from openai import OpenAI
import utils as ut

# initialize OpenAI client using Groq API key (env first, then fallback to local key file)
api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    key_file = "Groq churn API key.txt"
    if os.path.exists(key_file):
        with open(key_file, "r", encoding="utf-8") as f:
            api_key = f.read().strip()

if not api_key:
    raise RuntimeError("GROQ_API_KEY (or OPENAI_API_KEY) not set and Groq churn API key.txt not found.")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

# function to predict churn using the selected model
def make_predictions(input_df, input_dict, selected_customer):
    def safe_probability(model):
        try:
            if hasattr(model, "predict_proba"):
                return model.predict_proba(input_df)[0][1]
            if hasattr(model, "decision_function"):
                score = model.decision_function(input_df)[0]
                return 1 / (1 + np.exp(-score))
            # fallback: use predicted label (0/1) if no scores available
            pred = model.predict(input_df)[0]
            return float(np.clip(pred, 0, 1))
        except Exception:
            return 0.0

    # Core models used for the headline churn probability (matches the original trio)
    core_probabilities = {
        'XGBoost': safe_probability(xgboost_model),
        'Random Forest': safe_probability(random_forest_model),
        'K-Nearest Neighbors': safe_probability(knn_model),
    }
    avg_probability = np.mean(list(core_probabilities.values()))

    # Models shown in the bar chart (per provided screenshot)
    chart_probabilities = {
        'Support Vector Machine': safe_probability(svm_model),
        'XGBoost': core_probabilities['XGBoost'],
        'K-Nearest Neighbors': core_probabilities['K-Nearest Neighbors'],
        'Random Forest': core_probabilities['Random Forest'],
        'Gradient Boosting': safe_probability(xgboost_SMOTE_model),
    }

    col1, col2 = st.columns(2)
    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The customer has a {avg_probability:.2%} probability of churning.")
        estimated_clv = (selected_customer["Balance"] + selected_customer["EstimatedSalary"]) * (selected_customer["Tenure"] + 1) / 10
        st.write(f"Estimated Customer Lifetime Value: ${estimated_clv:,.2f}")

    with col2:
        fig_probs = ut.create_model_probability_chart(chart_probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)

    # Customer percentile section (placed below the gauge/model charts)
    percentiles = {
        "NumOfProducts": (df["NumOfProducts"] <= selected_customer["NumOfProducts"]).mean() * 100,
        "Balance": (df["Balance"] <= selected_customer["Balance"]).mean() * 100,
        "EstimatedSalary": (df["EstimatedSalary"] <= selected_customer["EstimatedSalary"]).mean() * 100,
        "Tenure": (df["Tenure"] <= selected_customer["Tenure"]).mean() * 100,
        "CreditScore": (df["CreditScore"] <= selected_customer["CreditScore"]).mean() * 100,
    }
    st.markdown("### Customer Percentiles")
    fig_percentiles = ut.create_percentile_chart(percentiles)
    st.plotly_chart(fig_percentiles, use_container_width=True)

    return avg_probability


# function to explain prediction
def explain_prediction(probability, input_dict, surname):
    prompt = f"""
    You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models. 
    
    Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)}% probability of churning, based on the information provided below, 
    
    Here is the customer's information: 
    {input_dict}

    Here are the top 10 most important features for predicting churn:
    | Feature             | Importance  |
    |---------------------|-------------|
    | NumOfProducts       | 0.288658    |
    | IsActiveMember      | 0.194039    |
    | Age                 | 0.113240    |
    | Geography_Germany   | 0.099048    |
    | Balance             | 0.053883    |
    | Gender_Female       | 0.042302    |
    | Geography_France    | 0.041067    |
    | Geography_Spain     | 0.037478    |
    | CreditScore         | 0.034965    |
    | EstimatedSalary     | 0.033707    |
    | Tenure              | 0.033399    |
    | HasCrCard           | 0.028215    |
    | Gender_Male         | 0.000000    |

    Here are summary statistics for churned customers:
    {df[df['Exited'] == 1].describe()}

    Here are summary statistics for non-churned customers:
    {df[df['Exited'] == 0].describe()}

    - If the customer has over a 40% risk of churning, first state their exact churning risk percentage, then provide a clear and professional explanation in 3–4 sentences (no more than 90-100 words) describing the main factors in their profile that suggest they may be at risk of leaving the bank. 
    
    - If the customer has less than a 40% risk of churning, first state their exact churning risk percentage, then provide a clear and professional explanation in 3–4 sentences (no more than 90-100 words) describing the main factors in their profile that suggest they are likely to remain with the bank. 
    
    - Base your explanation on the customer’s individual information, the summary statistics of churned vs. non-churned customers, and the feature importance rankings provided. 
    
    - Do not mention machine learning, predictions, or models. The explanation should read naturally, as if written by a financial analyst reviewing customer data. Do not suggest any actions or next steps — only explain the reasoning behind the prediction.
    
    """

    raw_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return raw_response.choices[0].message.content


# function to generate email
def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""
    You are a manager at HS Bank. Your role is to build customer loyalty by creating professional, personalized emails that encourage customers to stay with the bank. You noticed that a customer named {surname} may be at risk of leaving. 
    
    Here is the customer's information: {input_dict} 
    
    Here is some background explanation of their situation: {explanation} 
    
    Write a clear, professional email addressed to the customer. The tone should be warm, respectful, and persuasive — as if written by a relationship manager, not by AI. 
    
    - If the customer is at risk of churning, politely acknowledge their value as a client and offer tailored incentives to stay. 
    
    - If the customer is not at risk, thank them for their continued loyalty and reinforce their engagement by offering additional perks. 
    
    Make sure to: 
    - Include 3–5 specific incentives in bullet points, customized to their profile. 
    - Keep the email concise (around 120–150 words). 
    - Do not mention probabilities, models, or predictions. 
    - Write naturally, like a human banker reaching out to a valued customer.
    
    """

    raw_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return raw_response.choices[0].message.content


# load models
def load_model(filename):
    with open(filename, 'rb') as file:
        return joblib.load(file)

decision_tree_model = load_model("dt_model.joblib")
knn_model = load_model("knn_model.joblib")
random_forest_model = load_model("rf_model.joblib")
svm_model = load_model("svm_model.joblib")
xgboost_model = load_model("xgb_model.joblib")
xgboost_featureEngineered_model = load_model("xgboost-featureEngineered.joblib")
xgboost_SMOTE_model = load_model("xgboost-SMOTE.joblib")
naive_bayes_model = load_model("nb_model.joblib")
voting_classifier_model = load_model("voting_classifier.joblib")
xgboost_grid_search_model = load_model("xgboost-gridsearch.joblib")


# prepare input
def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': 1 if has_credit_card else 0,
        'IsActiveMember': 1 if is_active_member else 0,
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == 'France' else 0,
        'Geography_Germany': 1 if location == 'Germany' else 0,
        'Geography_Spain': 1 if location == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0,
    }
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


# streamlit UI
st.title("Customer Churn Prediction")
df = pd.read_csv("customer_churn_dataset.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]
selected_customer_option = st.selectbox("Select a customer:", customers)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    selected_customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score", 300, 850, int(selected_customer['CreditScore']))
        location = st.selectbox("Location", ['Spain', 'France', 'Germany'],
                                index=['Spain', 'France', 'Germany'].index(selected_customer['Geography']))
        gender = st.radio("Gender", ['Male', 'Female'], index=0 if selected_customer['Gender'] == 'Male' else 1)
        age = st.number_input("Age", 18, 100, int(selected_customer['Age']))
        tenure = st.number_input("Tenure (years)", 0, 50, int(selected_customer['Tenure']))

    with col2:
        balance = st.number_input("Balance", 0.0, value=float(selected_customer['Balance']))
        num_products = st.number_input("Number of Products", 1, 10, int(selected_customer['NumOfProducts']))
        has_credit_card = st.checkbox("Has Credit Card", value=bool(selected_customer['HasCrCard']))
        is_active_member = st.checkbox("Is Active Member", value=bool(selected_customer['IsActiveMember']))
        estimated_salary = st.number_input("Estimated Salary", 0.0, value=float(selected_customer['EstimatedSalary']))

    input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)

    avg_probability = make_predictions(input_df, input_dict, selected_customer)

    explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])
    st.markdown("---")
    st.subheader("Explanation of Prediction")
    st.markdown(explanation)

    email = generate_email(avg_probability, input_dict, explanation, selected_customer['Surname'])
    st.markdown("---")
    st.subheader("Email to Customer")
    st.markdown(email)
