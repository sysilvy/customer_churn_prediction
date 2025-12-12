# Customer Churn Prediction System

This project is an end-to-end machine learning application designed to predict customer churn for a banking dataset.
The system identifies customers who are likely to leave and provides insights to support retention strategies.

## 🚀 Live Demo
https://customer-churn-prediction-eafo.onrender.com/

## 🎥 Pitch Deck Demo
[PASTE_YOUR_PITCH_DECK_VIDEO_LINK_HERE](https://youtu.be/ENmJDRB_52Q)

## 📊 Dataset
The dataset is sourced from Kaggle and contains bank customer information such as credit score, geography, tenure, balance, and churn status.

## 🧠 Machine Learning Pipeline
- Data cleaning and preprocessing
- Feature engineering
- Train–test split
- Class imbalance handling using SMOTE
- Baseline model training (Logistic Regression, KNN, Random Forest, SVM)
- XGBoost model training
- Hyperparameter tuning using GridSearchCV (recall-focused)

## 📈 Model Evaluation
Recall was prioritized as the primary evaluation metric to minimize false negatives and correctly identify potential churners.

## 🌐 Deployment
The final tuned model was saved using joblib and deployed as a Streamlit web application on Render.
The application supports real-time predictions, visualizations, and AI-generated customer retention emails.

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Streamlit
- Plotly
- Render


## 📌 Author
Samira Yasmin 

LinkedIn: https://www.linkedin.com/in/samira-yasmin-495416203/
