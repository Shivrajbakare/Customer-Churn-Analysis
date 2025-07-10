# Customer Churn Prediction & Retention Strategy

This project builds a machine learning pipeline to predict customer churn, estimate lifetime value (CLV), and simulate retention strategies to minimize revenue loss.

## Project Highlights
- Churn prediction with XGBoost (ROC AUC ~80%)
-  Explainability via SHAP values
-  CLV modeling to prioritize interventions
-  A/B test simulator to estimate ROI
-  Segment-aware modeling via KMeans + Random Forest

## Folder Structure
- `Teleco_Customer_Churn`: Raw Data  files
- `Churn_prediction`: Modular Python scripts
- `clv_modelling`: Visuals and final PDF report
- `Segment_Modelling_AB_testing`: A/B testing and Segment Based Modelling
- `app`:  Streamlit dashboard

## How to Run
1. Clone repo
2. Install dependencies: `pip install -r requirements.txt`
3. To launch the Dashboard run the below command in cd:
   streamlit run app.py

## 📎 Dashboard Visuals
![Alt text](Dashboard_image.png)


---


