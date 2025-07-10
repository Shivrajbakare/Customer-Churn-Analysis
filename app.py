import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# -------------------------------
# Load Data and Models
# -------------------------------
df = pd.read_csv('Telco-Customer-Churn.csv')

# Clean TotalCharges
df = df[df['TotalCharges'] != ' '].copy()
df['TotalCharges'] = df['TotalCharges'].astype(float)

# Load encoders and model
label_encoders = joblib.load('label_encoders.pkl')
model = joblib.load('xgb_churn_model.pkl')

# Save customerID for later
customer_ids = df['customerID']

# Preprocess features
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Encode churn manually
if 'Churn' in df.columns:
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical features
for col in df.select_dtypes(include='object').columns:
    if col in label_encoders:
        df[col] = label_encoders[col].transform(df[col])

# Predict churn probabilities
X = df.drop(columns=['Churn'])
df['Churn_Probability'] = model.predict_proba(X)[:, 1]

# Add customerID back
df['customerID'] = customer_ids

# Compute CLV
GROSS_MARGIN = 0.65
df['CLV'] = (df['MonthlyCharges'].astype(float) * GROSS_MARGIN) / df['Churn_Probability']

# Risk Score & Segments
df['RiskScore'] = df['CLV'] * df['Churn_Probability']
df['RiskSegment'] = pd.qcut(df['RiskScore'], q=3, labels=['Low', 'Medium', 'High'])

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Customer Retention Dashboard", layout="wide")
st.title("📊 Customer Retention Strategy Dashboard")

# Sidebar controls
st.sidebar.header("Retention Strategy Settings")
eff_high = st.sidebar.slider("Effectiveness for High Risk", 0.0, 0.5, 0.3, 0.01)
eff_med = st.sidebar.slider("Effectiveness for Medium Risk", 0.0, 0.3, 0.15, 0.01)
eff_low = st.sidebar.slider("Effectiveness for Low Risk", 0.0, 0.1, 0.0, 0.01)
cost_high = st.sidebar.number_input("Cost for High Risk ($)", 0, 100, 20)
cost_med = st.sidebar.number_input("Cost for Medium Risk ($)", 0, 100, 10)
cost_low = st.sidebar.number_input("Cost for Low Risk ($)", 0, 100, 0)

# Map interventions
effectiveness = {'High': eff_high, 'Medium': eff_med, 'Low': eff_low}
costs = {'High': cost_high, 'Medium': cost_med, 'Low': cost_low}

df['RetainedRevenue'] = df.apply(lambda row: row['CLV'] * effectiveness[row['RiskSegment']], axis=1)
df['InterventionCost'] = df['RiskSegment'].map(costs).astype(float)


# Compute ROI
total_retained = df['RetainedRevenue'].sum()
total_cost = df['InterventionCost'].sum()
roi = total_retained / total_cost if total_cost > 0 else 0

# Layout
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(df))
col2.metric("Total Revenue Retained ($)", f"{total_retained:,.2f}")
col3.metric("Estimated ROI", f"{roi:.2f}x")

# Plot
st.subheader("🎯 CLV vs. Churn Probability")
fig = px.scatter(df, x='Churn_Probability', y='CLV', color='RiskSegment',
                 hover_data=['customerID', 'MonthlyCharges', 'RetainedRevenue'],
                 title="Customer Segmentation by Risk")
st.plotly_chart(fig, use_container_width=True)

# ROI by Segment
st.subheader("💸 Retention ROI by Segment")
summary = df.groupby('RiskSegment').agg({
    'RetainedRevenue': 'sum',
    'InterventionCost': 'sum'
}).reset_index()
summary['ROI'] = summary['RetainedRevenue'] / summary['InterventionCost']

fig2 = px.bar(summary, x='RiskSegment', y='ROI', color='RiskSegment', title="ROI by Risk Segment")
st.plotly_chart(fig2, use_container_width=True)

# Data preview
st.subheader("📋 Sample Data")
st.dataframe(df[['customerID', 'MonthlyCharges', 'Churn_Probability', 'CLV',
                 'RiskSegment', 'RetainedRevenue', 'InterventionCost']].head(20))
