# FailSafe AI – Startup Risk Intelligence Platform

FailSafe AI is an AI-driven platform designed to analyze startup financial health, predict failure risk, and simulate a structured risk-sharing mechanism similar to an insurance model.

---

## Overview

The platform evaluates startups using key financial and operational indicators to generate a comprehensive risk profile. It combines rule-based scoring with machine learning to support data-driven decision-making for founders, investors, and risk analysts.

---

## Key Features

- Risk Intelligence Dashboard  
  Provides a real-time overview of startup risk metrics, including risk scores, expected failures, payouts, and system buffer.

- AI Risk Mentor  
  Identifies critical risk factors and offers actionable mitigation strategies based on financial inputs.

- Failure Verification System  
  Assesses whether a startup failure claim is genuine or potentially fraudulent using heuristic and ML-based validation.

- Premium and Payout Engine  
  Simulates a financial protection model by calculating premium contributions and estimated payouts.

- Machine Learning Integration  
  Uses a Random Forest classifier to predict risk levels and probability distributions.

---

## Technology Stack

- Frontend: Streamlit  
- Backend: Python  
- Database: SQLite  
- Machine Learning: Scikit-learn (Random Forest Classifier)  
- Data Processing: Pandas, NumPy  

---

## Methodology

The system evaluates startups based on:

- Monthly Revenue and Expenses (Burn Rate)  
- Growth Rate  
- Runway (in months)  
- Team Size  
- Market Risk  
- Funding Level  

These inputs are used to:

1. Compute a composite risk score  
2. Classify startups into risk categories (Low, Moderate, High)  
3. Estimate premium contributions and payout coverage  
4. Predict risk probabilities using a trained ML model  

---

## Live Application

Streamlit App: http : https://failsafe-ai.streamlit.app/

---


cd failsafe-ai
pip install -r requirements.txt
streamlit run app.py
