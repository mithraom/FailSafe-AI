# 🛡️ FailSafe AI — Startup Insurance & Risk Intelligence Platform
 
> An AI-powered Streamlit web application that assesses startup financial risk, calculates insurance premiums, detects fraudulent failure claims, and provides a real-time portfolio risk dashboard — all wrapped in a stunning neon cyberpunk UI.
 
---
 
##  Quick Start
 
```bash
# 1. Clone the repository
git clone https://github.com/your-username/failsafe-ai.git
cd failsafe-ai
 
# 2. Install dependencies
pip install -r requirements.txt
 
# 3. Run the app
streamlit run app.py
```
 
Open your browser at `http://localhost:8501`
 
---
 
##  About the Project
 
**FailSafe AI** is a startup insurance intelligence platform that combines rule-based financial scoring with a **Random Forest ML model** to evaluate startup risk, generate insurance premiums, detect fraudulent payout claims, and provide actionable portfolio-level insights.
 
The app uses **SQLite** as its embedded database and features a fully custom **neon cyberpunk UI** built with Streamlit and injected CSS — no external UI framework required.
 
---
 
##  Features
 
### 📊 Main Dashboard
- Portfolio-level KPI metrics: Total Premium Collected, Expected Failures, Expected Payout Liability, System Buffer, Failure Rate
- Interactive **Portfolio Failure Probability** meter
- Filter by Industry and Risk Level via sidebar
- Real-time financial engine calculations
### 🤖 ML Risk Predictor
- **Random Forest Classifier** trained on startup financial features
- Predicts risk level: Low / Moderate / High
- Inputs: Monthly Revenue, Expenses, Growth Rate, Runway, Team Size, Market Risk, Funding, Industry
- Displays ML confidence probabilities per class
### 🔎 Fraud / Claim Verifier
- Verify startup failure payout claims against financial data
- Detects **red flags** (e.g. positive growth with failure claim, revenue > expenses, excess runway)
- Highlights **genuine indicators** (high burn rate, short runway, negative growth)
- Generates a **Suspicion Score (0–100%)** with a visual meter
- Issues a **Payout Decision**: Approve / Conditional / Reject
- Keyword analysis of claim description for distress vs. growth language
- Full verification summary table
### 🗃️ Startup Database (SQLite)
- Persistent storage via embedded SQLite
- Add, view, and manage startup records
- Auto-seeded with sample startups on first run
- Live DB connection status in sidebar
---
 
##  Tech Stack
 
| Technology | Purpose |
|---|---|
| **Python 3.9+** | Core language |
| **Streamlit** | Web app framework & UI |
| **SQLite3** | Embedded database |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical operations |
| **scikit-learn** | Random Forest ML model, Label Encoding |
| **Custom CSS** | Neon cyberpunk theme (Orbitron, Space Mono, Rajdhani fonts) |
 
---
 
##  How Risk Is Calculated
 
```
Risk Score = Burn Rate + Revenue Decline Factor + Runway Risk
           + Team Size Risk + Market Risk Penalty
```
 
| Risk Score | Level | Premium Rate | Payout Rate |
|---|---|---|---|
| > 100,000 | 🔴 High Risk | 12% of Funding | 40% of Funding |
| 50,000–100,000 | 🟡 Moderate Risk | 10% of Funding | 50% of Funding |
| < 50,000 | 🟢 Low Risk | 8% of Funding | 60% of Funding |
 
---
 
##  Fraud Detection Logic
 
The claim verifier flags suspicious claims based on:
 
| Red Flag | Score Added |
|---|---|
| Low/Moderate risk startup claiming payout | +30 |
| Claimed payout > 1.5× expected payout | +25 |
| Positive growth rate with failure claim | +20 |
| Revenue exceeds expenses (profitable) | +20 |
| Runway > 12 months | +15 |
| Growth-positive language in description | +15 |
| Large team with zero burn rate | +10 |
 
| Suspicion Score | Verdict |
|---|---|
| ≥ 60% | ❌ SUSPICIOUS CLAIM — Reject & Audit |
| 30–59% | ⚠️ NEEDS FURTHER REVIEW — Conditional Approval |
| < 30% | ✅ CLAIM APPEARS GENUINE — Approve |
 
---
 
##  Project Structure
 
```
failsafe-ai/
│
├── META-INF/            # Project metadata
├── app.py               # Main Streamlit application (all pages)
├── requirements.txt     # Python dependencies
├── runtime.txt          # Python version specification (for deployment)
├── startups.csv         # Startup dataset (seed data)
├── startups.ctl         # Database control file
├── .gitignore           # Secrets and environment exclusions
└── README.md            # Project documentation
```
 
---
 
##  Requirements
 
```
streamlit>=1.28
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
```
 
Python 3.9+ recommended.
 
Install all at once:
```bash
pip install streamlit pandas numpy scikit-learn
```
 
---
 
##  UI Design
 
The app features a fully custom **neon cyberpunk** aesthetic:
- Dark background (`#050508`) with neon green, magenta, cyan, and purple accents
- **Orbitron** font for headers, **Space Mono** for metrics, **Rajdhani** for body text
- Animated neon pulse effects on metric cards
- Glowing progress bars, suspicion meters, and verdict banners
- Custom scrollbar, tab styles, and button hover effects
---
 
##  Future Enhancements
 
- 📈 Historical risk trend charts per startup
- 🔐 User authentication and role-based access (Insurer / Startup)
- 📧 Automated email alerts for high-risk portfolio events
- 📄 PDF report generation for claim verification results
- 🌐 REST API backend (FastAPI) for enterprise integration
- 🧠 Advanced NLP for claim description analysis
---
 
