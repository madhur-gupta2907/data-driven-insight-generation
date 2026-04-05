# 🔍 DataLens – Data-Driven Insight Generation Platform

A complete EDA (Exploratory Data Analysis) project built with **Streamlit, Pandas, Matplotlib & Seaborn**.

---

## 📁 Project Structure
```
data_insights_app/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🚀 How to Run

### Step 1 – Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 – Run the app
```bash
streamlit run app.py
```

### Step 3 – Open browser
The app opens automatically at: **http://localhost:8501**

---

## ✨ Features

| Tab | What it does |
|-----|-------------|
| 📊 Overview | Dataset preview, shape, statistical summary |
| 🧹 Data Cleaning | Missing values, duplicates, outlier detection (IQR) |
| 📈 EDA Charts | Histogram, Boxplot, Bar, Pie, Heatmap, Scatter |
| 🔮 Insights | Auto-generated insights with trend analysis |
| 💼 Recommendations | Business recs + Priority Matrix + Executive Summary |

---

## 📦 Libraries Used

- **Streamlit** — Web application framework
- **Pandas** — Data loading, cleaning, aggregation
- **NumPy** — Numerical operations
- **Matplotlib** — Custom charts
- **Seaborn** — Heatmaps, statistical plots

---

## 📂 Data

- **Default:** Built-in sample retail sales dataset (500 rows, 10 columns)
- **Custom:** Upload any `.csv` file via sidebar

---

## 🎓 Project Highlights (for evaluation)

1. ✅ Data Cleaning with automated missing value & outlier report
2. ✅ 6+ Chart types (Distribution, Category, Correlation, Trend)
3. ✅ Auto-generated data insights
4. ✅ Business recommendations with priority matrix
5. ✅ Executive summary for presentation
6. ✅ Interactive Streamlit web interface
7. ✅ Supports custom CSV upload

---

*Built for academic Data Analysis project submission.*
