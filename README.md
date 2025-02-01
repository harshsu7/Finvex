# Finvex
 Finvex is a machine learning-powered financial analytics tool that automates the extraction, categorization, and visualization of bank transactions from e-statements (PDFs). It also evaluates loan eligibility by generating a dynamic bank score.
 Features
📄 PDF Parsing – Extracts debit/credit transactions from uploaded e-statements.
📊 Data Clustering – Uses K-means clustering to categorize transactions by payment method (UPI, ATM, ACH, etc.).
📈 Dynamic Visualization – Generates pie charts and stacked bar charts for financial insights.
🤖 Loan Eligibility Prediction – Uses Logistic Regression to assess eligibility and generate a random bank score (500-900).
📂 Excel Export – Allows users to download extracted transactions in an Excel file.
🌐 Web Interface – Built using Flask + HTML/CSS, providing a clean and interactive UI.
🛠️ Tech Stack
Backend: Python, Flask
Machine Learning: Scikit-learn, K-Means, Logistic Regression
Data Processing: Pandas, NumPy, PDFPlumber
Visualization: Matplotlib, Seaborn
Frontend: HTML, CSS, Bootstrap
Storage: SQLite (or local file storage for Excel exports)
