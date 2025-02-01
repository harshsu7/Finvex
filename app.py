from flask import Flask, render_template, request, flash, send_file, redirect, url_for
from werkzeug.utils import secure_filename  # Correct import
import os
import pdfplumber
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = 'supersecretkey'

# Ensure necessary directories exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists('static'):
    os.makedirs('static')

# Train a simple loan model
def train_loan_model():
    # Sample data for training
    data = {
        'Income': [40000, 50000, 60000, 35000, 70000],
        'Expenses': [20000, 25000, 15000, 18000, 30000],
        'Savings': [10000, 20000, 30000, 8000, 40000],
        'Loan_Approved': [1, 1, 1, 0, 1]  # 1 = Approved, 0 = Not Approved
    }
    df = pd.DataFrame(data)

    # Split data into train and test sets
    X = df[['Income', 'Expenses', 'Savings']]
    y = df['Loan_Approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

loan_model = train_loan_model()  # Train the model once

# Extract transactions from PDF
def extract_transactions_from_pdf(pdf_path):
    transactions = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            lines = text.split('\n')
            for line in lines:
                parts = line.split()
                if len(parts) >= 4:
                    date, narration, debit, credit = parts[0], parts[1], parts[2], parts[3]

                    try:
                        debit = float(debit.replace(',', '')) if debit.replace('.', '', 1).isdigit() else 0.0
                        credit = float(credit.replace(',', '')) if credit.replace('.', '', 1).isdigit() else 0.0
                    except ValueError:
                        debit, credit = 0.0, 0.0  # Handle conversion error

                    transactions.append({
                        'Date': date,
                        'Narration': narration,
                        'Debit': debit,
                        'Credit': credit
                    })

    return pd.DataFrame(transactions)

# Perform clustering on transactions
def cluster_transactions(df):
    df['Payment Method'] = df['Narration'].apply(lambda x: 'UPI' if 'UPI' in x else
                                                 'ACH' if 'ACH' in x else
                                                 'ATM' if 'ATM' in x else 'Cash')
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['Debit', 'Credit']])
    return df

# Generate charts dynamically
def generate_charts(df):
    # Pie chart for payment method distribution
    plt.figure(figsize=(6, 6))
    payment_counts = df['Payment Method'].value_counts()
    plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Payment Method Distribution')
    plt.savefig('static/pie_chart.png', bbox_inches='tight')

    # Optional: Stacked bar chart for debit vs credit comparison
    plt.figure(figsize=(8, 6))
    df.groupby(['Payment Method'])[['Debit', 'Credit']].sum().plot(
        kind='bar', stacked=True, color=['red', 'green'], ax=plt.gca()
    )
    plt.title('Debit vs Credit Comparison by Payment Method')
    plt.savefig('static/stacked_bar.png', bbox_inches='tight')


# Loan eligibility function
def loan_eligibility(income, expenses, savings):
    prediction = loan_model.predict([[income, expenses, savings]])
    return 'Eligible' if prediction[0] == 1 else 'Not Eligible'

# Routes for the app
@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'statement' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['statement']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and file.filename.endswith('.pdf'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Process transactions and save Excel
        df_transactions = extract_transactions_from_pdf(filepath)
        df_clustered = cluster_transactions(df_transactions)
        generate_charts(df_clustered)

        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], 'transactions_extracted.xlsx')
        df_clustered.to_excel(excel_path, index=False)

        # Generate random bank score and eligibility
        bank_score = np.random.randint(500, 901)
        loan_status = 'Eligible' if bank_score >= 650 else 'Not Eligible'

        flash('File successfully uploaded and processed!')
        return render_template('result.html', loan_status=loan_status, bank_score=bank_score)

    else:
        flash('Allowed file type is PDF')
        return redirect(request.url)

@app.route('/download')
def download_file():
    try:
        # Construct the full path for the Excel file
        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], 'transactions_extracted.xlsx')
        
        # Check if the file exists before sending it
        if os.path.exists(excel_path):
            return send_file(excel_path, as_attachment=True)
        else:
            flash("Excel file not found. Please upload and process a PDF first.")
            return redirect(url_for('upload_form'))
    except Exception as e:
        flash(f"An error occurred: {str(e)}")
        return redirect(url_for('upload_form'))

if __name__ == '__main__':
    app.run(debug=True)
