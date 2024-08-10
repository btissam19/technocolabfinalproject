from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved pipelines
pipeline_rf = joblib.load('random_forest_classifier_pipeline.joblib')
pipeline_lasso = joblib.load('lasso_regression_pipeline.joblib')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.form.get('model_type')
    if model_type == 'classification':
        return redirect(url_for('classification'))
    elif model_type == 'regression':
        return redirect(url_for('regression'))
    else:
        return redirect(url_for('home'))

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    if request.method == 'POST':
        try:
            features = {
                'MonthsDelinquent': float(request.form.get('MonthsDelinquent')),
                'CreditScore': float(request.form.get('CreditScore')),
                'MaturityMonth': float(request.form.get('MaturityMonth')),
                'FirstPaymentMonth': float(request.form.get('FirstPaymentMonth')),
                'OrigLoanTerm': float(request.form.get('OrigLoanTerm')),
                'OrigInterestRate': float(request.form.get('OrigInterestRate')),
                'OrigUPB': float(request.form.get('OrigUPB')),
                'Units': float(request.form.get('Units')),
                'PropertyType': float(request.form.get('PropertyType')),
                'MaturityYear': float(request.form.get('MaturityYear')),
                'FirstPaymentYear': float(request.form.get('FirstPaymentYear')),
                'DTI': float(request.form.get('DTI')),
                'MIP': float(request.form.get('MIP'))
            }
            df = pd.DataFrame([features])
            prediction = pipeline_rf.predict(df)
            return render_template('classification.html', prediction=prediction[0], error=None)
        except Exception as e:
            return render_template('classification.html', prediction=None, error=str(e))

    return render_template('classification.html', prediction=None, error=None)

@app.route('/regression', methods=['GET', 'POST'])
def regression():
    if request.method == 'POST':
        try:
            features = {
                'CreditScore': float(request.form.get('CreditScore')),
                'Units': float(request.form.get('Units')),
                'DTI': float(request.form.get('DTI')),
                'OrigUPB': float(request.form.get('OrigUPB')),
                'OrigInterestRate': float(request.form.get('OrigInterestRate')),
                'MIP': float(request.form.get('MIP')),
                'MonthsDelinquent': float(request.form.get('MonthsDelinquent')),
                'OrigLoanTerm': float(request.form.get('OrigLoanTerm')),
                'MonthsInRepayment': float(request.form.get('MonthsInRepayment')),
                'EMI': float(request.form.get('EMI')),
                'interest_amt': float(request.form.get('interest_amt')),
                'MonthlyIncome': float(request.form.get('MonthlyIncome')),
                'cur_principal': float(request.form.get('cur_principal'))
            }
            df = pd.DataFrame([features])
            prediction = pipeline_lasso.predict(df)
            return render_template('regression.html', prediction=prediction[0], error=None)
        except Exception as e:
            return render_template('regression.html', prediction=None, error=str(e))

    return render_template('regression.html', prediction=None, error=None)

if __name__ == '__main__':
    app.run(debug=True)
