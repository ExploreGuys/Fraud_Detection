# app.py

import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from sklearn.preprocessing import LabelEncoder
import json

from database.models import db, User, Transaction

# --- App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-very-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fraud_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# --- Flask-Login Initialization ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Redirect to login page if user is not authenticated

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- ML Model Loading ---
upi_model = None
credit_model = None
try:
    with open('ml/upi_model.pkl', 'rb') as f:
        upi_model = pickle.load(f)
    with open('ml/credit_model.pkl', 'rb') as f:
        credit_model = pickle.load(f)
except FileNotFoundError:
    print("Warning: One or more model files not found. Please train the models first.")

# --- Helper Functions ---
def preprocess_input(data, transaction_type):
    """Preprocesses input data for prediction."""
    df = pd.DataFrame([data])
    if transaction_type == 'upi':
        # This is a simplified encoding for a single prediction.
        # In a real app, you'd use the same LabelEncoders from training.
        for col in ['payeeId', 'payerId', 'deviceId']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        return df[['amount', 'payeeId', 'payerId', 'deviceId']]
    else: # credit
        for col in ['cardType', 'merchant', 'location', 'device']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        return df[['amount', 'cardType', 'merchant', 'location', 'device']]

# --- CLI Command to Create DB ---
@app.cli.command("init-db")
def init_db_command():
    """Creates the database tables."""
    with app.app_context():
        db.create_all()
    print("Initialized the database.")

# --- Routes ---

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid email or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(email=email).first():
            flash('Email address already exists', 'warning')
            return redirect(url_for('login'))
        
        new_user = User(name=name, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('login.html') # The registration form is on the same page

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', transactions=transactions)

@app.route('/upload')
@login_required
def upload_page():
    return render_template('upload.html')

@app.route('/upload-transactions', methods=['POST'])
@login_required
def upload_transactions():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('upload_page'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('upload_page'))

    if file and file.filename.endswith('.csv'):
        transaction_type = request.form['transaction_type']
        df = pd.read_csv(file)
        
        model = upi_model if transaction_type == 'upi' else credit_model
        if model is None:
            flash(f"The {transaction_type.upper()} model is not loaded.", 'danger')
            return redirect(url_for('upload_page'))
            
        # Preprocess the entire dataframe
        df_processed = df.copy()
        if transaction_type == 'upi':
            features = ['amount', 'payeeId', 'payerId', 'deviceId']
            for col in ['payeeId', 'payerId', 'deviceId']:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
        else:
            features = ['amount', 'cardType', 'merchant', 'location', 'device']
            for col in ['cardType', 'merchant', 'location', 'device']:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
        
        predictions = model.predict(df_processed[features])
        
        # Save to database
        for index, row in df.iterrows():
            details = row.to_dict()
            # Safely remove 'isFraud' if it exists
            details.pop('isFraud', None)
            
            new_transaction = Transaction(
                user_id=current_user.id,
                transaction_type=transaction_type,
                amount=row['amount'],
                is_fraud=bool(predictions[index]),
                details=json.dumps(details)
            )
            db.session.add(new_transaction)
        db.session.commit()
        
        flash('File processed and transactions saved!', 'success')
        return redirect(url_for('dashboard'))

    flash('Invalid file type. Please upload a CSV.', 'danger')
    return redirect(url_for('upload_page'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    data = request.form.to_dict()
    transaction_type = data.pop('transaction_type')
    
    model = upi_model if transaction_type == 'upi' else credit_model
    if model is None:
        flash(f"The {transaction_type.upper()} model is not loaded.", 'danger')
        return redirect(url_for('dashboard'))

    # Ensure amount is float
    try:
        data['amount'] = float(data['amount'])
    except ValueError:
        flash('Invalid amount entered.', 'danger')
        return redirect(url_for('dashboard'))
    
    processed_data = preprocess_input(data, transaction_type)
    prediction = model.predict(processed_data)[0]
    
    # Save the prediction to DB
    new_transaction = Transaction(
        user_id=current_user.id,
        transaction_type=transaction_type,
        amount=data['amount'],
        is_fraud=bool(prediction),
        details=json.dumps(data)
    )
    db.session.add(new_transaction)
    db.session.commit()
    
    result_text = "Fraudulent" if prediction else "Not Fraudulent"
    flash(f'Prediction for the transaction is: {result_text}', 'info' if prediction else 'success')
    return redirect(url_for('dashboard'))

@app.route('/reports')
@login_required # In a real app, you'd add an @admin_required decorator
def reports():
    # UPI Data
    upi_fraud_count = Transaction.query.filter_by(transaction_type='upi', is_fraud=True).count()
    upi_not_fraud_count = Transaction.query.filter_by(transaction_type='upi', is_fraud=False).count()
    upi_fraud_transactions = Transaction.query.filter_by(transaction_type='upi', is_fraud=True).all()
    
    # Credit Card Data
    credit_fraud_count = Transaction.query.filter_by(transaction_type='credit', is_fraud=True).count()
    credit_not_fraud_count = Transaction.query.filter_by(transaction_type='credit', is_fraud=False).count()
    credit_fraud_transactions = Transaction.query.filter_by(transaction_type='credit', is_fraud=True).all()

    upi_chart_data = {"fraud": upi_fraud_count, "not_fraud": upi_not_fraud_count}
    credit_chart_data = {"fraud": credit_fraud_count, "not_fraud": credit_not_fraud_count}
    
    return render_template(
        'reports.html',
        upi_fraud=upi_fraud_transactions,
        credit_fraud=credit_fraud_transactions,
        upi_chart_data=upi_chart_data,
        credit_chart_data=credit_chart_data
    )

