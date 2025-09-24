# database/models.py

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin # <-- IMPORT THIS

db = SQLAlchemy()

#                           ADD UserMixin HERE   
#                                   |
#                                   v
class User(db.Model, UserMixin):
    """User model for storing user details."""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    transactions = db.relationship('Transaction', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Transaction(db.Model):
    """Transaction model for storing transaction details."""
    __tablename__ = 'transactions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    transaction_type = db.Column(db.String(20), nullable=False)  # 'UPI' or 'Credit'
    amount = db.Column(db.Float, nullable=False)
    is_fraud = db.Column(db.Boolean, default=False)
    # Storing other features as a JSON string for flexibility
    details = db.Column(db.String(500))

