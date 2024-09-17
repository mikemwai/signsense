import itsdangerous
from flask_mail import Message
from flask import url_for
from app import app
from utilities.extensions import mail
import secrets

def generate_reset_token(email):
    serializer = itsdangerous.URLSafeTimedSerializer(app.config['SECRET_KEY'])
    return serializer.dumps(email, salt=app.config['SECURITY_PASSWORD_SALT'])

def send_reset_email(email, token):
    reset_url = url_for('reset_password', token=token, _external=True)
    msg = Message('Password Reset Request', sender='noreply@yourapp.com', recipients=[email])
    msg.body = f'Please click the link to reset your password: {reset_url}'
    mail.send(msg)

def generate_secret_key():
    return secrets.token_hex(16)

def generate_password_salt():
    return secrets.token_hex(16)

if __name__ == "__main__":
    print(generate_password_salt())