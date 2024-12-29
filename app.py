import os
import secrets
from flask import Flask
from pymongo import MongoClient
from database.config import Config
from dotenv import load_dotenv
from utilities.extensions import mail

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SECURITY_PASSWORD_SALT'] = os.getenv('SECURITY_PASSWORD_SALT')
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = os.getenv('MAIL_PORT')
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS')
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config.from_object(Config)
app.secret_key = secrets.token_hex(16)
client = MongoClient(app.config['MONGO_URI'])
db = client[app.config['MONGO_DB_NAME']]

from routes import routes

mail.init_app(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)