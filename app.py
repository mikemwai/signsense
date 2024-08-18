import os
import secrets
from flask import Flask
from pymongo import MongoClient
from database.config import Config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'documents')
app.config.from_object(Config)
app.secret_key = secrets.token_hex(16)  # Set the secret key
client = MongoClient(app.config['MONGO_URI'])
db = client[app.config['MONGO_DB_NAME']]

# Import routes
from routes import routes

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)