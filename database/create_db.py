# create_db.py
import sys
import os
from flask import Flask
from pymongo import MongoClient
from models import User, UserActivity, Resource, Feedback, ProcessedVideo
from pymongo.errors import OperationFailure
from werkzeug.security import generate_password_hash
from config import Config

# Ensure the parent directory is in the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.config.from_object(Config)

client = MongoClient(app.config['MONGO_URI'])
db = client[app.config['MONGO_DB_NAME']]

try:
    # Create collections by inserting a dummy document and then deleting it
    collections = ['users', 'user_activities', 'resources', 'feedbacks', 'processed_videos']
    for collection in collections:
        db[collection].insert_one({"init": True})
        db[collection].delete_one({"init": True})

    user1 = User(
        first_name="John",
        last_name="Doe",
        email="john.doe@example.com",
        phone_no="1234567890",
        gender="male",
        password=generate_password_hash("signsense"),
        privilege="user"
    )
    
    user2 = User(
        first_name="Jane",
        last_name="Smith",
        email="jane.smith@example.com",
        phone_no="0987654321",
        gender="female",
        password=generate_password_hash("signsense"),
        privilege="admin"
    )
    
    db.users.insert_one(user1.to_dict())
    db.users.insert_one(user2.to_dict())
    print("User inserted successfully.")
    print("Collections created successfully.")
except OperationFailure as e:
    print(f"Operation failed: {e.details['errmsg']}")
