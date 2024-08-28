# create_db.py
import sys
import os

# Ensure the parent directory is in the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import db
from database.models import User
from pymongo.errors import OperationFailure
from werkzeug.security import generate_password_hash

try:
    # Create a sample user with hashed password and default privilege
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
except OperationFailure as e:
    print(f"Operation failed: {e.details['errmsg']}")
