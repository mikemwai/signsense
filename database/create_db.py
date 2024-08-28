import sys
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from werkzeug.security import generate_password_hash

# Ensure the parent directory is in the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your User model
from database.models import User

# Load environment variables from .env file
load_dotenv()

# MongoDB connection details from environment variables
MONGO_URI = os.getenv('MONGO_URI')
DATABASE_NAME = os.getenv('DATABASE_NAME')
USERS_COLLECTION = 'users'

def seed_database():
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]

        # Create sample users with hashed passwords
        users = [
            User(
                first_name="John",
                last_name="Doe",
                email="john.doe@example.com",
                phone_no="1234567890",
                gender="male",
                password=generate_password_hash("signsense"),
                privilege="user"
            ),
            User(
                first_name="Jane",
                last_name="Smith",
                email="jane.smith@example.com",
                phone_no="0987654321",
                gender="female",
                password=generate_password_hash("signsense"),
                privilege="admin"
            )
        ]

        # Convert user objects to dictionaries
        user_dicts = [user.to_dict() for user in users]

        # Insert users into the collection
        result = db[USERS_COLLECTION].insert_many(user_dicts)
        print(f"Inserted {len(result.inserted_ids)} users.")

    except OperationFailure as e:
        print(f"Operation failed: {e.details['errmsg']}")
    finally:
        client.close()

if __name__ == "__main__":
    seed_database()
