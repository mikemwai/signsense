# filepath: /c:/Projects/signsense/database/models.py
from datetime import datetime
import pytz
from bson import ObjectId

class User:
    def __init__(self, first_name, last_name, email, phone_no, gender, password, privilege="user"):
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.phone_no = phone_no
        self.gender = gender
        self.password = password
        self.privilege = privilege

    def to_dict(self):
        return {
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "phone_no": self.phone_no,
            "gender": self.gender,
            "password": self.password,
            "privilege": self.privilege
        }

class UserActivity:
    def __init__(self, user_id, email, activity_type, success, timestamp=None):
        self.user_id = user_id
        self.email = email
        self.activity_type = activity_type
        self.success = success
        self.timestamp = timestamp or datetime.now(pytz.timezone('Africa/Nairobi'))

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "email": self.email,
            "activity_type": self.activity_type,
            "success": self.success,
            "timestamp": self.timestamp
        }

class Resource:
    def __init__(self, filename, resource_id=None, upload_date=None):
        self.resource_id = resource_id or str(ObjectId())
        self.filename = filename
        self.upload_date = upload_date or datetime.now(pytz.timezone('Africa/Nairobi'))

    def to_dict(self):
        return {
            "resource_id": self.resource_id,
            "filename": self.filename,
            "upload_date": self.upload_date
        }