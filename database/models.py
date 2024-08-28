# database/models.py
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