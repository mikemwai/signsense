# import os
# secret_key = os.urandom(24)
# print(secret_key)

# import os
# password_salt = os.urandom(16)
# print(password_salt)

import smtplib

def test_smtp_connection():
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login('hellomikemwai@gmail.com', 'eqfi sjep ywvr pfxc')
        server.quit()
        print("SMTP connection successful")
    except Exception as e:
        print(f"SMTP connection failed: {e}")

test_smtp_connection()