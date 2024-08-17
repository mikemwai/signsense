# generate_secret_key.py
import secrets

# Generate a random secret key
secret_key = secrets.token_hex(16)
print(f"Your secret key: {secret_key}")