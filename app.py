# app.py
import os
import secrets
from flask import Flask, request, redirect, url_for, render_template, session
from pymongo import MongoClient
from database.config import Config
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'documents')
app.config.from_object(Config)
app.secret_key = secrets.token_hex(16)  # Set the secret key
client = MongoClient(app.config['MONGO_URI'])
db = client[app.config['MONGO_DB_NAME']]

@app.route('/', methods=['GET'])
def index():
    return render_template('pages/index.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('pages/about.html')

@app.route('/resources', methods=['GET'])
def resources():
    return render_template('pages/resources.html')

######### Authentication Routes #########
@app.route('/authentication', methods=['GET'])
def authentication():
    return render_template('pages/authentication/authentication.html')

@app.route('/forget_password', methods=['GET'])
def forget_password():
    return render_template('pages/authentication/forget_password.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    # Check if the user exists in the database
    user = db.users.find_one({"email": email})
    if user:
        if check_password_hash(user['password'], password):
            # Authentication successful
            session['user_email'] = email
            return redirect(url_for('user_dashboard'))
        else:
            # Password does not match
            return render_template('pages/authentication/authentication.html', error="Invalid password")
    else:
        # User does not exist
        return render_template('pages/authentication/authentication.html', error="User not found")

@app.route('/register', methods=['POST'])
def register():
    first_name = request.form['fname']
    last_name = request.form['lname']
    email = request.form['email']
    phone_no = request.form['phone_no']
    password = request.form['password']

    # Check if the user already exists
    if db.users.find_one({"email": email}):
        return render_template('pages/authentication/authentication.html', error="User already exists")

    # Hash the password
    hashed_password = generate_password_hash(password)

    # Create a new user with default privilege "user"
    user = {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone_no": phone_no,
        "password": hashed_password,
        "privilege": "user"
    }

    # Insert the user into the database
    db.users.insert_one(user)

    # Redirect to the login page
    return redirect(url_for('authentication'))

############# User Menu Routes ##################
@app.route('/user_dashboard', methods=['GET'])
def user_dashboard():
    return render_template('pages/user_menu/user_dashboard.html')

@app.route('/model', methods=['GET'])
def model():
    user_email = session.get('user_email')
    if not user_email:
        return redirect(url_for('authentication'))

    user = db.users.find_one({"email": user_email})
    if not user:
        return redirect(url_for('authentication'))

    is_admin = user.get('privilege') == 'admin'
    return render_template('pages/user_menu/model.html', current_user=user, is_admin=is_admin)

@app.route('/settings', methods=['GET'])
def settings():
    user_email = session.get('user_email')
    if not user_email:
        return redirect(url_for('authentication'))

    user = db.users.find_one({"email": user_email})
    if not user:
        return redirect(url_for('authentication'))

    is_admin = user.get('privilege') == 'admin'
    return render_template('pages/user_menu/settings.html', current_user=user, is_admin=is_admin)

############### Admin Menu Routes ##############
@app.route('/admin_dashboard', methods=['GET'])
def admin_dashboard():
    return render_template('pages/admin_menu/admin_dashboard.html')

@app.route('/manage_users', methods=['GET'])
def manage_users():
    return render_template('pages/admin_menu/manage_users.html')

@app.route('/manage_resources', methods=['GET'])
def manage_resources():
    return render_template('pages/admin_menu/manage_resources.html')

@app.route('/list_resources', methods=['GET'])
def list_resources():
    documents_path = os.path.join(app.static_folder, 'documents')
    resources = os.listdir(documents_path)
    return jsonify(resources)

@app.route('/upload_resource', methods=['POST'])
def upload_resource():
    if 'resourceFile' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['resourceFile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'success': 'File uploaded successfully'}), 200

@app.route('/delete_resource', methods=['POST'])
def delete_resource():
    filename = request.form['filename']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({'success': 'File deleted successfully'}), 200
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)