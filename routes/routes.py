from flask import request, redirect, url_for, render_template, session, jsonify, Response
from bson import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
import os
from authlib.integrations.flask_client import OAuth
from utilities.utils import generate_reset_token, send_reset_email
import itsdangerous
from PIL import Image
import io
import base64
import cv2
from ultralytics import YOLO

from app import app, db

app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')
oauth = OAuth(app)

google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_ID'),
    client_secret=os.getenv('GOOGLE_SECRET'),
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    userinfo_endpoint='https://www.googleapis.com/oauth2/v1/userinfo',
    jwks_uri='https://www.googleapis.com/oauth2/v3/certs',
    client_kwargs={'scope': 'openid profile email'},
)

@app.route('/', methods=['GET'])
def index():
    return render_template('pages/index.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('pages/about.html')

@app.route('/resources', methods=['GET'])
def resources():
    return render_template('pages/resources.html')

######### Model Routes #########
@app.route("/webcam_feed")
def webcam_feed():
    cap = cv2.VideoCapture(0)

    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                break

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            print(type(frame))

            img = Image.open(io.BytesIO(frame))

            model = YOLO("best.pt")
            results = model(img, save=True)

            print(results)
            cv2.waitKey(1)

            res_plotted = results[0].plot()
            # cv2.imshow("result", res_plotted)

            if cv2.waitKey(1) == ord('q'):
                break

            # Convert from BGR to RGB
            img_BGR = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

######### Authentication Routes #########
@app.route('/authentication', methods=['GET', 'POST'])
def authentication():
    notification = session.pop('notification', None)
    if request.method == 'POST':
        pass
    return render_template('pages/authentication/authentication.html', notification=notification)

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = db.users.find_one({"email": email})
        if user:
            token = generate_reset_token(email)
            send_reset_email(email, token)
            session['notification'] = {"type": "success", "message": "Password reset email sent!"}
        else:
            session['notification'] = {"type": "danger", "message": "Email not found!"}
        return redirect(url_for('forgot_password'))
    notification = session.pop('notification', None)
    return render_template('pages/authentication/forgot_password.html', notification=notification)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        serializer = itsdangerous.URLSafeTimedSerializer(app.config['SECRET_KEY'])
        email = serializer.loads(token, salt=app.config['SECURITY_PASSWORD_SALT'], max_age=3600)
    except itsdangerous.SignatureExpired:
        session['notification'] = {"type": "danger", "message": "The reset link is expired"}
        return redirect(url_for('forgot_password'))
    except itsdangerous.BadSignature:
        session['notification'] = {"type": "danger", "message": "Invalid reset link"}
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        db.users.update_one({"email": email}, {"$set": {"password": hashed_password}})
        session['notification'] = {"type": "success", "message": "Password reset successfully!"}
        return redirect(url_for('authentication'))

    return render_template('pages/authentication/reset_password.html', token=token)

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    user = db.users.find_one({"email": email})
    if user:
        if check_password_hash(user['password'], password):
            session['user_email'] = email
            session['user_privilege'] = user['privilege']
            session['notification'] = {"type": "success", "message": "Login successful. Welcome back!"}
            if user['privilege'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('user_dashboard'))
        else:
            return render_template('pages/authentication/authentication.html', error="Invalid password")
    else:
        return render_template('pages/authentication/authentication.html', error="User not found!")

@app.route('/register', methods=['POST'])
def register():
    first_name = request.form['fname']
    last_name = request.form['lname']
    email = request.form['email']
    phone_no = request.form['phone_no']
    gender = request.form['gender']
    password = request.form['password']

    if db.users.find_one({"email": email}):
        return render_template('pages/authentication/authentication.html', error="User already exists!")

    hashed_password = generate_password_hash(password)

    user = {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone_no": phone_no,
        "gender": gender,
        "password": hashed_password,
        "privilege": "user"
    }

    db.users.insert_one(user)

    session['notification'] = {"type": "success", "message": "User registered successfully! Please login to continue."}

    return redirect(url_for('authentication'))

@app.route('/google_login')
def google_login():
    return google.authorize_redirect(url_for('google_callback', _external=True))


@app.route('/google_callback')
def google_callback():
    token = google.authorize_access_token()
    response = google.get('https://www.googleapis.com/oauth2/v2/userinfo')

    if response is None or response.status_code != 200:
        session['notification'] = {"type": "danger", "message": "Failed to fetch user info from Google."}
        return redirect(url_for('authentication'))

    user_info = response.json()
    user_email = user_info.get('email')

    user = db.users.find_one({"email": user_email})

    if user:
        session['user_email'] = user_email
        session['user_privilege'] = user['privilege']
        session['notification'] = {"type": "success", "message": "Google login successful!"}

        if user['privilege'] == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('user_dashboard'))
    else:
        first_name = user_info.get('given_name', '')
        last_name = user_info.get('family_name', '')

        gender = None
        phone_no = None

        password = None

        new_user = {
            "first_name": first_name,
            "last_name": last_name,
            "email": user_email,
            "phone_no": phone_no,
            "gender": gender,
            "password": password,
            "privilege": "user"
        }

        db.users.insert_one(new_user)

        session['user_email'] = user_email
        session['user_privilege'] = "user"
        session['notification'] = {"type": "success", "message": "Registration successful. Welcome to Signsense!"}

        return redirect(url_for('user_dashboard'))

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    session.pop('user_privilege', None)
    session['notification'] = {"type": "success", "message": "You have been logged out successfully."}
    return redirect(url_for('authentication'))

############# User Menu Routes ##################
@app.route('/user_dashboard', methods=['GET'])
def user_dashboard():
    user_email = session.get('user_email')
    user_privilege = session.get('user_privilege')
    if not user_email or user_privilege != 'user':
        return redirect(url_for('authentication'))

    user = db.users.find_one({"email": user_email})
    if not user:
        return redirect(url_for('authentication'))

    notification = session.pop('notification', None)
    return render_template('pages/user_menu/user_dashboard.html', user=user, notification=notification)

@app.route('/model', methods=['GET'])
def model():
    user_email = session.get('user_email')
    if not user_email:
        return redirect(url_for('authentication'))

    user = db.users.find_one({"email": user_email})
    if not user:
        return redirect(url_for('authentication'))

    return render_template('pages/user_menu/model.html', current_user=user)


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    user_email = session.get('user_email')
    if not user_email:
        return redirect(url_for('authentication'))

    user = db.users.find_one({"email": user_email})
    if not user:
        return redirect(url_for('authentication'))

    notification = None
    if request.method == 'POST':
        updated_data = {
            "first_name": request.form.get('fname'),
            "last_name": request.form.get('lname'),
            "email": request.form.get('email'),
            "phone_no": request.form.get('phone'),
            "gender": request.form.get('gender')
        }

        if request.form.get('password'):
            updated_data["password"] = generate_password_hash(request.form.get('password'))

        db.users.update_one({"email": user_email}, {"$set": updated_data})

        session['user_email'] = updated_data['email']

        user = db.users.find_one({"email": updated_data['email']})

        notification = {"type": "success", "message": "Profile updated successfully"}

    return render_template('pages/user_menu/settings.html', current_user=user, notification=notification)

############### Admin Menu Routes ##############
@app.route('/admin_dashboard', methods=['GET'])
def admin_dashboard():
    user_email = session.get('user_email')
    user_privilege = session.get('user_privilege')
    if not user_email or user_privilege != 'admin':
        return redirect(url_for('authentication'))

    user = db.users.find_one({"email": user_email})
    if not user:
        return redirect(url_for('authentication'))

    notification = session.pop('notification', None)
    return render_template('pages/admin_menu/admin_dashboard.html', user=user, notification=notification)

@app.route('/manage_users', methods=['GET'])
def manage_users():
    users = list(db.users.find())
    notification = session.pop('notification', None)
    return render_template('pages/admin_menu/manage_users.html', users=users, notification=notification)

@app.route('/add_user', methods=['POST'])
def add_user():
    first_name = request.form['newFname']
    last_name = request.form['newLname']
    email = request.form['newEmail']
    phone_no = request.form['newPhone']
    gender = request.form['newGender']
    privilege = request.form['newPrivilege']
    password = request.form['newPassword']

    hashed_password = generate_password_hash(password)

    user = {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone_no": phone_no,
        "gender": gender,
        "privilege": privilege,
        "password": hashed_password
    }

    db.users.insert_one(user)

    session['notification'] = {"type": "success", "message": "User added successfully"}

    return redirect(url_for('manage_users'))

@app.route('/edit_user/<user_id>', methods=['POST'])
def edit_user(user_id):
    first_name = request.form['editFname']
    last_name = request.form['editLname']
    email = request.form['editEmail']
    phone_no = request.form['editPhone']
    gender = request.form['editGender']
    privilege = request.form['editPrivilege']
    password = request.form['editPassword']

    update_data = {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone_no": phone_no,
        "gender": gender,
        "privilege": privilege
    }

    if password:
        update_data["password"] = generate_password_hash(password)

    db.users.update_one({"_id": ObjectId(user_id)}, {"$set": update_data})

    session['notification'] = {"type": "success", "message": "User updated successfully"}

    return redirect(url_for('manage_users'))

@app.route('/delete_user/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    result = db.users.delete_one({"_id": ObjectId(user_id)})

    if result.deleted_count == 1:
        session['notification'] = {"type": "success", "message": "User deleted successfully"}
        response = {"type": "success"}
    else:
        session['notification'] = {"type": "danger", "message": "Failed to delete user"}
        response = {"type": "danger"}

    return jsonify(response)

@app.route('/manage_resources', methods=['GET'])
def manage_resources():
    user_email = session.get('user_email')
    user_privilege = session.get('user_privilege')
    if not user_email or user_privilege != 'admin':
        return redirect(url_for('authentication'))

    user = db.users.find_one({"email": user_email})
    if not user:
        return redirect(url_for('authentication'))

    return render_template('pages/admin_menu/manage_resources.html', user=user)

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

############# APIs ##################
@app.route('/api/dashboard_data', methods=['GET'])
def get_dashboard_data():
    total_users = db.users.count_documents({"privilege": "user"})
    total_admins = db.users.count_documents({"privilege": "admin"})
    total_resources = len(os.listdir(os.path.join(app.static_folder, 'documents')))
    gender_distribution = db.users.aggregate([
        {"$group": {"_id": {"$toLower": "$gender"}, "count": {"$sum": 1}}}
    ])

    gender_data = {item['_id'].capitalize(): item['count'] for item in gender_distribution}

    return jsonify({
        'total_users': total_users,
        'total_admins': total_admins,
        'total_resources': total_resources,
        'gender_distribution': gender_data
    })