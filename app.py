import os
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'documents')


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


############# User Menu Routes ##################
@app.route('/user_dashboard', methods=['GET'])
def user_dashboard():
    return render_template('pages/user_menu/user_dashboard.html')


@app.route('/model', methods=['GET'])
def model():
    # Simulate current_user object
    class User:
        def __init__(self, is_admin):
            self.is_admin = is_admin

    current_user = User(is_admin=True)  # Change to False to simulate a regular user
    return render_template('pages/user_menu/model.html', current_user=current_user)


@app.route('/settings')
def settings():
    # Simulate current_user object
    class User:
        def __init__(self, is_admin):
            self.is_admin = is_admin

    current_user = User(is_admin=True)  # Change to False to simulate a regular user
    return render_template('pages/user_menu/settings.html', current_user=current_user)


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
