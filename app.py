import os
from flask import Flask, request, render_template

app = Flask(__name__)


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
    return render_template('pages/user_menu/model.html')


@app.route('/settings', methods=['GET'])
def settings():
    return render_template('pages/user_menu/settings.html')


############### Admin Menu Routes ##############
@app.route('/admin_dashboard', methods=['GET'])
def admin_dashboard():
    return render_template('pages/admin_menu/admin_dashboard.html')


@app.route('/manage_users', methods=['GET'])
def manage_users():
    return render_template('pages/admin_menu/manage_users.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
