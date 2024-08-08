import os
from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('pages/index.html')


@app.route('/about', methods=['GET'])
def about():
    return render_template('pages/about.html')


@app.route('/authentication', methods=['GET'])
def authentication():
    return render_template('pages/authentication.html')


@app.route('/model', methods=['GET'])
def model():
    return render_template('pages/model.html')


@app.route('/user_dashboard', methods=['GET'])
def user_dashboard():
    return render_template('pages/user_dashboard.html')


@app.route('/forget_password', methods=['GET'])
def forget_password():
    return render_template('pages/forget_password.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
