import os
from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


@app.route('/authentication', methods=['GET'])
def authentication():
    return render_template('authentication.html')


@app.route('/model', methods=['GET'])
def model():
    return render_template('model.html')


@app.route('/user_dashboard', methods=['GET'])
def user_dashboard():
    return render_template('user_dashboard.html')


if __name__ == '__main__':
    app.run(debug=True)
