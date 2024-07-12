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


if __name__ == '__main__':
    app.run(debug=True)
