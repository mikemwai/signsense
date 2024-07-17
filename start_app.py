import os
from find_free_port import find_free_port

# Find a free port
free_port = find_free_port()

# Set environment variables
os.environ['FLASK_APP'] = 'app.py'
os.environ['FLASK_RUN_PORT'] = str(free_port)

# Start the Flask application
os.system('flask run')