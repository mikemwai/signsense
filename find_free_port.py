import socket
from contextlib import closing


def find_free_port(start=5000, end=5100):
    for port in range(start, end):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex(('127.0.0.1', port)) != 0:
                return port
    raise IOError("No free port found in range")


# Example usage
if __name__ == "__main__":
    free_port = find_free_port()
    print(f"Found free port: {free_port}")
    # Now you can start your Flask app on this port
    # flask run --port {free_port}
