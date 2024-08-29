# Signsense

This is a machine learning model designed for the recognition of Kenyan Sign Language. This project aims to bridge the communication gap by providing an efficient and accessible tool for understanding and interpreting Kenyan Sign Language.

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/mikemwai/signsense.git
    ```
2. Navigate to the project directory:
    ```
    cd signsense
    ```
3. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Usage

- Run the following command to start the application on your local machine:

  - On Windows:

    ```sh
        set FLASK_APP=app.py
        flask run --host=0.0.0.0
    ```

  - On Unix/Linux/Mac:

    ```sh
        export FLASK_APP=app.py
        flask run
    ```

- This will start a development server on http://127.0.0.1:5000/ where you can access the application.
- You can also access the application, running on the development server, on different devices:
  - Download and install [ngrok](https://ngrok.com/download).
  - Unzip the downloaded file and move the ngrok executable to the project's path.
  - Add the authtoken to authenticate your account:
    ```sh
    ./ngrok authtoken <your_auth_token>
    ```
    
  - Start ngrok:
    ```sh
    ./ngrok http 5000
    ```
    
  - Access the application using the generated URL by ngrok through your phone's web browser.

## Deployment

- The flask application has been deployed using nginx and gunicorn. The deployment process is outlined in the [Deployment.md](Deployment.md) file.

## Contributions

If you'd like to contribute to this project:

- Please fork the repository.
- Create a new branch for your changes.
- Submit a [pull request](https://github.com/mikemwai/signsense/pulls).

Contributions, bug reports, and feature requests are welcome!

## Issues

If you have any issues with the project, feel free to open up an [issue](https://github.com/mikemwai/signsense/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.